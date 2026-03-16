#done by Logan Mifflin
#This script builds probe training data from sql generations

# Functions:
#_gpu_mem shows gpu and ram stats
#_log prints timestamped logs
#_show prints short text previews
#_setup_signal_handlers handles stop signals
#_bpe_inclusion_labels marks kept generated tokens
#_save_checkpoint writes mid run checkpoints
#parse_args reads cli args
#_make_client builds a model client by name
#_load_models loads base and fixer clients
#_load_resume_state restores pca and saved arrays
#main loads data and starts collection
#_collect runs generation fixing labeling pca and save

import argparse, sys, signal, traceback, time, resource
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

from dotenv import load_dotenv
import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.consistency_scoring import (
    sql_prompt, postprocess_sql, quote_unquoted_identifiers, tokenize_sql,
)
from src.llm_client import LLMClient, GemmaClient, TextToSQLClient, ArcticText2SQLClient
from src.fixer import fix_sql
from src.loader import load_mini_dev, load_bird_file, exec_sql, results_match

#show memory usage
def _gpu_mem() -> str:
    parts = []
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc   = torch.cuda.memory_allocated(i) / 1024**3
                reserved= torch.cuda.memory_reserved(i)  / 1024**3
                total   = torch.cuda.get_device_properties(i).total_memory / 1024**3
                parts.append(f"GPU{i}: {alloc:.1f}/{total:.0f}GB (res {reserved:.1f}GB)")
    except Exception as e:
        parts.append(f"GPU err: {e}")
    try:
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        parts.append(f"RAM: {rss_gb:.1f}GB")
    except Exception:
        pass
    return "  |  ".join(parts) if parts else "unavailable"


#print one log line
def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)


#print short text preview
def _show(label: str, text: str, max_lines: int = 12, max_cols: int = 120) -> None:
    lines = (text or "").strip().splitlines()
    truncated = len(lines) > max_lines
    for ln in lines[:max_lines]:
        if len(ln) > max_cols:
            ln = ln[:max_cols] + " …"
    if truncated:
        print(f"│({len(lines) - max_lines} more lines)")

#handle kill signals
def _setup_signal_handlers():
    def _handler(signum, frame):
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        msg = f"SIGNAL {name} — likely SLURM OOM/timeout\nMem: {_gpu_mem()}"
        print(msg, flush=True)
        sys.stderr.write(msg); sys.stderr.flush()
        traceback.print_stack(frame, file=sys.stdout)
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2, signal.SIGXCPU):
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            pass

#build token labels by bpe alignment
def _bpe_inclusion_labels(
    gen_token_strs: list[str],
    fix_sql_str: str,
    client,
) -> list[int]:

    try:
        fixed_ids = client._tokenizer.encode(fix_sql_str, add_special_tokens=False)
        fixed_tok_strs = [
            client._tokenizer.decode([t], skip_special_tokens=False)
            for t in fixed_ids
        ]
    except Exception:
        fixed_tok_strs = fix_sql_str.split()

    flags = [0] * len(gen_token_strs)
    matcher = SequenceMatcher(None, gen_token_strs, fixed_tok_strs, autojunk=False)
    opcodes = matcher.get_opcodes()

    for op_idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            has_prior_insert = (
                op_idx > 0 and opcodes[op_idx - 1][0] == "insert"
            )
            lbl = 0 if has_prior_insert else 1
            for i in range(i1, i2):
                flags[i] = lbl

    return flags

#save current checkpoint
def _save_checkpoint(
    raw_vecs: np.ndarray,
    labels_arr: np.ndarray,
    n_total: int,
    pca: "IncrementalPCA",
    args,
    out_path: Path,
    attempted: int,
    total: int,
) -> None:
    try:
        hidden_dim = raw_vecs.shape[1]
        CHUNK = 2048
        parts = []
        for start in range(0, n_total, CHUNK):
            parts.append(pca.transform(raw_vecs[start : start + CHUNK]))
        reduced = np.concatenate(parts, axis=0)

        ckpt = {
            "hidden_states":  torch.from_numpy(reduced.astype(np.float32)),
            "labels":         torch.from_numpy(labels_arr[:n_total].astype(np.float32)),
            "pca_mean":       torch.from_numpy(pca.mean_.astype(np.float32)),
            "pca_components": torch.from_numpy(pca.components_.astype(np.float32)),
            "input_dim":      hidden_dim,
            "proj_dim":       int(pca.n_components_),
            "dataset":        args.dataset,
            "base_model":     args.base_model,
            "n_examples":     attempted,
            "n_tokens":       n_total,
        }
        tmp = out_path.with_suffix(".tmp.pt")
        torch.save(ckpt, tmp)
        import os; os.replace(tmp, out_path)
        _log(f"Checkpoint saved -> {out_path}  "
             f"({n_total} tokens / {attempted}/{total} examples)")
    except Exception as exc:
        _log(f"WARNING: checkpoint save failed: {exc}")

#read cli args
def parse_args():
    p = argparse.ArgumentParser(
        description="Collect whitebox probe training data from BIRD SQL datasets"
    )
    p.add_argument("--dataset", choices=["mini_dev", "bird_json"], default="mini_dev")
    p.add_argument("--path", default=None,
                   help="Path to BIRD JSON/JSONL (required for bird_json)")
    p.add_argument("--limit", type=int, default=5,
                   help="Examples to process (0 = all)")
    p.add_argument("--base_model", default="arctic",
                   choices=["gemma", "qwen", "text2sql", "arctic"])
    p.add_argument("--fixer_model", default="qwen",
                   choices=["qwen", "gemma", "text2sql", "arctic"])
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--proj_dim", type=int, default=64,
                   help="PCA output dimension (default: 64)")
    p.add_argument("--pca_batch", type=int, default=256,
                   help="Number of examples to accumulate before partial_fit PCA (default: 256)")
    p.add_argument("--output", default=None,
                   help="Output .pt path (default: outputs/probe_train_<dataset>_<ts>.pt)")
    p.add_argument("--skip", type=int, default=0,
                   help="Skip first N examples from the dataset (use with --resume to continue a run)")
    p.add_argument("--resume", default=None,
                   help="Path to an existing .pt checkpoint; rehydrates PCA and pre-populates "
                        "accumulated vectors so the new run continues from where it left off.")
    p.add_argument("--sql_timeout", type=float, default=30.0,
                   help="Max seconds allowed for each SQL execution call (default: 30)")
    return p.parse_args()

#build a client from model name
def _make_client(name: str):
    if name == "arctic":
        return ArcticText2SQLClient()
    elif name == "text2sql":
        return TextToSQLClient()
    elif name == "gemma":
        return GemmaClient()
    else:
        return LLMClient()

#load base and fixer models
def _load_models(args):
    print("\nLoading models", flush=True)
    print(f"Memory before: {_gpu_mem()}", flush=True)

    if args.base_model == args.fixer_model:
        print(f"Sharing one model instance for base+fixer ({args.base_model})", flush=True)
        base = _make_client(args.base_model)
        fixer = base
    elif args.base_model == "gemma" or (
        args.base_model in ("text2sql", "arctic") and
        args.fixer_model not in ("text2sql", "arctic")
    ):
        fixer = _make_client(args.fixer_model)
        print(f"Memory after fixer ({args.fixer_model}): {_gpu_mem()}", flush=True)
        base  = _make_client(args.base_model)
    else:
        base  = _make_client(args.base_model)
        print(f"Memory after base ({args.base_model}): {_gpu_mem()}", flush=True)
        fixer = _make_client(args.fixer_model)

    print(f"Models loaded. Memory: {_gpu_mem()}", flush=True)
    return base, fixer

#load resume checkpoint state
def _load_resume_state(resume_path: str, proj_dim: int) -> dict:
    print(f"[resume] Loading checkpoint: {resume_path}", flush=True)
    ckpt = torch.load(resume_path, map_location="cpu")

    saved_proj  = int(ckpt.get("proj_dim", proj_dim))
    n_tokens    = int(ckpt["n_tokens"])
    hidden_dim  = int(ckpt["input_dim"])
    pca_mean    = ckpt["pca_mean"].numpy().astype(np.float32)
    pca_comp    = ckpt["pca_components"].numpy().astype(np.float32)
    reduced     = ckpt["hidden_states"].numpy().astype(np.float32)[:n_tokens]
    labels_saved = ckpt["labels"].numpy().astype(np.int8)

    pca = IncrementalPCA(n_components=saved_proj)
    pca.mean_            = pca_mean
    pca.components_      = pca_comp
    pca.n_samples_seen_  = n_tokens
    pca.n_components_    = saved_proj
    pca.noise_variance_  = 0.0

    raw_approx = (reduced @ pca_comp + pca_mean).astype(np.float32)

    INIT_CAP = max(8192, n_tokens * 2)
    raw_vecs   = np.empty((INIT_CAP, hidden_dim), dtype=np.float32)
    labels_arr = np.empty((INIT_CAP,),            dtype=np.int8)
    raw_vecs  [:n_tokens] = raw_approx
    labels_arr[:n_tokens] = labels_saved

    print(f"[resume] Pre-loaded {n_tokens} tokens  |  hidden_dim={hidden_dim}  proj={saved_proj}",
          flush=True)
    return {
        "pca":        pca,
        "raw_vecs":   raw_vecs,
        "labels_arr": labels_arr,
        "n_total":    n_tokens,
        "_cap":       INIT_CAP,
        "hidden_dim": hidden_dim,
    }

#load data then start collection
def main():
    load_dotenv()
    _setup_signal_handlers()
    args = parse_args()

    #load dataset examples
    if args.dataset == "mini_dev":
        examples = load_mini_dev(limit=args.limit or None)
    else:
        if not args.path:
            print("Error: --path required for bird_json"); sys.exit(1)
        examples = load_bird_file(args.path, limit=args.limit or None)
    print(f"Loaded {len(examples)} examples  [{args.dataset}]", flush=True)

    #optional skip for resume workflows
    if args.skip > 0:
        if args.skip >= len(examples):
            print(f"Error: --skip {args.skip} >= total examples {len(examples)}"); sys.exit(1)
        print(f"Skipping first {args.skip} examples (--skip {args.skip})", flush=True)
        examples = examples[args.skip:]

    base_client, fixer_client = _load_models(args)

    #choose output path
    if args.output:
        out_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"whitebox_data_{timestamp}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    #optional resume state
    initial_state = None
    if args.resume and Path(args.resume).exists():
        if args.skip == 0:
            peek = torch.load(args.resume, map_location="cpu")
            args.skip = int(peek.get("n_examples", 0))
            del peek
            print(f"[resume] Auto-skip: {args.skip} examples (from checkpoint n_examples)",
                  flush=True)
        initial_state = _load_resume_state(args.resume, args.proj_dim)

    _collect(base_client, fixer_client, examples, args, out_path,
             initial_state=initial_state)

#collect hidden states and labels for probe training
def _collect(base_client, fixer_client, examples, args, out_path: Path,
             initial_state: dict | None = None):
    #restore arrays if resuming
    if initial_state:
        pca        = initial_state["pca"]
        raw_vecs   = initial_state["raw_vecs"]
        labels_arr = initial_state["labels_arr"]
        n_total    = initial_state["n_total"]
        _cap       = initial_state["_cap"]
        hidden_dim = initial_state["hidden_dim"]
        print(f"[resume] Continuing from {n_total} pre-loaded tokens", flush=True)
    else:
        pca = IncrementalPCA(n_components=args.proj_dim)
        INIT_CAP = 4096
        _cap = INIT_CAP
        raw_vecs:   np.ndarray = np.empty((_cap, 1), dtype=np.float32)
        labels_arr: np.ndarray = np.empty((_cap,),   dtype=np.int8)
        n_total    = 0
        hidden_dim = None

    #buffers for incremental pca
    pca_buf: list[np.ndarray] = []
    pca_buf_n = 0
    PCA_BATCH = args.pca_batch

    n_passed = n_fix_passed = n_errors = 0

    for i, ex in enumerate(examples, 1):
        #process one example
        db_id    = ex.get("db_id", "")
        ex_id    = ex.get("example_id", "")
        t0       = time.time()
        print(f"\n[{i}/{len(examples)}] {db_id} | {ex['question'][:80]}", flush=True)
        _log(f"START  mem: {_gpu_mem()}")
        if ex.get("gold_sql"):
            _log(f"  gold: {ex['gold_sql'][:120]}")

        try:
            #build prompt and generate
            prompt = sql_prompt(
                ex["question"], db_id,
                ex.get("schema"), ex.get("evidence"),
            )

            _log("step 1: generate_with_full_logprobs (base)…")
            raw_text, _, full_logprobs, hidden_states = \
                base_client.generate_with_full_logprobs(
                    prompt, max_tokens=args.max_tokens, return_hidden_states=True
                )

            clean_sql = postprocess_sql(raw_text)
            clean_sql = quote_unquoted_identifiers(clean_sql, ex.get("schema", ""))
            gen_tok_strs = [tok for tok, _ in full_logprobs]
            n_gen = len(gen_tok_strs)
            _log(f"step 1 done: {n_gen} generated tokens | hs {tuple(hidden_states.shape) if hidden_states is not None else 'None'}")
            _log(f"mem: {_gpu_mem()}")
            _show(f"base ({args.base_model})", clean_sql)

            #skip bad hidden state batches
            if hidden_states is None or hidden_states.ndim != 2 or hidden_states.shape[0] == 0:
                _log("step 1: no hidden states — skipping example")
                continue

            if hidden_dim is None:
                hidden_dim = hidden_states.shape[1]
                raw_vecs = np.empty((_cap, hidden_dim), dtype=np.float32)

            assert hidden_states.shape[1] == hidden_dim, (
                f"hidden_dim changed: expected {hidden_dim}, got {hidden_states.shape[1]}"
            )

            #skip truncated generations
            if n_gen >= args.max_tokens:
                _log(f"step 1: SQL truncated at token limit ({n_gen} >= {args.max_tokens}) — skipping")
                del hidden_states
                torch.cuda.empty_cache()
                continue

            hs_np = hidden_states.float().cpu().numpy()
            del hidden_states
            torch.cuda.empty_cache()

            n_gen = min(n_gen, hs_np.shape[0])
            hs_np = hs_np[:n_gen]

            #evaluate against gold (which is the main oracle for pass/fail)
            _log("step 2: evaluate gold…")
            _log(f"mem before eval: {_gpu_mem()}")
            gold     = (ex.get("gold_sql") or "").strip()
            db_path  = ex.get("db_path")
            if gold and db_path and Path(db_path).exists():
                gold_rows, gold_err = exec_sql(db_path, gold, timeout=args.sql_timeout)
                pred_rows, pred_err = exec_sql(db_path, clean_sql, timeout=args.sql_timeout)
                if pred_err:
                    passed    = False
                    error_msg = f"SQL error: {pred_err}"
                elif gold_err:
                    passed    = tokenize_sql(clean_sql.lower()) == tokenize_sql(gold.lower())
                    error_msg = "" if passed else "string mismatch"
                else:
                    passed    = results_match(gold_rows, pred_rows)
                    error_msg = "" if passed else "row mismatch"
            elif gold:
                passed    = tokenize_sql(clean_sql.lower()) == tokenize_sql(gold.lower())
                error_msg = "" if passed else "string mismatch"
            else:
                passed, error_msg = False, "no gold_sql"
            _log(f"step 2 done: passed={passed}  err={error_msg[:80] if error_msg else ''}")

            fix_sql_str = ""
            fix_passed  = False
            if not passed:
                #try one fix pass
                _log(f"step 3: fixing with {args.fixer_model}…")
                _log(f"mem before fix: {_gpu_mem()}")
                fix_sql_str, fix_passed, fix_tries, fix_err = fix_sql(
                    fixer_client, clean_sql, error_msg, ex, args.fixer_model,
                    max_tokens=args.max_tokens, max_attempts=1,
                )
                if fix_sql_str:
                    tag = "FIXED" if fix_passed else "DISCARDED"
                    _show(f"fixer ({args.fixer_model}) → {tag}", fix_sql_str)
                _log(f"step 3 done: fix_passed={fix_passed}, tries={fix_tries}, last='{fix_err[:80]}'")
                _log(f"mem after fix: {_gpu_mem()}")

            #build token labels
            if passed:
                tok_labels = [1] * n_gen
            elif fix_sql_str and fix_passed:
                tok_labels = _bpe_inclusion_labels(gen_tok_strs[:n_gen], fix_sql_str, base_client)
            else:
                _log("step 4: both solver and fixer failed — skipping example")
                n_errors += 1
                continue
            _log(f"step 4 done: {sum(tok_labels)}/{n_gen} tokens kept ({100*sum(tok_labels)//max(n_gen,1)}%)")

            labels_np = np.array(tok_labels, dtype=np.int8)

            #grow arrays if needed
            if n_total + n_gen > _cap:
                new_cap = max(_cap * 2, n_total + n_gen)
                new_raw = np.empty((new_cap, hidden_dim), dtype=np.float32)
                new_raw[:n_total] = raw_vecs[:n_total]
                raw_vecs = new_raw
                new_lbl = np.empty((new_cap,), dtype=np.int8)
                new_lbl[:n_total] = labels_arr[:n_total]
                labels_arr = new_lbl
                _cap = new_cap

            raw_vecs  [n_total : n_total + n_gen] = hs_np
            labels_arr[n_total : n_total + n_gen] = labels_np
            n_total += n_gen

            #update pca in mini batches
            pca_buf.append(hs_np)
            pca_buf_n += n_gen
            if pca_buf_n >= PCA_BATCH:
                batch = np.concatenate(pca_buf, axis=0)
                pca.partial_fit(batch)
                _log(f"PCA partial_fit: {pca_buf_n} rows, seen={pca.n_samples_seen_}")
                pca_buf   = []
                pca_buf_n = 0

            status  = "PASS" if passed else ("FIXED" if fix_passed else "FAIL")
            elapsed = time.time() - t0
            print(f"-> {status}  ({elapsed:.1f}s)  [total tokens: {n_total}]", flush=True)
            if passed:       
                n_passed += 1
            elif fix_passed: 
                n_fix_passed += 1

            #periodic checkpoint
            if i % 20 == 0 and n_total > 0 and pca.n_samples_seen_ >= args.proj_dim:
                _save_checkpoint(raw_vecs, labels_arr, n_total, pca, args, out_path, i, len(examples))

        except Exception as exc:
            #keep loop running on errors
            n_errors += 1
            elapsed = time.time() - t0
            print(f"ERROR on {db_id}/{ex_id} after {elapsed:.1f}s: {exc}", flush=True)
            traceback.print_exc()
            try: torch.cuda.empty_cache()
            except Exception: pass
            continue

            #flush remaining pca buffer
    if pca_buf:
        batch = np.concatenate(pca_buf, axis=0)
        pca.partial_fit(batch)
        _log(f"PCA final partial_fit: {pca_buf_n} rows, seen={pca.n_samples_seen_}")
        pca_buf = []

    raw_vecs   = raw_vecs  [:n_total]
    labels_arr = labels_arr[:n_total]

    #final run summary
    total = len(examples)
    print(f"\nCollection complete: {n_total} tokens from {total} examples", flush=True)
    print(f"Pass rate (base): {n_passed}/{total}", flush=True)
    if n_fix_passed:
        print(f"Pass rate (fixed): {n_passed+n_fix_passed}/{total}", flush=True)
    if n_errors:
        print(f"Errors: {n_errors}/{total}", flush=True)

    if n_total == 0:
        print("ERROR: no training data collected.", flush=True); sys.exit(1)

    #fit fallback pca if too few samples
    if pca.n_samples_seen_ < args.proj_dim:
        print(f"WARNING: {pca.n_samples_seen_} samples < proj_dim={args.proj_dim}. "
              f"Reducing proj_dim to {pca.n_samples_seen_}.", flush=True)
        args.proj_dim = pca.n_samples_seen_
        pca = IncrementalPCA(n_components=args.proj_dim)
        pca.fit(raw_vecs)

    #apply final pca transform
    print(f"\nApplying PCA ({hidden_dim}d → {args.proj_dim}d)…", flush=True)
    CHUNK = 2048
    reduced_parts: list[np.ndarray] = []
    for start in range(0, n_total, CHUNK):
        reduced_parts.append(pca.transform(raw_vecs[start : start + CHUNK]))
    reduced = np.concatenate(reduced_parts, axis=0)

    hidden_states_t = torch.from_numpy(reduced.astype(np.float32))
    labels_t        = torch.from_numpy(labels_arr.astype(np.float32))
    pca_mean_t      = torch.from_numpy(pca.mean_.astype(np.float32))
    pca_comp_t      = torch.from_numpy(pca.components_.astype(np.float32)) # (proj_dim, hidden_dim)

    #save final training checkpoint
    checkpoint = {
        "hidden_states":  hidden_states_t,
        "labels":         labels_t,
        "pca_mean":       pca_mean_t,
        "pca_components": pca_comp_t,
        "input_dim":      hidden_dim,
        "proj_dim":       args.proj_dim,
        "dataset":        args.dataset,
        "base_model":     args.base_model,
        "n_examples":     total,
        "n_tokens":       n_total,
    }
    torch.save(checkpoint, out_path)

    n_pos = int(labels_t.sum().item())
    n_neg = n_total - n_pos
    print(f"\nSaved -> {out_path}", flush=True)
    print(f"{n_total} tokens  |  {n_pos} correct / {n_neg} incorrect", flush=True)
    print(f"Shape: {tuple(hidden_states_t.shape)}  |  {hidden_dim}d -> {args.proj_dim}d", flush=True)


if __name__ == "__main__":
    main()
