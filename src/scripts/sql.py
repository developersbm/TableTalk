#done by Sebastian Bastida Marin and Rei Shindo
# This script runs sql scoring and writes json outputs

# Functions:
#_gpu_mem shows gpu and ram stats
#_log prints timestamped logs
#_setup_signal_handlers handles stop signals
#_show prints a short text preview
#parse_args reads cli args
#main loads data models and optional probe
#_run_combined runs the full scoring loop

import argparse, json, os, sys, math, signal, traceback, time, resource
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.consistency_scoring import (
    generate_samples, compute_consistency,
    sql_prompt, postprocess_sql, quote_unquoted_identifiers, tokenize_sql,
)
from src.llm_client import LLMClient, GemmaClient, TextToSQLClient, ArcticText2SQLClient
from src.verbalized_confidence import score_sql_example as verbalized_score_sql_example
from src.model_tokenizing import load_tokenizer
from src.tokenize_confidence import aggregate_line_min
from src.fixer import fix_sql
from src.loader import load_mini_dev, load_bird_file, exec_sql, results_match
from src.inclusion import token_inclusion
from src.whitebox_probing import WhiteBoxProbe



#show memory usage text
def _gpu_mem() -> str:
    parts = []
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                parts.append(f"GPU{i}: {alloc:.1f}/{total:.0f}GB (res {reserved:.1f}GB)")
    except Exception as e:
        parts.append(f"GPU stats error: {e}")
    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_gb = rss_kb / (1024**2 if sys.platform == "darwin" else 1024**2)
        parts.append(f"RAM_RSS: {rss_gb:.1f}GB")
    except Exception:
        pass
    return "|".join(parts) if parts else "stats unavailable"


#print one log line
def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


#handle kill signals
def _setup_signal_handlers():
    def _handler(signum, frame):
        sys.stderr.flush()
        traceback.print_stack(frame, file=sys.stdout)
        sys.stdout.flush()
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2, signal.SIGXCPU):
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            pass



#print short text preview
def _show(label: str, text: str, max_lines: int = 12, max_cols: int = 220) -> None:
    lines = (text or "").strip().splitlines()
    truncated = len(lines) > max_lines
    for ln in lines[:max_lines]:
        if len(ln) > max_cols:
            ln = ln[:max_cols] + "..."
    if truncated:
        print(f"({len(lines) - max_lines} more lines)")


#read cli args
def parse_args():
    p = argparse.ArgumentParser(
        description="Combined SQL confidence scoring (BIRD datasets)"
    )
    p.add_argument("--dataset", choices=["mini_dev", "bird_json"], default="mini_dev")
    p.add_argument("--path", default=None,
                   help="Path to BIRD JSON/JSONL (required for bird_json)")
    p.add_argument("--limit", type=int, default=5,
                   help="Examples to process (default: 5, use 0 for all)")
    p.add_argument("--k", type=int, default=5,
                   help="Stochastic samples for consistency scoring")
    p.add_argument("--base_model", default="arctic",
                   choices=["gemma", "qwen", "text2sql", "arctic"],
                   help="Base generator: 'arctic' (default), 'text2sql', 'gemma', or 'qwen'")
    p.add_argument("--fixer_model", default="qwen",
                   choices=["qwen", "gemma", "text2sql", "arctic"],
                   help="Fixer model: 'qwen' (default), 'gemma', 'text2sql', or 'arctic'")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--output", default=None,
                   help="Output JSONL path (default: outputs/combined_sql_<dataset>_<ts>.jsonl)")
    p.add_argument("--probe_path", default=None,
                   help="Path to trained WhiteBoxProbe (optional).")
    return p.parse_args()


#load data and models then start run
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

    print(f"\nLoading models", flush=True)
    print(f"Memory before model loading: {_gpu_mem()}", flush=True)

    #build client from model name
    def _make_client(name: str):
        if name == "arctic":
            return ArcticText2SQLClient()
        elif name == "text2sql":
            return TextToSQLClient()
        elif name == "gemma":
            return GemmaClient()
        else:
            return LLMClient()

    if args.base_model == args.fixer_model:
        print(f"Base and fixer are both '{args.base_model}' — sharing one model instance.",
              flush=True)
        base_client = _make_client(args.base_model)
        fixer_client = base_client
        print(f"Memory after {args.base_model}: {_gpu_mem()}", flush=True)
    elif args.base_model == "gemma" or (args.base_model in ("text2sql", "arctic") and args.fixer_model not in ("text2sql", "arctic")):
        fixer_client = _make_client(args.fixer_model)
        print(f"Memory after {args.fixer_model}: {_gpu_mem()}", flush=True)
        base_client = _make_client(args.base_model)
        print(f"Memory after {args.base_model}: {_gpu_mem()}", flush=True)
    else:
        base_client = _make_client(args.base_model)
        print(f"Memory after {args.base_model}: {_gpu_mem()}", flush=True)
        fixer_client = _make_client(args.fixer_model)
        print(f"Memory after {args.fixer_model}: {_gpu_mem()}", flush=True)

    tokenizer = load_tokenizer()
    print(f"Models loaded. Memory: {_gpu_mem()}", flush=True)

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent.parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"combined_sql_{args.dataset}_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    probe = None
    if args.probe_path and os.path.exists(args.probe_path):
        probe = WhiteBoxProbe.load(args.probe_path, device='cpu')
        probe.eval()
        print(f"Loaded probe from {args.probe_path}", flush=True)

    _run_combined(base_client, fixer_client, examples, args, out_path, tokenizer, probe)


#run full sql pipeline and save outputs
def _run_combined(base_client, fixer_client, examples, args, out_path, tokenizer, probe=None):
    n_passed = n_fix_passed = n_errors = 0
    all_records = []

    jsonl_fh = open(out_path, "w", encoding="utf-8")

    for i, ex in enumerate(examples, 1):
        #process one example
        example_id = ex.get("example_id", "")
        db_id = ex.get("db_id", "")
        t0 = time.time()
        print(f"\n[{i}/{len(examples)}] {db_id} | {ex['question'][:60]}", flush=True)
        _log(f"START mem: {_gpu_mem()}")

        try:
            #build sql prompt
            prompt = sql_prompt(
                ex["question"], db_id,
                ex.get("schema"), ex.get("evidence"),
            )

            #generate sql and token scores
            _log("step 1: generate_with_full_logprobs (base)…")
            need_hidden = probe is not None
            if need_hidden:
                raw_text, ref_lp, full_logprobs, raw_hidden_states = \
                    base_client.generate_with_full_logprobs(
                        prompt, max_tokens=args.max_tokens, return_hidden_states=True
                    )
                with torch.no_grad():
                    hidden_states = probe.apply_pca(raw_hidden_states) \
                        if raw_hidden_states is not None else None
                del raw_hidden_states
                torch.cuda.empty_cache()
            else:
                raw_text, ref_lp, full_logprobs = base_client.generate_with_full_logprobs(
                    prompt, max_tokens=args.max_tokens
                )
                hidden_states = None
            clean_sql = postprocess_sql(raw_text)
            clean_sql = quote_unquoted_identifiers(clean_sql, ex.get("schema", ""))
            sql_tokens = tokenize_sql(clean_sql)
            _log(f"step 1 done: {len(full_logprobs)} tokens, {len(clean_sql)} chars")
            _log(f"mem: {_gpu_mem()}")

            #check against gold sql (which is the main oracle for pass/fail)
            _log("step 2: evaluate against gold…")
            gold = (ex.get("gold_sql") or "").strip()
            db_path = ex.get("db_path")
            if gold and db_path and Path(db_path).exists():
                gold_rows, gold_err = exec_sql(db_path, gold)
                pred_rows, pred_err = exec_sql(db_path, clean_sql)
                if pred_err:
                    passed = False
                    error_msg = f"SQL execution error: {pred_err}"
                elif gold_err:
                    passed = tokenize_sql(clean_sql.lower()) == tokenize_sql(gold.lower())
                    error_msg = "" if passed else "SQL does not match gold"
                else:
                    passed = results_match(gold_rows, pred_rows)
                    error_msg = "" if passed else "Result set does not match expected output"
            elif gold:
                passed = tokenize_sql(clean_sql.lower()) == tokenize_sql(gold.lower())
                error_msg = "" if passed else "SQL does not match gold"
            else:
                passed, error_msg = False, "no gold_sql available"
            _log(f"step 2 done: passed={passed}")

            SQL_BASE_RATE = 0.850

            if probe is not None:
                #fill defaults for non whitebox methods
                _log("step 3: whitebox-only mode — skipping tokenized/verbalized/consistency")
                token_probs    = [SQL_BASE_RATE] * len(full_logprobs)
                line_tok_confs = []
                verb_line_confs   = {}
                consist_line_confs = {}
            else:
                #token confidence
                _log("step 3a: tokenized confidence…")
                token_probs = [math.exp(lp) for _, lp in full_logprobs]
                line_tok_map = aggregate_line_min(clean_sql, full_logprobs)
                line_tok_confs = list(line_tok_map.values())
                _log(f"step 3a done")

                #verbalized confidence
                _log("step 3b: verbalized confidence…")
                _log(f"mem before verbalized: {_gpu_mem()}")
                verb_result = verbalized_score_sql_example(
                    client=base_client, example=ex,
                    max_tokens=args.max_tokens,
                    generated_sql=clean_sql,
                    check_result=(passed, error_msg),
                )
                verb_line_confs = {}
                try:
                    verb_line_confs = json.loads(verb_result.get("line_confs_json", "{}"))
                    verb_line_confs = {int(k): float(v) for k, v in verb_line_confs.items()}
                except (json.JSONDecodeError, ValueError):
                    pass
                _log(f"step 3b done: {len(verb_line_confs)} lines scored")
                _log(f"mem after verbalized: {_gpu_mem()}")

                #consistency confidence
                _log(f"step 3c: consistency confidence (k={args.k})…")
                _log(f"mem before consistency: {_gpu_mem()}")
                raw_samples = generate_samples(
                    base_client, prompt, k=args.k,
                    temperature=args.temperature, top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )
                schema_str = ex.get("schema", "")
                samples = [quote_unquoted_identifiers(postprocess_sql(t), schema_str) for t, _ in raw_samples]
                consist_m = compute_consistency(clean_sql, samples, mode="sql")
                consist_line_confs = {}
                try:
                    consist_line_confs = json.loads(consist_m.get("line_scores_json", "{}"))
                    consist_line_confs = {int(k): float(v) for k, v in consist_line_confs.items()}
                except (json.JSONDecodeError, ValueError):
                    pass
                _log(f"step 3c done: {len(consist_line_confs)} lines scored")
                _log(f"mem after consistency: {_gpu_mem()}")

            #whitebox confidence
            _log("step 3d: white-box probing confidence…")
            wb_token_confs = {}
            wb_line_confs  = {}

            if probe is not None and hidden_states is not None and hidden_states.size(0) > 0:
                with torch.no_grad():
                    gen_tok_strs = [t for t, _ in full_logprobs]
                    gen_text_concat = "".join(gen_tok_strs)

                    gen_starts: list[int] = []
                    _pos = 0
                    for _gt in gen_tok_strs:
                        gen_starts.append(_pos)
                        _pos += len(_gt)

                    sql_text_start = gen_text_concat.find(clean_sql) if clean_sql else -1
                    search_from = max(0, sql_text_start)

                    for t_idx, sql_tok in enumerate(sql_tokens):
                        if not sql_tok:
                            wb_token_confs[t_idx] = 0.5
                            continue
                        char_pos = gen_text_concat.find(sql_tok, search_from)
                        if char_pos == -1:
                            char_pos = gen_text_concat.find(sql_tok)
                        if char_pos == -1:
                            wb_token_confs[t_idx] = 0.5
                            continue
                        char_end = char_pos + len(sql_tok)
                        search_from = char_pos + 1

                        indices = [
                            gi for gi, gs in enumerate(gen_starts)
                            if max(gs, char_pos) < min(gs + len(gen_tok_strs[gi]), char_end)
                            and gi < hidden_states.size(0)
                        ]
                        if not indices:
                            wb_token_confs[t_idx] = 0.5
                            continue

                        span_hs = hidden_states[indices]
                        conf = probe(span_hs, apply_platt=True).item()
                        wb_token_confs[t_idx] = conf

                    _ci = 0
                    line_tok_indices: dict[int, list[int]] = {}
                    for t_idx, sql_tok in enumerate(sql_tokens):
                        _p = clean_sql.find(sql_tok, _ci) if clean_sql else -1
                        _ln = clean_sql[:_p].count("\n") if _p >= 0 else 0
                        if _p >= 0:
                            _ci = _p + len(sql_tok)
                        line_tok_indices.setdefault(_ln, []).append(t_idx)

                    _sql_lines_wb = clean_sql.splitlines() if clean_sql else []
                    for ln_idx in range(len(_sql_lines_wb)):
                        toks = line_tok_indices.get(ln_idx, [])
                        wb_line_confs[ln_idx] = (
                            sum(wb_token_confs.get(ti, 0.5) for ti in toks) / len(toks)
                            if toks else 0.5
                        )
            _log(f"step 3d done: {len(wb_token_confs)} tok / {len(wb_line_confs)} line confs")


            fix_passed, fix_sql_str = False, ""
            if not passed:
                #try one fixer pass
                _show(f"base ({args.base_model}) → {error_msg[:60]}", clean_sql)
                _log(f"step 4: fixing with {args.fixer_model}…")
                _log(f"mem before fix: {_gpu_mem()}")
                fix_sql_str, fix_passed, fix_tries, fix_err = fix_sql(
                    fixer_client, clean_sql, error_msg, ex, args.fixer_model,
                    max_tokens=args.max_tokens,
                    max_attempts=1,
                )
                if fix_sql_str:
                    tag = "FIXED" if fix_passed else "DISCARDED"
                    _show(f"fixer ({args.fixer_model}) → {tag}", fix_sql_str)
                _log(f"step 4 done: fix_passed={fix_passed}, tries={fix_tries}, last='{fix_err[:80]}'")
                _log(f"mem after fix: {_gpu_mem()}")

            #build inclusion flags
            _log("step 5: inclusion flags…")
            sql_lines  = clean_sql.splitlines() if clean_sql else []

            if passed or not fix_sql_str:
                tok_in_final  = [1] * len(sql_tokens)
                line_in_final = [1] * len(sql_lines)
            else:
                tok_in_final = token_inclusion(sql_tokens, fix_sql_str, tokenize_sql)

                _line_for_tok_step5 = []
                _cur = 0
                _ci  = 0
                for t in sql_tokens:
                    p = clean_sql.find(t, _ci)
                    if p >= 0:
                        _cur = clean_sql[:p].count("\n")
                        _ci  = p + len(t)
                    _line_for_tok_step5.append(_cur)

                line_in_final = [1] * len(sql_lines)
                for t_idx, ln in enumerate(_line_for_tok_step5):
                    if tok_in_final[t_idx] == 0 and ln < len(line_in_final):
                        line_in_final[ln] = 0

            _log(f"step 5 done: {len(sql_tokens)} tokens, {len(line_in_final)} lines")

            #build output arrays
            _log("step 6: building arrays")

            token_array = []
            cur_line = 0
            char_idx = 0
            line_for_tok = []
            for tok in sql_tokens:
                pos = clean_sql.find(tok, char_idx) if clean_sql else -1
                if pos >= 0:
                    cur_line = clean_sql[:pos].count("\n")
                    char_idx = pos + len(tok)
                line_for_tok.append(cur_line)

            for t_idx, tok in enumerate(sql_tokens):
                ln = line_for_tok[t_idx] if t_idx < len(line_for_tok) else 0
                v_conf = verb_line_confs.get(ln, SQL_BASE_RATE)
                t_conf = round(token_probs[t_idx], 4) if t_idx < len(token_probs) else 0.0
                c_conf = consist_line_confs.get(ln, 0.0)
                in_f = tok_in_final[t_idx] if t_idx < len(tok_in_final) else 1
                wb_conf = wb_token_confs.get(t_idx, 0.5)
                token_array.append([tok, round(v_conf, 4), t_conf, round(c_conf, 4), round(wb_conf, 4), in_f])

            line_array = []
            for ln_idx in range(len(sql_lines)):
                v_conf = verb_line_confs.get(ln_idx, SQL_BASE_RATE)
                t_conf = round(line_tok_confs[ln_idx], 4) if ln_idx < len(line_tok_confs) else 0.0
                c_conf = consist_line_confs.get(ln_idx, 0.0)
                in_f = line_in_final[ln_idx] if ln_idx < len(line_in_final) else 1
                wb_conf = wb_line_confs.get(ln_idx, 0.5)
                line_array.append([ln_idx, round(v_conf, 4), t_conf, round(c_conf, 4), round(wb_conf, 4), in_f])

            status = "PASS" if passed else ("FIXED" if fix_passed else "FAIL")
            elapsed = time.time() - t0
            print(f"  → {status}  ({elapsed:.1f}s)", flush=True)
            if passed:
                n_passed += 1
            elif fix_passed:
                n_fix_passed += 1

            record = {
                "problem_index": i - 1,
                "example_id": example_id,
                "db_id": db_id,
                "dataset": args.dataset,
                "passed": passed,
                "fix_passed": fix_passed,
                "token_array": token_array,
                "line_array": line_array,
            }
            all_records.append(record)

            jsonl_fh.write(json.dumps(record) + "\n")
            jsonl_fh.flush()

        except Exception as exc:
            #keep going after errors
            elapsed = time.time() - t0
            n_errors += 1
            print(f"\n{'='*60}", flush=True)
            print(f"ERROR on {db_id}/{example_id} after {elapsed:.1f}s: {exc}", flush=True)
            print(f"Memory: {_gpu_mem()}", flush=True)
            traceback.print_exc()
            print(f"{'='*60}\n", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue

    jsonl_fh.close()

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"SQL": all_records}, f, indent=2)

    total = len(examples)
    print(f"\nResults (JSONL) -> {out_path}", flush=True)
    print(f"Results (JSON)  -> {json_path}", flush=True)
    print(f"Pass rate (base): {n_passed}/{total}  ({100*n_passed/total:.1f}%)", flush=True)
    if n_fix_passed:
        print(f"Pass rate (fixed): {n_passed+n_fix_passed}/{total}  "
              f"({100*(n_passed+n_fix_passed)/total:.1f}%)", flush=True)
    if n_errors:
        print(f"Errors: {n_errors}/{total}", flush=True)


if __name__ == "__main__":
    main()
