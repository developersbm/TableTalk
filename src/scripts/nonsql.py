#done by Sebastian Bastida Marin
# This script runs the non sql confidence pipeline and saves outputs

# Functions:
#_mbpp_funcname gets the function name from mbpp asserts
#_gpu_mem shows quick gpu and ram usage text (impoting up for setting up HPC3)
#_log prints timestamped logs and flushes streams
#_setup_signal_handlers exits on job signals
#_show trims preview text for debug prints
#parse_args reads cli options for dataset and models
#main loads data models and optional probe then starts the run
#_run_combined does generation scoring fixing and output writing
import argparse, json, os, sys, math, signal, traceback, time, resource
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.consistency_scoring import (
    generate_reference, generate_samples, compute_consistency,
    code_prompt, mbpp_code_prompt, postprocess_code,
)
from src.llm_client import LLMClient, GemmaClient
from src.verbalized_confidence import (
    score_example as verbalized_score_example, run_code_tests,
)
from src.model_tokenizing import load_tokenizer, tokenize, decode
from src.tokenize_confidence import aggregate_line_min
from src.fixer import fix_code
from src.loader import load_humaneval, load_mbpp
from src.inclusion import token_inclusion, line_inclusion
from src.whitebox_probing import WhiteBoxProbe, extract_span_hidden_states



#get mbpp entry point from test asserts
def _mbpp_funcname(test_list):
    import re
    for t in (test_list or []):
        m = re.search(r'assert\s+(?:set\()?([a-zA-Z_]\w*)\s*\(', t)
        if m:
            return m.group(1)
    return None



#format memory stats for logs
def _gpu_mem() -> str:
    parts = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_mem / 1024**3
                parts.append(f"GPU{i}: {alloc:.1f}/{total:.0f}GB (res {reserved:.1f}GB)")
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    alloc = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    props = torch.cuda.get_device_properties(i)
                    total = props.total_memory / 1024**3
                    parts.append(f"GPU{i}: {alloc:.1f}/{total:.0f}GB (res {reserved:.1f}GB)")
        except Exception as e:
            parts.append(f"GPU stats error: {e}")
    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_gb = rss_kb / (1024**2 if sys.platform == "darwin" else 1024**2)
        parts.append(f"RAM_RSS: {rss_gb:.1f}GB")
    except Exception:
        pass
    return "  |  ".join(parts) if parts else "stats unavailable"



#print log line with timestamp
def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()



#register signal handlers for slurm exits
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



#clip debug text preview
def _show(label: str, text: str, max_lines: int = 12, max_cols: int = 220) -> None:
    lines = (text or "").strip().splitlines()
    truncated = len(lines) > max_lines
    for ln in lines[:max_lines]:
        if len(ln) > max_cols:
            ln = ln[:max_cols] + " …"



#parse command line args
def parse_args():
    p = argparse.ArgumentParser(
        description="Combined confidence scoring (HumanEval+ and  MBPP+)"
    )
    p.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    p.add_argument("--limit", type=int, default=5,
                   help="Examples to process (default: 5, use 0 for all)")
    p.add_argument("--k", type=int, default=5,
                   help="Stochastic samples for consistency scoring")
    p.add_argument("--base_model", default="gemma",
                   help="Base generator: 'gemma' or 'qwen' (default: gemma)")
    p.add_argument("--fixer_model", default="qwen",
                   help="Fixer model: 'qwen' or 'gemma' (default: qwen)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--output", default=None,
                   help="Output JSONL path (default: outputs/combined_<dataset>_<ts>.jsonl)")
    p.add_argument("--probe_path", default=None,
                   help="Path to trained WhiteBoxProbe (optional).")
    return p.parse_args()



#load data and clients then run pipeline
def main():
    load_dotenv()
    _setup_signal_handlers()
    args = parse_args()

    if args.dataset == "humaneval":
        examples = load_humaneval(limit=args.limit or None)
    else:
        examples = load_mbpp(limit=args.limit or None)
    print(f"Loaded {len(examples)} examples from {args.dataset}", flush=True)

    print(f"\nLoading models", flush=True)
    print(f"Memory before model loading: {_gpu_mem()}", flush=True)

    if args.base_model == "gemma":
        fixer_client = LLMClient()
        print(f"Memory after: {_gpu_mem()}", flush=True)
        base_client = GemmaClient()
        print(f"Memory after: {_gpu_mem()}", flush=True)
    else:
        fixer_client = GemmaClient()
        print(f"Memory after: {_gpu_mem()}", flush=True)
        base_client = LLMClient()
        print(f"Memory after: {_gpu_mem()}", flush=True)

    tokenizer = load_tokenizer()
    print(f"Models loaded. Memory: {_gpu_mem()}", flush=True)

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent.parent / "outputs"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"combined_{args.dataset}_{ts}.jsonl"

    probe = None
    if args.probe_path and os.path.exists(args.probe_path):
        probe = WhiteBoxProbe.load(args.probe_path, device='cpu')
        probe.eval()
        print(f"Loaded probe from {args.probe_path}", flush=True)

    _run_combined(base_client, fixer_client, examples, args, out_path, tokenizer, probe)



#run full scoring loop and save outputs
def _run_combined(base_client, fixer_client, examples, args, out_path, tokenizer, probe=None):
    n_passed = n_fix_passed = n_errors = 0
    all_records = []

    jsonl_fh = open(out_path, "w", encoding="utf-8")

    for i, ex in enumerate(examples, 1):
        #process one example
        task_id = ex["task_id"]
        entry_point = ex.get("entry_point", "")
        t0 = time.time()
        print(f"\n[{i}/{len(examples)}] {task_id}", flush=True)
        _log(f"START  mem: {_gpu_mem()}")

        try:
            #build prompt per dataset
            if args.dataset == "mbpp":
                func_name = _mbpp_funcname(ex.get("test_list") or [])
                if not func_name:
                    func_name = entry_point or "solution"
                entry_point = func_name
                prompt = mbpp_code_prompt(
                    ex["prompt"],
                    func_name,
                    test_list=ex.get("test_list") or [],
                    test_imports=ex.get("test_imports") or [],
                )
            else:
                prompt = code_prompt(ex["prompt"])

            #generate code and token logprobs
            _log("step 1: generate_with_full_logprobs (base)…")
            raw_text, ref_lp, full_logprobs, hidden_states = base_client.generate_with_full_logprobs(
                prompt, max_tokens=args.max_tokens, return_hidden_states=True
            )
            clean_code = postprocess_code(raw_text)
            _tok = lambda text: tokenize(text, tokenizer, add_special_tokens=False).tokens
            code_tokens = _tok(clean_code)
            _log(f"step 1 done: {len(full_logprobs)} tokens, {len(clean_code)} chars")
            _log(f"  mem: {_gpu_mem()}")

            #run tests on generated code
            _log("step 2: run_code_tests…")
            passed, error_msg = run_code_tests(
                func_code=clean_code,
                entry_point=entry_point,
                test_body=ex.get("test_body", ""),
                test_list=ex.get("test_list") or [],
                test_imports=ex.get("test_imports") or [],
            )
            if passed:
                _log(f"step 2 done: passed=True")
            else:
                _log(f"step 2 done: passed=False err={error_msg[:120]}")

            #token confidence from logprobs
            _log("step 3a: tokenized confidence…")
            token_probs = [math.exp(lp) for _, lp in full_logprobs]
            token_strings = [tok for tok, _ in full_logprobs]
            line_tok_map = aggregate_line_min(clean_code, full_logprobs)
            line_tok_confs = list(line_tok_map.values())
            _log(f"step 3a done")

            #line confidence from verbalized self report
            _log("step 3b: verbalized confidence…")
            _log(f"mem before verbalized: {_gpu_mem()}")
            verb_result = verbalized_score_example(
                client=base_client, example=ex,
                max_tokens=args.max_tokens,
                generated_code=clean_code,
                test_result=(passed, error_msg),
            )
            verb_line_confs = {}
            try:
                verb_line_confs = json.loads(verb_result.get("line_confs_json", "{}"))
                verb_line_confs = {int(k): float(v) for k, v in verb_line_confs.items()}
            except (json.JSONDecodeError, ValueError):
                pass
            _log(f"step 3b done: {len(verb_line_confs)} lines scored")
            _log(f"mem after verbalized: {_gpu_mem()}")

            #line confidence from sample agreement
            _log(f"step 3c: consistency confidence (k={args.k})…")
            _log(f"mem before consistency: {_gpu_mem()}")
            raw_samples = generate_samples(
                base_client, prompt, k=args.k,
                temperature=args.temperature, top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            samples = [postprocess_code(t) for t, _ in raw_samples]
            consist_m = compute_consistency(clean_code, samples, mode="code", tokenizer=tokenizer)
            consist_line_confs = {}
            try:
                consist_line_confs = json.loads(consist_m.get("line_scores_json", "{}"))
                consist_line_confs = {int(k): float(v) for k, v in consist_line_confs.items()}
            except (json.JSONDecodeError, ValueError):
                pass
            _log(f"step 3c done: {len(consist_line_confs)} lines scored")
            _log(f"mem after consistency: {_gpu_mem()}")

            #token and line confidence from hidden states
            _log("step 3d: white-box probing confidence…")
            wb_token_confs = {}
            wb_line_confs = {}
            
            if probe is not None and hidden_states is not None and hidden_states.size(0) > 0:
                with torch.no_grad():
                    gen_tokens = [t_str for t_str, _ in full_logprobs]
                    for t_idx, tok in enumerate(code_tokens):
                        span_hs = extract_span_hidden_states(raw_text, gen_tokens, hidden_states, tok)
                        conf = probe(span_hs, apply_platt=True).item()
                        wb_token_confs[t_idx] = conf
                        
                    for ln_idx, ln_text in enumerate(clean_code.splitlines() if clean_code else []):
                        span_hs = extract_span_hidden_states(raw_text, gen_tokens, hidden_states, ln_text)
                        conf = probe(span_hs, apply_platt=True).item()
                        wb_line_confs[ln_idx] = conf
            _log(f"step 3d done")

            fix_passed, fix_code_str = False, ""
            if not passed:
                #try one fixer pass if base fails
                _log(f"error: {error_msg}")
                _show(f"base ({args.base_model}), it fails", clean_code)
                _log(f"step 4: fixing with {args.fixer_model}…")
                _log(f"mem before fix: {_gpu_mem()}")
                fix_code_str, fix_passed, fix_tries, fix_err = fix_code(
                    fixer_client, clean_code, error_msg, ex, args.fixer_model,
                    max_tokens=args.max_tokens,
                    max_attempts=1,
                )
                if fix_code_str:
                    tag = "FIXED" if fix_passed else "DISCARDED"
                    _show(f"fixer ({args.fixer_model}) → {tag}", fix_code_str)
                if not fix_passed and fix_err:
                    _log(f"fixer error: {fix_err}")
                _log(f"step 4 done: fix_passed={fix_passed}, tries={fix_tries}")
                _log(f"mem after fix: {_gpu_mem()}")

            #mark which tokens and lines survive in final code
            _log("step 5: inclusion flags…")
            code_lines  = clean_code.splitlines() if clean_code else []

            if passed or not fix_code_str:
                tok_in_final  = [1] * len(code_tokens)
                line_in_final = [1] * len(code_lines)
            else:
                tok_in_final = token_inclusion(code_tokens, fix_code_str, _tok)

                _line_for_tok_step5 = []
                _cur = 0
                _ci  = 0
                for t in code_tokens:
                    p = clean_code.find(t, _ci)
                    if p >= 0:
                        _cur = clean_code[:p].count("\n")
                        _ci  = p + len(t)
                    _line_for_tok_step5.append(_cur)

                line_in_final = [1] * len(code_lines)
                for t_idx, ln in enumerate(_line_for_tok_step5):
                    if tok_in_final[t_idx] == 0 and ln < len(line_in_final):
                        line_in_final[ln] = 0

            _log(f"step 5 done: {len(code_tokens)} tokens, {len(line_in_final)} lines")

            #build output arrays used by eval scripts
            _log("step 6: building arrays…")

            PY_BASE_RATE = 0.776 if args.dataset == "humaneval" else 0.605
            token_array = []
            cur_line = 0
            char_idx = 0
            line_for_tok = []
            for tok in code_tokens:
                pos = clean_code.find(tok, char_idx)
                if pos >= 0:
                    cur_line = clean_code[:pos].count("\n")
                    char_idx = pos + len(tok)
                line_for_tok.append(cur_line)

            for t_idx, tok in enumerate(code_tokens):
                ln = line_for_tok[t_idx] if t_idx < len(line_for_tok) else 0
                v_conf = verb_line_confs.get(ln, PY_BASE_RATE)
                t_conf = round(token_probs[t_idx], 4) if t_idx < len(token_probs) else 0.0
                c_conf = consist_line_confs.get(ln, 0.0)
                in_f = tok_in_final[t_idx] if t_idx < len(tok_in_final) else 1
                wb_conf = wb_token_confs.get(t_idx, 0.5)
                token_array.append([tok, round(v_conf, 4), t_conf, round(c_conf, 4), round(wb_conf, 4), in_f])

            line_array = []
            for ln_idx in range(len(code_lines)):
                v_conf = verb_line_confs.get(ln_idx, PY_BASE_RATE)
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
                "task_id": task_id,
                "entry_point": entry_point,
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
            #keep going even if one item crashes
            elapsed = time.time() - t0
            n_errors += 1
            print(f"ERROR on {task_id} after {elapsed:.1f}s: {exc}", flush=True)
            print(f"Memory: {_gpu_mem()}", flush=True)
            traceback.print_exc()
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
        json.dump({"HumanEval": all_records}, f, indent=2)

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
