#done by Logan Mifflin
#This file calculates model confidence scores using token probabilities
#it uses log probabilities to find the weakest links in generated text

# Functions:
#generate_with_full_logprobs: gets text and token scores from the model
#aggregate_line_min: finds the lowest token probability for each line
#build_line_array: packages line scores into a numpy array for analysis
#score_sql_example: evaluates a sql example and computes its confidence metrics

import json
import math
import numpy as np
from typing import Dict, List, Tuple, Any

#gets text and token scores from the model
def generate_with_full_logprobs(
    client,
    prompt: str,
    model: str = "",
    max_tokens: int = 512,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    return client.generate_with_full_logprobs(prompt, max_tokens=max_tokens)

#finds the lowest token probability for each line
def aggregate_line_min(text: str, token_logprobs: List[Tuple[str, float]]) -> Dict[int, float]:
    if not token_logprobs or not text:
        return {i: 1.0 for i in range(len(text.split("\n")))}

    current_line = 0
    line_probs: Dict[int, List[float]] = {}

    #map tokens to their corresponding lines
    for tok, lp in token_logprobs:
        prob = math.exp(lp)
        newlines = tok.count("\n")
        for spanned in range(current_line, current_line + newlines + 1):
            line_probs.setdefault(spanned, []).append(prob)
        current_line += newlines

    n_lines = len(text.split("\n"))
    result: Dict[int, float] = {}
    for ln in range(n_lines):
        probs = line_probs.get(ln)
        result[ln] = float(np.min(probs)) if probs else 1.0
    return result

#packages line scores into a numpy array for analysis
def build_line_array(line_confs: Dict[int, float], passed: bool) -> np.ndarray:
    outcome = 1.0 if passed else 0.0
    rows = [[float(idx), conf, outcome] for idx, conf in sorted(line_confs.items())]
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, 3), dtype=np.float32)

#evaluates a sql example and computes its confidence metrics
def score_sql_example(
    client,
    example: dict,
    model: str = "",
    max_tokens: int = 512,
) -> dict:
    from src.consistency_scoring import sql_prompt, postprocess_sql, tokenize_sql

    prompt = sql_prompt(
        example["question"],
        example.get("db_id", ""),
        example.get("schema"),
        example.get("evidence"),
    )
    
    #generate the sql string and raw scores
    raw_text, mean_lp, token_logprobs = client.generate_with_full_logprobs(
        prompt, max_tokens=max_tokens
    )
    clean_sql = postprocess_sql(raw_text)

    #check if generated sql matches the right answer
    gold = (example.get("gold_sql") or "").strip()
    if not clean_sql:
        passed, error_msg = False, "empty SQL generated"
    elif gold:
        passed = tokenize_sql(clean_sql.lower()) == tokenize_sql(gold.lower())
        error_msg = "" if passed else "SQL does not match gold"
    else:
        passed, error_msg = False, "no gold_sql available"

    #calculate token and line level metric stats
    line_scores_map = aggregate_line_min(raw_text, token_logprobs)
    token_probs = [math.exp(lp) for _, lp in token_logprobs]
    line_vals = list(line_scores_map.values())
    line_array = build_line_array(line_scores_map, passed)

    return {
        "example_id":   example.get("example_id", ""),
        "db_id":        example.get("db_id", ""),
        "difficulty":   example.get("difficulty") or "",
        "question":     example.get("question", ""),
        "gold_sql":     gold,
        "passed":       passed,
        "error":        error_msg,
        "reference":    clean_sql,
        "ref_logprob":  round(mean_lp, 6) if not math.isnan(mean_lp) else "",
        "token_mean":   round(float(np.mean(token_probs)), 4) if token_probs else 0.0,
        "token_min":    round(float(np.min(token_probs)), 4) if token_probs else 0.0,
        "token_p10":    round(float(np.percentile(token_probs, 10)), 4) if token_probs else 0.0,
        "line_mean":    round(float(np.mean(line_vals)), 4) if line_vals else 0.0,
        "line_min":     round(float(np.min(line_vals)), 4) if line_vals else 0.0,
        "n_ref_tokens": len(token_probs),
        "n_ref_lines":  len(line_vals),
        "line_scores":  json.dumps({str(k): round(v, 4) for k, v in line_scores_map.items()}),
        "token_scores": json.dumps([round(p, 4) for p in token_probs]),
        "line_array":   line_array,
    }
