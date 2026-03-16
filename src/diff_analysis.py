#done by Rei Shindo
#This file compares original vs final outputs and builds diff records

# Functions:
#_tokenize_code_model tokenizes code with the model tokenizer
#_tokenize_code_fallback tokenizes code with a regex fallback
#_tokenize_sql tokenizes sql text with a simple regex
#token_pairs marks original tokens as kept or removed
#_normalise_line normalizes spaces in one line
#line_pairs marks original lines as kept or removed
#build_diff_record builds one json-ready diff summary
#write_diff_record appends one diff record to a jsonl file

import difflib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.model_tokenizing import tokenize as model_tokenize

#tokenize code using model tokenizer if possible, fallback to regex otherwise
def _tokenize_code_model(code: str, tokenizer) -> List[str]:
    if not code:
        return []
    return model_tokenize(code, tokenizer, add_special_tokens=False).tokens

#fallback code tokenizer using regex
def _tokenize_code_fallback(code: str) -> List[str]:
    if not code:
        return []
    return re.findall(r'\w+|[^\s\w]', code)

#tokenize sql text with a simple regex
def _tokenize_sql(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

#mark original tokens as kept or removed
def token_pairs(original: str, final: str, lang: str = "code",
                tokenizer=None) -> List[Tuple[str, int]]:
    if lang == "code" and tokenizer is not None:
        tok_fn = lambda text: _tokenize_code_model(text, tokenizer)
    elif lang == "code":
        tok_fn = _tokenize_code_fallback
    else:
        tok_fn = _tokenize_sql
    orig_toks = tok_fn(original)
    final_toks = tok_fn(final)

    if not orig_toks:
        return []

    sm = difflib.SequenceMatcher(None, orig_toks, final_toks, autojunk=False)
    pairs: List[Tuple[str, int]] = []

    previous_tag = None

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for idx, tok in enumerate(orig_toks[i1:i2]):
                if idx == 0 and previous_tag == "insert":
                    pairs.append((tok, 0))
                else:
                    pairs.append((tok, 1))
        elif tag == "replace":
            for tok in orig_toks[i1:i2]:
                pairs.append((tok, 0))
        elif tag == "delete":
            for tok in orig_toks[i1:i2]:
                pairs.append((tok, 0))
        previous_tag = tag

    return pairs

#normalize spaces in one line
def _normalise_line(line: str) -> str:
    return " ".join(line.split())

#mark original lines as kept or removed
def line_pairs(original: str, final: str) -> List[Tuple[str, int]]:
    def split_lines(text: str) -> List[str]:
        return [l for l in text.splitlines() if l.strip()]

    orig_lines = split_lines(original)
    final_lines = split_lines(final)

    if not orig_lines:
        return []

    orig_norm = [_normalise_line(l) for l in orig_lines]
    final_norm = [_normalise_line(l) for l in final_lines]

    sm = difflib.SequenceMatcher(None, orig_norm, final_norm, autojunk=False)
    pairs: List[Tuple[str, int]] = []

    previous_tag = None

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for idx, line in enumerate(orig_lines[i1:i2]):
                if idx == 0 and previous_tag == "insert":
                    pairs.append((line, 0))
                else:
                    pairs.append((line, 1))
        elif tag in ("replace", "delete"):
            for line in orig_lines[i1:i2]:
                pairs.append((line, 0))
        previous_tag = tag

    return pairs

#compute diff record for one original vs final pair, with token and line level diffs and summary stats
def build_diff_record(
    task_id: str,
    original: str,
    final: str,
    method: str,
    lang: str = "code",
    extra: Optional[Dict] = None,
    tokenizer=None,
) -> dict:
    tpairs = token_pairs(original, final, lang=lang, tokenizer=tokenizer)
    lpairs = line_pairs(original, final)

    tokens_kept    = sum(1 for _, v in tpairs if v == 1)
    tokens_removed = sum(1 for _, v in tpairs if v == 0)
    lines_kept     = sum(1 for _, v in lpairs if v == 1)
    lines_removed  = sum(1 for _, v in lpairs if v == 0)

    if lang == "code" and tokenizer is not None:
        n_tokens_final = len(_tokenize_code_model(final, tokenizer))
    elif lang == "code":
        n_tokens_final = len(_tokenize_code_fallback(final))
    else:
        n_tokens_final = len(_tokenize_sql(final))

    record: dict = {
        "task_id":        task_id,
        "method":         method,
        "lang":           lang,
        "token_pairs":    [[t, v] for t, v in tpairs],
        "line_pairs":     [[l, v] for l, v in lpairs],
        "n_tokens_orig":  len(tpairs),
        "n_tokens_final": n_tokens_final,
        "n_lines_orig":   len([l for l in original.splitlines() if l.strip()]),
        "n_lines_final":  len([l for l in final.splitlines() if l.strip()]),
        "tokens_kept":    tokens_kept,
        "tokens_removed": tokens_removed,
        "lines_kept":     lines_kept,
        "lines_removed":  lines_removed,
    }
    if extra:
        record.update(extra)
    return record

#append one diff record to a jsonl file
def write_diff_record(path: str | Path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
