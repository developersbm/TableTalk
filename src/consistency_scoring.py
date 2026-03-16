#done by Sebastian Bastida Marin
#This file handles prompts cleanup tokenizing and consistency scores

# Functions:
#generate_reference gets one model output
#generate_samples gets k sampled outputs
#postprocess_sql cleans sql model output
#quote_unquoted_identifiers adds quotes for special schema names
#postprocess_code cleans code model output
#sql_prompt builds the sql generation prompt
#code_prompt builds the humaneval style code prompt
#mbpp_code_prompt builds the mbpp code prompt
#tokenize_sql splits sql into stable tokens
#_token_scores compares ref tokens vs samples
#_line_scores turns token scores into line scores
#compute_consistency returns summary consistency metrics

import json
import re
from src.model_tokenizing import tokenize as model_tokenize
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

import numpy as np


#get one reference generation
def generate_reference(client, prompt: str, model: str = "", max_tokens: int = 512) -> tuple[str, float]:
    return client.generate_reference(prompt, max_tokens=max_tokens)


def generate_samples(client, prompt: str, k: int, model: str = "",
                     temperature: float = 0.8, top_p: float = 0.95,
                     max_tokens: int = 512) -> list[tuple[str, float]]:
    return client.generate_samples(prompt, k=k, temperature=temperature,
                                   top_p=top_p, max_tokens=max_tokens)


#get k sampled generations
_DDL_PREFIXES = ('CREATE', 'DROP', 'ALTER', 'TRUNCATE')

#clean sql output text
def postprocess_sql(raw: str) -> str:
    if not raw:
        return ""
    text = raw.strip()
    m = re.search(r'^```(?:sql)?\s*\n(.*?)\n```$', text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    text = text.replace('`', '')
    _DML_KW = ('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')
    _STOP  = ('this query', 'explanation', 'here is', 'note that')
    lines, out, in_sql = text.split('\n'), [], False
    for line in lines:
        if not in_sql and any(line.strip().upper().startswith(k) for k in _DML_KW):
            in_sql = True
        if in_sql:
            lo = line.lower()
            if line.strip() and any(p in lo for p in _STOP):
                break
            out.append(line)
    text = '\n'.join(out).strip() if out else text
    if ';' in text:
        for part in text.split(';'):
            stripped = part.strip()
            if not stripped:
                continue
            fw = stripped.split()[0].upper() if stripped.split() else ""
            if fw not in _DDL_PREFIXES:
                text = stripped
                break
    text = text.strip()
    first_word = text.split()[0].upper() if text.split() else ""
    if first_word in _DDL_PREFIXES:
        return ""
    return text

#quote schema names with special chars
def quote_unquoted_identifiers(sql: str, schema: str) -> str:
    if not sql or not schema:
        return sql
    needs_quoting: set = set()
    for m in re.finditer(r'"([^"]+)"', schema):
        name = m.group(1)
        if re.search(r'[\s()\-]', name):
            needs_quoting.add(name)
    if not needs_quoting:
        return sql
    result = sql
    for col in sorted(needs_quoting, key=len, reverse=True):
        pattern = r'(?<!["\w])' + re.escape(col) + r'(?!["\w])'
        result = re.sub(pattern, f'"{col}"', result, flags=re.IGNORECASE)
    return result


#clean code output text
def postprocess_code(raw: str) -> str:
    if not raw:
        return ""
    text = raw.strip()
    m = re.search(r'^```(?:python)?\s*\n(.*?)\n```$', text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith(('def ', 'class ', 'import ', 'from ', 'return ')):
            text = '\n'.join(lines[i:])
            break
    return text.strip()




#build sql prompt text
def sql_prompt(question: str, db_id: str, schema: Optional[str], evidence: Optional[str]) -> str:
    parts = [
        "You are an expert SQLite programmer.",
        f"\nDatabase: {db_id}",
    ]
    if schema:
        parts.append(
            f"\nSchema (use ONLY these tables and columns — do not invent names):\n{schema}"
        )
    else:
        parts.append(
            "\n(No schema available — infer table/column names from the question carefully.)"
        )
    parts.append(f"\nQuestion: {question}")
    if evidence:
        parts.append(f"Hint: {evidence}")
    parts.append(
        "\nWrite a single SQLite SQL query that answers the question."
        "\nRules:"
        "\n- Output ONLY the SQL query. No markdown, no explanation, no commentary."
        "\n- Use only the exact table and column names shown in the schema above."
        "\n- Do not create new column names or alias them unless needed for the SELECT list."
        "\n- When joining tables, always qualify column references with the table name or alias"
        " (e.g. T1.CDSCode) to avoid ambiguous column errors."
        "\n- Do NOT wrap a dotted `table.column` reference in double quotes — only quote"
        " identifiers that contain spaces or special characters."
    )
    return "\n".join(parts)

#build generic code prompt
def code_prompt(problem: str) -> str:
    return (
        "You are an expert Python programmer. Complete the following function.\n"
        "Copy the EXACT function signature from the problem, then write the "
        "complete implementation below it.\n"
        "Output ONLY the complete function (signature + body) - no markdown, "
        "no explanation, no extra text.\n\n"
        f"{problem}"
    )

#build mbpp specific code prompt
def mbpp_code_prompt(
    description: str,
    func_name: str,
    test_list: Optional[List[str]] = None,
    test_imports: Optional[List[str]] = None,
) -> str:
    tests = [t.strip() for t in (test_list or []) if str(t).strip()]
    imports = [t.strip() for t in (test_imports or []) if str(t).strip()]

    test_preview = "\n".join(tests[:8]) if tests else ""
    import_preview = "\n".join(imports[:6]) if imports else ""
    import_block = f"\nRequired imports used by tests:\n{import_preview}\n" if import_preview else ""
    test_block = f"\nUnit tests your function must satisfy:\n{test_preview}\n" if test_preview else ""

    return (
        "You are an expert Python programmer.\n"
        f"Write exactly one function named `{func_name}` that satisfies the task and tests.\n\n"
        f"Task:\n{description}\n"
        f"{import_block}"
        f"{test_block}"
        "Rules:\n"
        f"- Function name MUST be exactly `{func_name}`.\n"
        "- Output ONLY Python code (no markdown, no prose).\n"
        "- Return exactly one top-level function definition (def + body).\n"
        "- Do not print anything.\n"
        "- Do not include comments or docstrings unless required for correctness."
    )

_SQL_KW = {
    'SELECT','FROM','WHERE','JOIN','LEFT','RIGHT','INNER','OUTER','CROSS','ON','AND','OR','NOT',
    'IN','EXISTS','BETWEEN','LIKE','IS','NULL','GROUP','BY','HAVING','ORDER','ASC','DESC',
    'LIMIT','OFFSET','INSERT','INTO','VALUES','UPDATE','SET','DELETE','CREATE','DROP','ALTER',
    'TABLE','AS','DISTINCT','ALL','UNION','INTERSECT','EXCEPT','CASE','WHEN','THEN','ELSE','END',
    'COUNT','SUM','AVG','MIN','MAX','CAST','WITH','OVER','PARTITION','IIF','COALESCE',
}

#tokenize sql into stable pieces
def tokenize_sql(sql: str) -> List[str]:
    if not sql:
        return []
    sql = re.sub(r'\s+', ' ', sql.strip())
    tokens, i = [], 0
    while i < len(sql):
        c = sql[i]
        if c.isspace():
            i += 1; continue
        if c in ('"', "'"):
            j = i + 1
            while j < len(sql) and sql[j] != c:
                j += (2 if sql[j] == '\\' and j + 1 < len(sql) else 1)
            j = min(j + 1, len(sql))
            tokens.append(sql[i:j]); i = j; continue
        if c.isdigit() or (c == '.' and i + 1 < len(sql) and sql[i+1].isdigit()):
            j, dot = i, False
            while j < len(sql) and (sql[j].isdigit() or (sql[j] == '.' and not dot)):
                if sql[j] == '.': dot = True
                j += 1
            tokens.append(sql[i:j]); i = j; continue
        if i + 1 < len(sql) and sql[i:i+2] in ('!=', '<>', '<=', '>=', '||', '::'):
            tokens.append(sql[i:i+2]); i += 2; continue
        if c in '()[]{},.;=<>+-*/%!':
            tokens.append(c); i += 1; continue
        if c == '`':
            j = i + 1
            while j < len(sql) and sql[j] != '`': j += 1
            j = min(j + 1, len(sql))
            tokens.append(sql[i:j]); i = j; continue
        if c.isalpha() or c == '_':
            j = i
            while j < len(sql) and (sql[j].isalnum() or sql[j] == '_'): j += 1
            tok = sql[i:j]
            tokens.append(tok.lower() if tok.upper() in _SQL_KW else tok)
            i = j; continue
        tokens.append(c); i += 1
    return tokens

#tokenize code using model tokenizer if possible, fallback to regex otherwise
def _token_scores(ref_tokens: List[str], samples: List[str], tokenizer) -> List[float]:
    if not ref_tokens or not samples:
        return [0.0] * len(ref_tokens)
    acc = [0.0] * len(ref_tokens)
    for s in samples:
        if not s:
            continue
        stok = tokenizer(s)
        matcher = SequenceMatcher(None, ref_tokens, stok)
        for tag, i1, i2, *_ in matcher.get_opcodes():
            if tag == 'equal':
                for i in range(i1, i2):
                    acc[i] += 1.0
    return [a / len(samples) for a in acc]

#turn token scores into line scores
def _line_scores(ref_text: str, ref_tokens: List[str],
                 tok_scores: List[float], tokenizer) -> Dict[int, float]:
    lines = ref_text.split('\n')
    out = {}
    idx = 0
    for ln, line in enumerate(lines):
        toks = tokenizer(line)
        if not toks:
            out[ln] = 1.0
        else:
            end = idx + len(toks)
            slice_ = tok_scores[idx:end]
            out[ln] = float(np.mean(slice_)) if slice_ else 1.0
            idx = end
        if idx >= len(tok_scores):
            break
    return out

#compute consistency metrics for one reference vs multiple samples
def compute_consistency(reference: str, samples: List[str], mode: str = "sql",
                        tokenizer=None) -> Dict[str, Any]:
    if mode == "sql":
        tok_fn = tokenize_sql
    else:
        if tokenizer is None:
            raise ValueError("compute_consistency(mode='code') requires a tokenizer argument")
        tok_fn = lambda text: model_tokenize(text, tokenizer, add_special_tokens=False).tokens
    empty = dict(token_mean=0.0, token_min=0.0, token_p10=0.0,
                 line_mean=0.0, line_min=0.0, n_ref_tokens=0,
                 n_ref_lines=0, line_scores_json='{}')

    if not reference:
        return empty

    ref_tokens = tok_fn(reference)
    if not ref_tokens:
        return empty

    tok_scores = _token_scores(ref_tokens, samples, tok_fn)
    token_mean = float(np.mean(tok_scores))
    token_min  = float(np.min(tok_scores))
    token_p10  = float(np.percentile(tok_scores, 10))

    line_sc = _line_scores(reference, ref_tokens, tok_scores, tok_fn)
    vals = list(line_sc.values())
    line_mean = float(np.mean(vals)) if vals else 0.0
    line_min  = float(np.min(vals))  if vals else 0.0

    return dict(
        token_mean=token_mean,
        token_min=token_min,
        token_p10=token_p10,
        line_mean=line_mean,
        line_min=line_min,
        n_ref_tokens=len(ref_tokens),
        n_ref_lines=len(reference.split('\n')),
        line_scores_json=json.dumps({str(k): round(v, 3) for k, v in line_sc.items()}),
    )
