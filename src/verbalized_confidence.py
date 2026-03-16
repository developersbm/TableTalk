#done by Rei Shindo

#This file evaluates model confidence by having the model explicitly state its confidence
#it creates prompts asking for estimations and parses the generated model answers for sql and python

#reflective_prompt: asks the model to rate the odds of existing python code being correct
#reflective_sql_prompt: asks the model to rate the odds of existing sql code being correct
#verbalized_prompt: asks for new python code with confidence ratings written in inline comments
#annotate_code_prompt: asks the model to add inline confidence comments to an existing python function
#parse_reflective_response: extracts the list of floats representing line probabilities from the text
#parse_verbalized_response: splits newly generated python code from its inline confidence ratings
#run_code_tests: executes generated python code locally to see if it actually works
#build_line_array: packages line level confidence scores into a structured array
#compute_verbalized_metrics: calculates average and worst case confidence across all rated lines
#score_example: gets confidence and testing python code for a single example
#verbalized_sql_prompt: asks for new sql code with confidence ratings as inline comments
#annotate_sql_prompt: asks the model to add inline confidence comments to an existing sql query
#parse_verbalized_sql_response: splits newly generated sql code from its inline confidence ratings
#_sql_matches_gold: compares generated sql to the known right answer using token matching
#score_sql_example: gets confidence and testing sql code for a single example

import json
import re
import subprocess
import sys
import textwrap
from typing import Dict, List, Optional, Tuple

import numpy as np

_CODE_EXEC_TIMEOUT = 10

from src.consistency_scoring import generate_reference, postprocess_sql, tokenize_sql

PY_FALLBACK_BASE_RATE  = 0.776
SQL_FALLBACK_BASE_RATE = 0.850


#asks the model to rate the odds of existing python code being correct
def reflective_prompt(code: str, lines: List[str]) -> str:
    n = len(lines)
    lines_repr = "[\n" + ",\n".join(f'  {repr(l)}' for l in lines) + "\n]"
    return textwrap.dedent(f"""\
        We are attempting to estimate calibrated probabilities that lines of code
        are correct or if they will need to be edited.

        Consider the following code:
        ```python
        {code}
        ```

        We are attempting to estimate the probability that each line of code is correct.
        Please provide your estimate as a list [float] where each element is between
        0 and 1 representing the probability that the line will be correct and
        unedited. One or two digits of precision is fine.
        This is the individual line probabilities so the sum is not expected to be 1.

        These are the line splitting we are using:
        {lines_repr}

        Create a calibrated estimate of the probability that each line is correct.
        You can consider any potential issues with the code, but then place your
        final answer in a markdown code block with only a list [float] of length {n}.
        Do not end early, and do not stop until you list all {n} probabilities
        corresponding to the given splits.
    """)


#asks the model to rate the odds of existing sql code being correct
def reflective_sql_prompt(
    sql: str,
    lines: List[str],
    question: str = "",
    schema: str = "",
    evidence: str = "",
) -> str:
    n = len(lines)
    lines_repr = "[\n" + ",\n".join(f'  {repr(l)}' for l in lines) + "\n]"

    context_parts = []
    if question:
        context_parts.append(f"Question: {question}")
    if schema:
        context_parts.append(f"Schema:\n{schema}")
    if evidence:
        context_parts.append(f"Hint: {evidence}")
    context_block = ("\n".join(context_parts) + "\n\n") if context_parts else ""

    return textwrap.dedent(f"""\
        We are attempting to estimate calibrated probabilities that lines of SQL
        are correct or if they will need to be edited.

        {context_block}Consider the following SQL query:
        ```sql
        {sql}
        ```

        Please provide your estimate as a list [float] where each element is between
        0 and 1 representing the probability that the line will be correct and
        unedited. One or two digits of precision is fine.
        This is the individual line probabilities so the sum is not expected to be 1.

        These are the line splitting we are using:
        {lines_repr}

        Place your final answer in a markdown code block with only a list [float]
        of length {n}. Do not end early.
    """)


#asks for new python code with confidence ratings written in inline comments
def verbalized_prompt(problem: str) -> str:
    instruction = textwrap.dedent("""\
        You are an expert Python programmer.

        Complete the function below. Output the COMPLETE function (including the def signature).
        No imports, no markdown, no explanation.

        IMPORTANT: every non-blank line of code must end with an inline comment in EXACTLY this format:
        where 0.XX is your confidence for that line (0.00 = very uncertain, 1.00 = fully certain).

        Example of correctly formatted output:
            def add(a, b):
                return a + b

        Do NOT skip the annotation on any non-blank line.
        Do NOT add any other comments or explanation.

        Problem:
    """)
    return instruction + problem


#asks the model to add inline confidence comments to an existing python function
def annotate_code_prompt(code: str) -> str:
    instruction = textwrap.dedent("""\
        You are an expert Python programmer.

        The following Python function has already been written.
        Annotate every non-blank line with an inline comment in EXACTLY this format:
        where 0.XX is your confidence for that line (0.00 = very uncertain, 1.00 = fully certain).

        Return the complete annotated function. No imports, no markdown, no explanation.

        Example of correctly annotated output:
            def add(a, b):
                return a + b

        Do NOT skip the annotation on any non-blank line.
        Do NOT add any other comments or explanation.

        Code to annotate:
    """)
    return instruction + "\n" + code



_CONF_RE     = re.compile(r"#\s*conf:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)
_LIST_FLOAT  = re.compile(r"\[([^\[\]]+)\]", re.DOTALL)


#extracts the list of floats representing line probabilities from the text
def parse_reflective_response(
    raw: str,
    line_indices: List[int],
    fallback: float = PY_FALLBACK_BASE_RATE,
) -> Tuple[Dict[int, float], List[int]]:
    line_confs: Dict[int, float] = {}
    missing_lines: List[int] = list(line_indices)

    if not raw:
        for idx in line_indices:
            line_confs[idx] = fallback
        return line_confs, missing_lines

    code_blocks = re.findall(r"```(?:\w+)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    candidate = code_blocks[-1].strip() if code_blocks else ""
    if not candidate:
        m = _LIST_FLOAT.search(raw)
        candidate = m.group(0) if m else ""

    if candidate:
        try:
            floats = [max(0.0, min(1.0, float(x)))
                      for x in re.findall(r"[0-9]*\.?[0-9]+", candidate)]
        except (ValueError, TypeError):
            floats = []
    else:
        floats = []

    missing_lines = []
    for i, idx in enumerate(line_indices):
        if i < len(floats):
            line_confs[idx] = floats[i]
        else:
            line_confs[idx] = fallback
            missing_lines.append(idx)

    return line_confs, missing_lines


#splits newly generated python code from its inline confidence ratings
def parse_verbalized_response(raw: str) -> Tuple[str, Dict[int, float], List[int]]:
    if not raw:
        return "", {}, []

    text = raw.strip()
    m = re.search(r"^```(?:python)?\s*\n(.*?)\n```$", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()

    clean_lines: List[str] = []
    line_confs: Dict[int, float] = {}
    missing_lines: List[int] = []

    for idx, line in enumerate(text.split("\n")):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            clean_lines.append(line)
            continue

        match = _CONF_RE.search(line)
        if match:
            score = max(0.0, min(1.0, float(match.group(1))))
            clean_line = line[: match.start()].rstrip()
            line_confs[idx] = score
        else:
            clean_line = line
            missing_lines.append(idx)

        clean_lines.append(clean_line)

    clean_code = "\n".join(clean_lines)
    return clean_code, line_confs, missing_lines

#executes generated python code locally to see if it actually works
def run_code_tests(
    func_code: str,
    entry_point: str,
    test_body: str = "",
    test_list: Optional[List[str]] = None,
    test_imports: Optional[List[str]] = None,
    timeout: int = _CODE_EXEC_TIMEOUT,
) -> Tuple[bool, str]:
    if not func_code:
        return False, "empty code"

    imports_code = "\n".join(test_imports or [])

    if test_body:
        if "def check(" in test_body:
            script = "\n".join([
                "import sys; from typing import *",
                imports_code,
                func_code,
                test_body,
                f"check({entry_point})",
                'print("__PASS__")',
            ])
        else:
            script = "\n".join([
                "import sys; from typing import *",
                imports_code,
                func_code,
                test_body,
                'print("__PASS__")',
            ])
    elif test_list:
        script = "\n".join([
            "import sys; from typing import *",
            imports_code,
            func_code,
            *test_list,
            'print("__PASS__")',
        ])
    else:
        return True, ""

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and "__PASS__" in result.stdout:
            return True, ""
        err = (result.stderr or result.stdout).strip()
        return False, err[-500:] if len(err) > 500 else err
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s (likely infinite loop in generated code)"
    except Exception as exc:
        return False, f"runner error: {exc}"



#packages line level confidence scores into a structured array
def build_line_array(line_confs: Dict[int, float], passed: bool) -> np.ndarray:
    outcome = 1.0 if passed else 0.0
    rows = [
        [float(line_idx), conf, outcome]
        for line_idx, conf in sorted(line_confs.items())
    ]
    if not rows:
        return np.empty((0, 3), dtype=np.float32)
    return np.array(rows, dtype=np.float32)



#calculates average and worst case confidence across all rated lines
def compute_verbalized_metrics(line_confs: Dict[int, float]) -> dict:
    if not line_confs:
        return dict(mean_conf=0.0, min_conf=0.0, max_conf=0.0,
                    n_lines_scored=0, line_confs_json="{}")

    vals = list(line_confs.values())
    return dict(
        mean_conf=round(sum(vals) / len(vals), 4),
        min_conf=round(min(vals), 4),
        max_conf=round(max(vals), 4),
        n_lines_scored=len(vals),
        line_confs_json=json.dumps(
            {str(k): round(v, 3) for k, v in sorted(line_confs.items())}
        ),
    )



#getting confidence and testing python code for a single example
def score_example(
    client,
    example: dict,
    model: str = "gemma-3-27b-it",
    max_tokens: int = 512,
    generator_fn=None,
    generated_code: Optional[str] = None,
    test_result: Optional[Tuple[bool, str]] = None,
) -> dict:
    if generated_code is not None:
        code_lines = generated_code.splitlines()
        scored_line_indices = [
            i for i, l in enumerate(code_lines)
            if l.strip() and not l.strip().startswith("#")
        ]
        scored_lines = [code_lines[i] for i in scored_line_indices]

        ref_prompt = reflective_prompt(generated_code, scored_lines)
        raw, ref_logprob = generate_reference(
            client, ref_prompt, model=model, max_tokens=max_tokens
        )
        line_confs, missing_lines = parse_reflective_response(
            raw, scored_line_indices, fallback=PY_FALLBACK_BASE_RATE
        )
        clean_code = generated_code
    else:
        prompt = verbalized_prompt(example["prompt"])
        if generator_fn is not None:
            raw, ref_logprob = generator_fn(prompt, max_tokens)
        else:
            raw, ref_logprob = generate_reference(
                client, prompt, model=model, max_tokens=max_tokens
            )
        clean_code, line_confs, missing_lines = parse_verbalized_response(raw)
        for idx in missing_lines:
            line_confs.setdefault(idx, PY_FALLBACK_BASE_RATE)

    if test_result is not None:
        passed, error_msg = test_result
    else:
        passed, error_msg = run_code_tests(
            func_code=clean_code,
            entry_point=example.get("entry_point", ""),
            test_body=example.get("test_body", ""),
            test_list=example.get("test_list") or [],
            test_imports=example.get("test_imports") or [],
        )

    annotation_note = (
        f"base-rate fallback applied to {len(missing_lines)} lines"
        if missing_lines else ""
    )

    metrics = compute_verbalized_metrics(line_confs)
    line_array = build_line_array(line_confs, passed)

    return {
        "task_id":          example["task_id"],
        "entry_point":      example.get("entry_point", ""),
        "passed":           passed,
        "error":            error_msg,
        "annotation_note":  annotation_note,
        "raw_response":     raw,
        "clean_code":       clean_code,
        "ref_logprob":      round(ref_logprob, 6) if ref_logprob == ref_logprob else "",
        "line_array":       line_array,
        **metrics,
    }

#asks for new sql code with confidence ratings as inline comments
def verbalized_sql_prompt(
    question: str,
    db_id: str,
    schema: Optional[str],
    evidence: Optional[str],
) -> str:
    parts = []
    if schema:
        parts.append(f"Schema:\n{schema}")
    parts.append(f"Database: {db_id}")
    parts.append(f"Question: {question}")
    if evidence:
        parts.append(f"Context: {evidence}")
    context = "\n".join(parts)

    instruction = textwrap.dedent("""\
        You are an expert SQL programmer.

        Write a SQLite SQL query to answer the question below.
        Output ONLY the SQL query, no explanation, no markdown.

        IMPORTANT: every non-blank line must end with an inline SQL comment in EXACTLY this format:
            -- conf: 0.XX
        where 0.XX is your confidence for that line (0.00 = very uncertain, 1.00 = fully certain).
        Lines that lack this annotation will be treated as a failure.

        Example of correctly formatted output:
            SELECT name         -- conf: 0.97
            FROM users          -- conf: 0.95
            WHERE age > 18;     -- conf: 0.90

        Do NOT skip the annotation on any non-blank line.
        Do NOT add any other comments or explanation.
    """)
    return instruction + "\n" + context


#asks the model to add inline confidence comments to an existing sql query
def annotate_sql_prompt(sql: str) -> str:
    instruction = textwrap.dedent("""\
        You are an expert SQL programmer.

        The following SQL query has already been written.
        Annotate every non-blank line with an inline SQL comment in EXACTLY this format:
            -- conf: 0.XX
        where 0.XX is your confidence for that line (0.00 = very uncertain, 1.00 = fully certain).

        Return the complete annotated SQL. No markdown, no explanation.

        Example:
            SELECT name         -- conf: 0.97
            FROM users          -- conf: 0.95
            WHERE age > 18;     -- conf: 0.90

        Do NOT skip the annotation on any non-blank line.
        Do NOT add any other comments or explanation.

        SQL to annotate:
    """)
    return instruction + "\n" + sql



_SQL_CONF_RE = re.compile(r"--\s*conf:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)


#splits newly generated sql code from its inline confidence ratings
def parse_verbalized_sql_response(raw: str) -> Tuple[str, Dict[int, float], List[int]]:
    if not raw:
        return "", {}, []

    text = raw.strip()
    m = re.search(r"^```(?:sql)?\s*\n(.*?)\n```$", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()

    clean_lines: List[str] = []
    line_confs: Dict[int, float] = {}
    missing_lines: List[int] = []

    for idx, line in enumerate(text.split("\n")):
        stripped = line.strip()
        if not stripped:
            clean_lines.append(line)
            continue

        match = _SQL_CONF_RE.search(line)
        if match:
            score = max(0.0, min(1.0, float(match.group(1))))
            clean_line = line[: match.start()].rstrip()
            line_confs[idx] = score
        else:
            clean_line = line
            missing_lines.append(idx)

        clean_lines.append(clean_line)

    clean_sql = postprocess_sql("\n".join(clean_lines))
    return clean_sql, line_confs, missing_lines

#compares generated sql to the known right answer using token matching
def _sql_matches_gold(generated: str, gold: str) -> bool:
    return tokenize_sql(generated.lower()) == tokenize_sql(gold.lower())

#gets confidence and testing sql code for a single example
def score_sql_example(
    client,
    example: dict,
    model: str = "gemma-3-27b-it",
    max_tokens: int = 512,
    generator_fn=None,
    generated_sql: Optional[str] = None,
    check_result: Optional[Tuple[bool, str]] = None,
) -> dict:
    gold = (example.get("gold_sql") or "").strip()

    if generated_sql is not None:
        sql_lines_raw = generated_sql.splitlines()
        scored_line_indices = [i for i, l in enumerate(sql_lines_raw) if l.strip()]
        scored_lines = [sql_lines_raw[i] for i in scored_line_indices]

        ref_prompt = reflective_sql_prompt(
            generated_sql, scored_lines,
            question=example.get("question", ""),
            schema=example.get("schema") or "",
            evidence=example.get("evidence") or "",
        )
        raw, ref_logprob = generate_reference(
            client, ref_prompt, model=model, max_tokens=max_tokens
        )
        line_confs, missing_lines = parse_reflective_response(
            raw, scored_line_indices, fallback=SQL_FALLBACK_BASE_RATE
        )
        clean_sql = generated_sql
    else:
        prompt = verbalized_sql_prompt(
            question=example["question"],
            db_id=example.get("db_id", ""),
            schema=example.get("schema"),
            evidence=example.get("evidence"),
        )
        if generator_fn is not None:
            raw, ref_logprob = generator_fn(prompt, max_tokens)
        else:
            raw, ref_logprob = generate_reference(client, prompt, model=model, max_tokens=max_tokens)
        clean_sql, line_confs, missing_lines = parse_verbalized_sql_response(raw)

    for idx in missing_lines:
        line_confs.setdefault(idx, SQL_FALLBACK_BASE_RATE)

    if check_result is not None:
        passed, error_msg = check_result
    elif not clean_sql:
        passed = False
        error_msg = "empty SQL generated"
    elif gold:
        passed = _sql_matches_gold(clean_sql, gold)
        error_msg = "" if passed else "SQL does not match gold"
    else:
        passed = False
        error_msg = "no gold_sql available"

    annotation_note = (
        f"base-rate fallback applied to {len(missing_lines)} lines"
        if missing_lines else ""
    )

    metrics = compute_verbalized_metrics(line_confs)
    line_array = build_line_array(line_confs, passed)

    return {
        "example_id":       example.get("example_id", ""),
        "db_id":            example.get("db_id", ""),
        "difficulty":       example.get("difficulty") or "",
        "question":         example.get("question", ""),
        "gold_sql":         gold,
        "passed":           passed,
        "error":            error_msg,
        "annotation_note":  annotation_note,
        "raw_response":     raw,
        "clean_sql":        clean_sql,
        "ref_logprob":      round(ref_logprob, 6) if ref_logprob == ref_logprob else "",
        "line_array":       line_array,
        **metrics,
    }
