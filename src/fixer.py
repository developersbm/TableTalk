#done by Sebastian Bastida Marin, Rei Shindo, Logan Mifflin
#This file builds fixer prompts and retries code or sql repairs

# Functions:
#_extract_failing_assertion gets the failing test line from an error
#_code_fix_prompt builds the python repair prompt
#_sql_fix_prompt builds the sql repair prompt
#_compact_block trims long text for prompts
#_extract_helpers gets helper defs from old code
#_extract_function_block gets the target function block
#fix_code retries python fixes and reruns tests
#fix_sql retries sql fixes and checks against gold

import re
import textwrap
from pathlib import Path
from typing import Tuple

from src.consistency_scoring import generate_reference, postprocess_code, postprocess_sql, quote_unquoted_identifiers, tokenize_sql
from src.verbalized_confidence import run_code_tests
from src.loader import exec_sql, results_match

#code and SQL fixers using the Gemini model, with iterative repair and test feedback.
def _extract_failing_assertion(test_code: str, error: str) -> str:
    if not test_code or not error:
        return ""
    m = re.search(r'line\s+(\d+)', error)
    if not m:
        return ""
    lineno = int(m.group(1))
    lines = test_code.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""

#build the python repair prompt
def _code_fix_prompt(
    failed_code: str,
    error: str,
    original_problem: str,
    entry_point: str,
    test_code: str = "",
) -> str:
    problem_ctx = _compact_block(original_problem, max_lines=80, max_chars=2600)
    error_ctx = _compact_block(error, max_lines=24, max_chars=1200)

    candidate = _extract_function_block(failed_code, entry_point)
    if not candidate:
        candidate = failed_code
    candidate_ctx = _compact_block(candidate, max_lines=120, max_chars=3600)
    if not candidate_ctx:
        candidate_ctx = f"def {entry_point}(...)"

    tests_ctx = _compact_block(test_code, max_lines=35, max_chars=2400)
    test_section = f"\nTEST SNIPPET (truncated):\n{tests_ctx}" if tests_ctx else ""

    failing_line = _extract_failing_assertion(test_code, error)
    failing_section = f"\nFAILING ASSERTION:\n    {failing_line}" if failing_line else ""

    target = f"def {entry_point}(...)" if entry_point else "one function definition"
    return textwrap.dedent(f"""\
        You are an expert Python debugger.

        Repair the candidate solution.

        STRICT OUTPUT RULES:
        - Output ONLY Python code (no markdown, no prose).
        - Output {target} as the main function.
        - Do not output tests, examples, or imports.
        - Do not add any inline comments (#) or docstrings.
        - Preserve the required signature from the problem statement.
        - You MUST define every helper function you call. Never call a function that is not a Python builtin or standard library unless you define it yourself ABOVE the main function. If your solution calls X(), write 'def X(...)' first.
        - Keep unrelated code unchanged when possible; make the smallest fix that passes tests.
        - If the previous code is unusable, rewrite the function from scratch.

        PROBLEM:
        {problem_ctx}

        PREVIOUS CANDIDATE:
        {candidate_ctx}

        LAST FAILURE:
        {error_ctx}
        {failing_section}
        {test_section}
        Return ONLY the corrected Python code (main function plus any helper functions it needs).
    """)

#build the sql repair prompt
def _sql_fix_prompt(
    failed_sql: str,
    error: str,
    question: str,
    schema: str = "",
    evidence: str = "",
    gold_sql: str = "",
) -> str:
    context_parts = [f"QUESTION: {question}"]
    if schema:
        context_parts.append(f"SCHEMA:\n{schema}")
    if evidence:
        context_parts.append(f"EVIDENCE: {evidence}")
    if gold_sql:
        context_parts.append(f"EXPECTED OUTPUT (gold SQL for reference):\n{gold_sql}")
    context = "\n".join(context_parts)

    return textwrap.dedent(f"""\
        You are an expert SQLite programmer. Fix the SQL query below.

        RULES (follow strictly):
        - Output ONLY the corrected SQL. No markdown, no explanation.
        - Do NOT use backticks. Use double quotes for identifiers that contain
          spaces or special characters (e.g. "Free Meal Count (K-12)").
        - For plain identifiers (no spaces/special chars) write them bare, e.g.
          frpm.CDSCode — NOT "frpm.CDSCode" (never wrap a table.column dotted
          reference in a single pair of quotes).
        - In queries with JOINs, always qualify column names with their table
          alias to avoid "ambiguous column name" errors, e.g. T1.CDSCode.
        - String literals use single quotes: 'value'.

        ERROR:
        {error}

        FAILING SQL:
        {failed_sql}

        {context}

        Return ONLY the corrected SQL query.
    """)

#trim long text for prompts
def _compact_block(text: str, max_lines: int, max_chars: int, max_line_chars: int = 200) -> str:
    if not text:
        return ""

    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    if max_line_chars > 0:
        compact_lines = []
        for ln in lines:
            if len(ln) > max_line_chars:
                compact_lines.append(ln[:max_line_chars] + " …")
            else:
                compact_lines.append(ln)
        lines = compact_lines

    if len(lines) > max_lines:
        keep_head = max_lines // 2
        keep_tail = max_lines - keep_head
        omitted = len(lines) - max_lines
        lines = (
            lines[:keep_head]
            + [f"... ({omitted} lines omitted) ..."]
            + lines[-keep_tail:]
        )

    joined = "\n".join(lines).strip()
    if len(joined) > max_chars:
        keep_head = max_chars // 2
        keep_tail = max_chars - keep_head - 32
        joined = (
            joined[:keep_head].rstrip()
            + "\n... (content truncated) ...\n"
            + joined[-max(0, keep_tail):].lstrip()
        )
    return joined.strip()

#extract from old code to preserve any useful context for the fixer without
def _extract_helpers(code: str, entry_point: str) -> str:
    if not code or not entry_point:
        return ""
    lines = code.strip().splitlines()
    sections: list[str] = []
    i = 0
    while i < len(lines):
        m = re.match(r'^(def|class)\s+(\w+)[\s\(:]', lines[i])
        if m:
            fn_name = m.group(2)
            end = len(lines)
            for j in range(i + 1, len(lines)):
                if re.match(r'^(def|class)\s+\w+', lines[j]):
                    end = j
                    break
            if fn_name != entry_point:
                sections.append("\n".join(lines[i:end]).rstrip())
            i = end
        else:
            i += 1
    return "\n\n".join(sections).strip()

#extract the function block to avoid feeding huge junk back into the fixer prompt
def _extract_function_block(code: str, entry_point: str) -> str:
    if not code:
        return ""

    lines = code.strip().splitlines()
    if not lines:
        return ""

    start_idx = None
    if entry_point:
        pat = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(")
        for i, ln in enumerate(lines):
            if pat.match(ln):
                start_idx = i
                break

    if start_idx is None:
        for i, ln in enumerate(lines):
            if re.match(r"^\s*def\s+\w+\s*\(", ln):
                start_idx = i
                break

    if start_idx is None:
        return ""

    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        if re.match(r"^(def|class)\s+\w+", lines[j]):
            end_idx = j
            break

    return "\n".join(lines[start_idx:end_idx]).strip()

#generate a fix using the model and return the text and token logprobs for confidence scoring
def fix_code(
    client,
    failed_code: str,
    error: str,
    example: dict,
    fixer_model: str,
    max_tokens: int = 1024,
    max_attempts: int = 3,
) -> Tuple[str, bool, int, str]:
    current_code = failed_code
    last_error = error
    entry_point = (example.get("entry_point") or "").strip()

    test_body  = example.get("test_body", "")
    test_list  = example.get("test_list") or []
    if test_body:
        test_code = test_body
    elif test_list:
        test_code = "\n".join(test_list)
    else:
        test_code = ""

    for attempt in range(1, max_attempts + 1):
        prompt = _code_fix_prompt(
            current_code,
            last_error,
            example.get("prompt", ""),
            entry_point=entry_point,
            test_code=test_code,
        )
        raw, _ = generate_reference(client, prompt, model=fixer_model, max_tokens=max_tokens)
        fixed_full = postprocess_code(raw)

        if not fixed_full:
            last_error = "fixer produced empty code"
            continue
        if entry_point and not re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", fixed_full, re.MULTILINE):
            last_error = f"fixer did not return entry point '{entry_point}'"
            current_code = fixed_full
            continue

        helpers = _extract_helpers(failed_code, entry_point)
        helpers_in_fix = {m.group(1) for m in re.finditer(r'^def (\w+)', fixed_full, re.MULTILINE)}
        if helpers:
            filtered_helpers = []
            for block in helpers.split("\n\n"):
                fn_m = re.match(r'def (\w+)', block.strip())
                if fn_m and fn_m.group(1) not in helpers_in_fix:
                    filtered_helpers.append(block)
            helpers_for_test = "\n\n".join(filtered_helpers)
        else:
            helpers_for_test = ""
        test_subject = (helpers_for_test + "\n\n" + fixed_full).strip() if helpers_for_test else fixed_full

        passed, err = run_code_tests(
            func_code=test_subject,
            entry_point=example.get("entry_point", ""),
            test_body=example.get("test_body", ""),
            test_list=example.get("test_list") or [],
            test_imports=example.get("test_imports") or [],
        )
        if passed:
            return test_subject, True, attempt, ""

        last_error = err
        current_code = fixed_full

    return current_code, False, max_attempts, last_error

def fix_sql(
    client,
    failed_sql: str,
    error: str,
    example: dict,
    fixer_model: str,
    max_tokens: int = 512,
    max_attempts: int = 3,
) -> Tuple[str, bool, int, str]:
    current_sql = failed_sql
    last_error = error
    gold = (example.get("gold_sql") or "").strip()

    for attempt in range(1, max_attempts + 1):
        prompt = _sql_fix_prompt(
            current_sql,
            last_error,
            example.get("question", ""),
            example.get("schema", ""),
            example.get("evidence", ""),
            gold_sql=gold,
        )
        raw, _ = generate_reference(client, prompt, model=fixer_model, max_tokens=max_tokens)
        fixed = postprocess_sql(raw)
        fixed = quote_unquoted_identifiers(fixed, example.get("schema", ""))

        if not fixed:
            last_error = "fixer produced empty SQL"
            continue

        if gold:
            db_path = example.get("db_path")
            if db_path and Path(db_path).exists():
                gold_rows, gold_err = exec_sql(db_path, gold)
                pred_rows, pred_err = exec_sql(db_path, fixed)
                if pred_err:
                    passed = False
                    last_error = f"SQL execution error: {pred_err}"
                elif gold_err:
                    passed = tokenize_sql(fixed.lower()) == tokenize_sql(gold.lower())
                    last_error = "" if passed else "SQL does not match gold"
                else:
                    passed = results_match(gold_rows, pred_rows)
                    last_error = "" if passed else "Result set does not match expected output"
            else:
                passed = tokenize_sql(fixed.lower()) == tokenize_sql(gold.lower())
                last_error = "" if passed else "SQL does not match gold"
        else:
            passed, last_error = False, "no gold_sql available"

        if passed:
            return fixed, True, attempt, ""

        current_sql = fixed

    return current_sql, False, max_attempts, last_error
