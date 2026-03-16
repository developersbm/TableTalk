#done by Logan Miffin
#This file prints quick pass fix and discard stats from pipeline outputs

#_load_records reads jsonl or json and returns records
#_as_bool normalizes mixed values into true or false
#_problem_label builds a short id for failed items
#main parses args computes summary counts and prints results

import argparse
import json
from typing import Any

#this reads jsonl or json and returns records
def _load_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return records

    jsonl_ok = True
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            jsonl_ok = False
            break
        if isinstance(obj, dict):
            records.append(obj)

    if jsonl_ok:
        return records

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        #this extracts lists of dicts from dictionary values
        out: list[dict[str, Any]] = []
        for value in data.values():
            if isinstance(value, dict):
                out.append(value)
            elif isinstance(value, list):
                out.extend(item for item in value if isinstance(item, dict))
        return out

    return records


#this normalizes mixed values into true or false
def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


#this builds a short id for failed items
def _problem_label(record: dict[str, Any]) -> str:
    if record.get("task_id") is not None:
        return f"task_id={record['task_id']}"
    if record.get("problem_index") is not None:
        return f"problem_index={record['problem_index']}"
    if record.get("entry_point"):
        return f"entry_point={record['entry_point']}"
    return "unknown-record"


#this parses args computes summary counts and prints results
def main():
    p = argparse.ArgumentParser(description="Pipeline result statistics")
    p.add_argument("--input", required=True, help="JSONL or JSON file")
    args = p.parse_args()

    records = _load_records(args.input)

    total = len(records)
    if total == 0:
        print("No records found.")
        return

    passed = [r for r in records if _as_bool(r.get("passed"))]
    fixed = [
        r
        for r in records
        if (not _as_bool(r.get("passed"))) and _as_bool(r.get("fix_passed"))
    ]
    failed = [
        r
        for r in records
        if (not _as_bool(r.get("passed"))) and (not _as_bool(r.get("fix_passed")))
    ]

    needed_fix = [r for r in records if not _as_bool(r.get("passed"))]
    discarded = failed

    pct = lambda n: f"{100*n/total:.1f}%"

    print(f"  Pipeline Stats  ({total} problems)")
    print(f"  Passed (Gemma):       {len(passed):>4}  ({pct(len(passed))})")
    print(f"  Needed fixing (Qwen): {len(needed_fix):>4}  ({pct(len(needed_fix))})")
    print(f"    ├─ Fixed by Qwen:   {len(fixed):>4}  ({pct(len(fixed))})")
    print(f"    └─ Discarded:       {len(discarded):>4}  ({pct(len(discarded))})")
    print(f"  Final usable:         {len(passed)+len(fixed):>4}  ({pct(len(passed)+len(fixed))})")
    print(f"  Fix success rate:     {len(fixed)}/{len(needed_fix)}  "
          f"({100*len(fixed)/len(needed_fix):.1f}%)" if needed_fix else "")
    print()

    if discarded:
        print(f"  Discarded problems:")
        for r in discarded:
            print(f"    - {_problem_label(r)}")
        print()


if __name__ == "__main__":
    main()
