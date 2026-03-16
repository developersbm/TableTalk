#done by Sebastian Bastida Marin and Rei Shindo

#This file calculates and prints sql pipeline result statistics
#_load reads records from a jsonl file
#_mean calculates the average of a list of numbers
#_conf_means averages confidence scores across all token rows
#_print_section prints a formatted section of statistics
#main runs the stats pipeline and handles arguments
import json, argparse, sys
from collections import defaultdict

_VERB   = 1
_TOK    = 2
_CONS   = 3
_WB     = 4


#this reads records from a jsonl file
def _load(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


#this calculates the average of a list of numbers
def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


#this averages confidence scores across all token rows
def _conf_means(records: list[dict]) -> dict[str, float]:
    verb, tok, cons, wb = [], [], [], []
    for r in records:
        for row in r.get("token_array") or []:
            if len(row) >= 5:
                verb.append(row[_VERB])
                tok.append(row[_TOK])
                cons.append(row[_CONS])
                wb.append(row[_WB])
    return {
        "verbalized": _mean(verb),
        "tokenized":  _mean(tok),
        "consistency": _mean(cons),
        "whitebox":   _mean(wb),
    }


#this prints a formatted section of statistics
def _print_section(label: str, records: list[dict], total: int) -> None:
    pct = lambda n: f"{100*n/total:.1f}%" if total else "—"
    conf = _conf_means(records)
    n = len(records)
    print(f"  {label:<28} {n:>4}  ({pct(n)})   "
          f"verb={conf['verbalized']:.3f}  tok={conf['tokenized']:.3f}  "
          f"cons={conf['consistency']:.3f}  wb={conf['whitebox']:.3f}")


#this runs the stats pipeline and handles arguments
def main():
    p = argparse.ArgumentParser(description="SQL pipeline result statistics")
    p.add_argument("--input", required=True, help="JSONL output file")
    p.add_argument("--by-db", action="store_true", help="Break down results per database")
    p.add_argument("--failed", action="store_true", help="List all failed example IDs")
    args = p.parse_args()

    records = _load(args.input)
    total = len(records)
    if total == 0:
        print("No records found.")
        sys.exit(1)

    passed      = [r for r in records if r["passed"]]
    fixed       = [r for r in records if not r["passed"] and r.get("fix_passed")]
    failed_recs = [r for r in records if not r["passed"] and not r.get("fix_passed")]
    needed_fix  = [r for r in records if not r["passed"]]
    usable      = passed + fixed

    pct = lambda n: f"{100*n/total:.1f}%"
    fix_rate = (f"{len(fixed)}/{len(needed_fix)} "
                f"({100*len(fixed)/len(needed_fix):.1f}%)" if needed_fix else "—")

    print(f"SQL Pipeline Stats  —  {total} examples   [{args.input}]")
    print(f"{'Category':<28} {'N':>4}   {'%':>6}   {'verb':>6}  {'tok':>6}  {'cons':>6}  {'wb':>6}")
    _print_section("Passed (base)", passed, total)
    _print_section("Needed fix", needed_fix, total)
    _print_section("  ├─ Fixed", fixed, total)
    _print_section("  └─ Failed (discard)",failed_recs, total)
    _print_section("Final usable (EX acc)",usable,      total)
    print(f"\nFix success rate: {fix_rate}")
    print(f"EX accuracy: {len(usable)}/{total}  ({pct(len(usable))})")

    if args.by_db:
        #this breaks down results per database
        by_db: dict[str, list] = defaultdict(list)
        for r in records:
            by_db[r.get("db_id", "?")].append(r)

        print(f"{'Database':<26} {'N':>4}  {'Pass':>4}  {'Fixed':>5}  {'Fail':>5}  {'EX%':>6}")
        for db in sorted(by_db):
            db_recs = by_db[db]
            n       = len(db_recs)
            p_      = sum(1 for r in db_recs if r["passed"])
            fx      = sum(1 for r in db_recs if not r["passed"] and r.get("fix_passed"))
            fa      = n - p_ - fx
            ex_pct  = f"{100*(p_+fx)/n:.0f}%" if n else "—"
            print(f"{db:<26} {n:>4}  {p_:>4}  {fx:>5}  {fa:>5}  {ex_pct:>6}")

    if args.failed and failed_recs:
        print(f"\nFailed examples ({len(failed_recs)}):")
        for r in failed_recs:
            print(f"[{r.get('example_id','?'):>4}]  {r.get('db_id','?')}")

    print()


if __name__ == "__main__":
    main()
