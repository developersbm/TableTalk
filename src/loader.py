#done by Sebastian Bastida Marin
#This file loads various datasets and their sql databases for evaluation
#it handles finding files, creating schemas, and running test queries

# Functions:
#_bird_root: gets bird dataset path from environment variables
#_load_tables_index: reads table information from json files
#_schema_from_tables_entry: creates sql statements from table metadata
#_find_tables_json: locates the related tables configuration file
#_extract_schema: pulls create table statements directly from a database
#_enrich_schema: adds database path and schema to an example dictionary
#exec_sql: safely runs a sql query with a time limit
#results_match: compares two sets of query results to see if they match
#load_mini_dev: reads the mini dev dataset and connects its schemas
#load_bird_file: reads main bird datasets and attaches their databases
#_is_jsonl: checks if a file uses json lines format
#load_humaneval: loads humaneval coding tasks
#load_mbpp: loads mbpp coding tasks
#_load_evalplus: downloads and formats tasks from huggingface hub

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional


_MINI_DEV_JSONL = (
    Path(__file__).parent.parent
    / "datasets" / "mini_dev" / "finetuning" / "inference" / "mini_dev_prompt.jsonl"
)

#gets bird dataset path from environment variables
def _bird_root() -> Optional[Path]:
    v = os.getenv("BIRD_PATH", "").strip()
    return Path(v) if v else None

#reads table information from json files
def _load_tables_index(tables_json: Path) -> dict:
    if not tables_json.exists():
        return {}
    try:
        with open(tables_json, encoding="utf-8") as f:
            entries = json.load(f)
        return {e["db_id"]: e for e in entries if "db_id" in e}
    except Exception:
        return {}


#creates sql statements from table metadata
def _schema_from_tables_entry(entry: dict) -> str:
    tables: List[str] = entry.get("table_names_original", [])
    col_names_raw: list = entry.get("column_names_original", [])
    col_types: list = entry.get("column_types", [])
    pk_set: set = set()
    for pk in entry.get("primary_keys", []):
        if isinstance(pk, list):
            pk_set.update(pk)
        else:
            pk_set.add(pk)
    fk_pairs: list = entry.get("foreign_keys", [])

    if not tables or not col_names_raw:
        return ""

    table_cols: dict = {i: [] for i in range(len(tables))}
    for idx, (tbl_idx, col_name) in enumerate(col_names_raw):
        if tbl_idx == -1:
            continue
        col_type = col_types[idx] if idx < len(col_types) else "text"
        sql_type = {"integer": "INTEGER", "real": "REAL", "number": "REAL",
                    "boolean": "INTEGER", "others": "TEXT"}.get(col_type.lower(), "TEXT")
        pk_mark = " PRIMARY KEY" if idx in pk_set else ""
        table_cols[tbl_idx].append(f'  "{col_name}" {sql_type}{pk_mark}')

    parts: List[str] = []
    for tbl_idx, tbl_name in enumerate(tables):
        cols = table_cols.get(tbl_idx, [])
        if cols:
            parts.append(f'CREATE TABLE "{tbl_name}" (\n' + ",\n".join(cols) + "\n);")

    fk_lines: List[str] = []
    for col_a, col_b in fk_pairs:
        if col_a < len(col_names_raw) and col_b < len(col_names_raw):
            _, name_a = col_names_raw[col_a]
            _, name_b = col_names_raw[col_b]
            tbl_a = tables[col_names_raw[col_a][0]] if col_names_raw[col_a][0] >= 0 else "?"
            tbl_b = tables[col_names_raw[col_b][0]] if col_names_raw[col_b][0] >= 0 else "?"
            fk_lines.append(f'-- FK: "{tbl_a}"."{name_a}" → "{tbl_b}"."{name_b}"')
    if fk_lines:
        parts.append("\n".join(fk_lines))

    return "\n\n".join(parts)


#locates the related tables configuration file
def _find_tables_json(data_path: Path) -> Optional[Path]:
    stem = data_path.stem
    for name in (f"{stem}_tables.json", "tables.json"):
        candidate = data_path.parent / name
        if candidate.exists():
            return candidate
    return None


#pulls create table statements directly from a database
def _extract_schema(db_path: Path) -> Optional[str]:
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        )
        ddl = ";\n".join(row[0] for row in cur.fetchall())
        con.close()
        return ddl or None
    except Exception:
        return None


#adds database path and schema to an example dictionary
def _enrich_schema(ex: dict, db_dir: Optional[Path]) -> None:
    if db_dir is None:
        return
    db_id = ex.get("db_id", "")
    if not db_id:
        return
    for ext in (".sqlite", ".db"):
        db_path = db_dir / db_id / f"{db_id}{ext}"
        if db_path.exists():
            if not ex.get("schema"):
                ex["schema"] = _extract_schema(db_path)
            ex["db_path"] = str(db_path)
            break


#safely runs a sql query with a time limit
def exec_sql(db_path: str, sql: str, timeout: float = 30.0):

    try:
        con = sqlite3.connect(str(db_path))
        con.text_factory = lambda b: b.decode("utf-8", errors="replace")
        timer = threading.Timer(timeout, con.interrupt)
        timer.start()
        try:
            cur = con.execute(sql)
            rows = cur.fetchall()
        finally:
            timer.cancel()
        con.close()
        return rows, None
    except Exception as exc:
        return None, str(exc)[:300]


#compares two sets of query results to see if they match
def results_match(gold_rows, pred_rows) -> bool:
    try:
        norm = lambda rows: sorted(str(r) for r in (rows or []))
        return norm(gold_rows) == norm(pred_rows)
    except Exception:
        return False


#reads the mini dev dataset and connects its schemas
def load_mini_dev(limit: Optional[int] = None) -> List[dict]:
    root = _bird_root()
    if root:
        jsonl: Path = root / "mini_dev" / "finetuning" / "inference" / "mini_dev_prompt.jsonl"
        db_dir: Optional[Path] = root / "mini_dev" / "databases"
    else:
        jsonl = _MINI_DEV_JSONL
        db_dir = None

    if not jsonl.exists():
        raise FileNotFoundError(
            f"mini_dev_prompt.jsonl not found at {jsonl}\n"
            "Set BIRD_PATH in .env or ensure datasets/mini_dev/ is present."
        )

    examples: List[dict] = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if limit and len(examples) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            examples.append({
                "example_id": str(item.get("question_id", len(examples))),
                "db_id":      item.get("db_id", ""),
                "question":   (item.get("question") or "").strip(),
                "evidence":   (item.get("evidence") or "").strip() or None,
                "gold_sql":   (item.get("SQL") or "").strip() or None,
                "schema":     (item.get("schema") or "").strip() or None,
                "difficulty": item.get("difficulty"),
                "db_path":    None,
            })

    tables_json_candidates: List[Path] = []
    if root:
        tables_json_candidates += [
            root / "mini_dev" / "tables.json",
            root / "mini_dev" / "mini_dev_tables.json",
        ]
    tables_json_candidates.append(
        Path(__file__).parent.parent / "datasets" / "mini_dev" / "tables.json"
    )
    mini_tables_json: Optional[Path] = next(
        (c for c in tables_json_candidates if c.exists()), None
    )
    tables_index: dict = _load_tables_index(mini_tables_json) if mini_tables_json else {}
    if tables_index:
        print(f"[loader] mini_dev schema from {mini_tables_json}", flush=True)
    for ex in examples:
        if ex.get("schema") is None and ex["db_id"] in tables_index:
            ex["schema"] = _schema_from_tables_entry(tables_index[ex["db_id"]])

    if db_dir and db_dir.exists():
        for ex in examples:
            _enrich_schema(ex, db_dir)

    return examples


#reads main bird datasets and attaches their databases
def load_bird_file(path: str, limit: Optional[int] = None) -> List[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw: list = []
    if p.suffix == ".jsonl" or _is_jsonl(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw.append(json.loads(line))
    else:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        raw = data if isinstance(data, list) else data.get("examples", [])

    examples: List[dict] = []
    for idx, item in enumerate(raw):
        if limit and len(examples) >= limit:
            break
        examples.append({
            "example_id": str(item.get("question_id", idx)),
            "db_id":      item.get("db_id", ""),
            "question":   (item.get("question") or "").strip(),
            "evidence":   (item.get("evidence") or "").strip() or None,
            "gold_sql":   (item.get("SQL") or item.get("query") or "").strip() or None,
            "schema":     None,
            "db_path":    None,
            "difficulty": item.get("difficulty"),
        })

    tables_json = _find_tables_json(p)
    root = _bird_root()
    if tables_json is None and root:
        bird_data = root / p.parent.name / p.name
        tables_json = _find_tables_json(bird_data)
    tables_index: dict = _load_tables_index(tables_json) if tables_json else {}
    if tables_index:
        print(f"[loader] schema loaded from {tables_json} ({len(tables_index)} dbs)",
              flush=True)
    for ex in examples:
        if ex.get("schema") is None and ex["db_id"] in tables_index:
            ex["schema"] = _schema_from_tables_entry(tables_index[ex["db_id"]])

    dev_db_env = os.getenv("DEV_DB_DIR", "").strip()
    candidates: List[Path] = []
    if dev_db_env:
        candidates.append(Path(dev_db_env))
        candidates.append(Path(dev_db_env) / "train_databases")
        candidates.append(Path(dev_db_env) / "dev_databases")
    candidates += [
        p.parent / "dev_databases",
        p.parent / "databases",
    ]
    if root:
        candidates += [
            root / "dev_20240627" / "dev_databases",
            root / "databases",
        ]
    db_dir: Optional[Path] = next((c for c in candidates if c.exists()), None)
    if db_dir:
        sample_dbs = list(db_dir.glob("*/*.sqlite")) or list(db_dir.glob("*/*.db"))
        print(f"[loader] db files at {db_dir} ({len(sample_dbs)} db files found)",
              flush=True)
        if not sample_dbs:
            nested = list(db_dir.glob("*/*/*.sqlite")) or list(db_dir.glob("*/*/*.db"))
            if nested:
                db_dir = db_dir / nested[0].relative_to(db_dir).parts[0]
                print(f"[loader] adjusted db_dir to {db_dir}", flush=True)
        for ex in examples:
            _enrich_schema(ex, db_dir)
        found = sum(1 for ex in examples if ex.get("db_path"))
        print(f"[loader] db_path resolved for {found}/{len(examples)} examples",
              flush=True)
    else:
        print("[loader] WARNING: no dev_databases/ dir found — EX accuracy unavailable",
              flush=True)

    return examples


#checks if a file uses json lines format
def _is_jsonl(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    return line.startswith("{")
    except Exception:
        pass
    return False



#loads humaneval coding tasks
def load_humaneval(limit: Optional[int] = None, cache_dir: Optional[str] = None) -> List[dict]:
    return _load_evalplus("evalplus/humanevalplus", limit, cache_dir)


#loads mbpp coding tasks
def load_mbpp(limit: Optional[int] = None, cache_dir: Optional[str] = None) -> List[dict]:
    return _load_evalplus("evalplus/mbppplus", limit, cache_dir)


#downloads and formats tasks from huggingface hub
def _load_evalplus(hf_id: str, limit: Optional[int], cache_dir: Optional[str]) -> List[dict]:
    from datasets import load_dataset

    if cache_dir is None:
        cache_dir = str(
            Path(__file__).parent.parent / "datasets" / hf_id.split("/")[-1]
        )

    ds = load_dataset(hf_id, split="test", cache_dir=cache_dir, token=os.getenv("HF_TOKEN"))

    examples: List[dict] = []
    for raw in ds:
        if limit and len(examples) >= limit:
            break
        item: dict = dict(raw)
        examples.append({
            "task_id":      item.get("task_id", ""),
            "prompt":       (item.get("prompt") or "").strip(),
            "entry_point":  item.get("entry_point", ""),
            "test_body":    (item.get("test") or "").strip(),
            "test_list":    item.get("test_list") or [],
            "test_imports": item.get("test_imports") or [],
        })
    return examples
