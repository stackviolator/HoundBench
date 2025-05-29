#!/usr/bin/env python3
"""
dedup_queries.py – detect and prune duplicate Cypher queries in a JSON list.

The input file must be a JSON array where each element is an object that
contains at least a "query" key (like your queries.json).

usage
-----
    # just report duplicates
    python dedup_queries.py queries.json

    # write a new file with duplicates removed
    python dedup_queries.py queries.json --output pruned.json

    # overwrite the original file in-place (be careful!)
    python dedup_queries.py queries.json --inplace
"""

import argparse
import json
import sys
from pathlib import Path
from textwrap import indent

def canonical(q: str) -> str:
    """collapse all runs of whitespace to a single space and strip ends."""
    return " ".join(q.split())

def find_duplicates(items):
    normalised = {}
    for idx, item in enumerate(items):
        key = canonical(item["query"])
        normalised.setdefault(key, []).append(idx)
    # only return keys that appear more than once
    return {k: v for k, v in normalised.items() if len(v) > 1}

def report(dupes, items):
    if not dupes:
        print("✓ no duplicate queries found.")
        return
    print(f"✗ found {len(dupes)} duplicate query strings:\n")
    for key, indices in dupes.items():
        print("query:")
        print(indent(key, "    "))
        print("    keep index:", indices[0])
        print("    remove indices:", ", ".join(map(str, indices[1:])), "\n")

def deduplicate(items, dupes):
    to_remove = {idx for indices in dupes.values() for idx in indices[1:]}
    return [item for idx, item in enumerate(items) if idx not in to_remove]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file", help="path to queries.json")
    ap.add_argument("--output", help="write deduplicated JSON here")
    ap.add_argument("--inplace", action="store_true",
                    help="overwrite the input file with deduplicated list")
    args = ap.parse_args()

    data_path = Path(args.json_file)
    if not data_path.exists():
        sys.exit(f"error: {data_path} not found")

    items = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        sys.exit("error: top-level JSON element must be a list")

    dupes = find_duplicates(items)
    report(dupes, items)

    if args.output or args.inplace:
        pruned = deduplicate(items, dupes)
        out_path = data_path if args.inplace else Path(args.output or "deduped_" + data_path.name)
        out_path.write_text(json.dumps(pruned, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n✏️  wrote deduplicated list to {out_path}")

if __name__ == "__main__":
    main()

