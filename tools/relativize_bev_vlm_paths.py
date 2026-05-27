import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite absolute paths in BEV-VLM json/jsonl files to paths relative to a project root."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="json/jsonl files to rewrite in place",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="project root used as the relative-path base",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only report how many strings would be rewritten",
    )
    return parser.parse_args()


def maybe_relativize_string(value, base_dir):
    if not isinstance(value, str):
        return value, False
    path = Path(value)
    if not path.is_absolute():
        return value, False
    try:
        relative = path.resolve().relative_to(base_dir)
    except ValueError:
        return value, False
    return relative.as_posix(), True


def relativize_payload(payload, base_dir):
    changed = 0
    if isinstance(payload, dict):
        rewritten = {}
        for key, value in payload.items():
            rewritten_value, child_changed = relativize_payload(value, base_dir)
            rewritten[key] = rewritten_value
            changed += child_changed
        return rewritten, changed
    if isinstance(payload, list):
        rewritten = []
        for value in payload:
            rewritten_value, child_changed = relativize_payload(value, base_dir)
            rewritten.append(rewritten_value)
            changed += child_changed
        return rewritten, changed
    rewritten_value, did_change = maybe_relativize_string(payload, base_dir)
    return rewritten_value, int(did_change)


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_file(path, base_dir, dry_run=False):
    if path.suffix == ".jsonl":
        payload = load_jsonl(path)
        rewritten, changed = relativize_payload(payload, base_dir)
        if not dry_run:
            save_jsonl(path, rewritten)
        return changed

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rewritten, changed = relativize_payload(payload, base_dir)
    if not dry_run:
        with path.open("w", encoding="utf-8") as f:
            json.dump(rewritten, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return changed


def main():
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    total_changed = 0
    for file_name in args.files:
        path = Path(file_name)
        changed = process_file(path, base_dir, dry_run=args.dry_run)
        total_changed += changed
        action = "would rewrite" if args.dry_run else "rewrote"
        print(f"{path}: {action} {changed} path strings")
    print(f"total_rewritten={total_changed}")


if __name__ == "__main__":
    main()
