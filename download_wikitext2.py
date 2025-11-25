#!/usr/bin/env python3
"""
Download the WikiText-2 dataset and export to JSONL.

Each output file contains one JSON object per line:
{"id": <int>, "text": <str>}

By default it uses the 'wikitext-2-raw-v1' config (untokenized, raw text).
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def export_split(ds, out_path: Path, split_name: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            # WikiText-2 has a single column "text"
            text = example.get("text", "")
            obj = {"id": i, "text": text}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count:,} lines to {out_path} ({split_name}).")


def main():
    parser = argparse.ArgumentParser(description="Export WikiText-2 to JSONL.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Directory to write JSONL files.",
    )
    parser.add_argument(
        "--config",
        default="wikitext-2-raw-v1",
        choices=["wikitext-2-raw-v1", "wikitext-2-v1"],
        help="Dataset config to use (default: wikitext-2-raw-v1)",
    )
    parser.add_argument(
        "--name",
        default="wikitext",
        choices=["wikitext"],
        help="HF dataset name (default: wikitext)",
    )

    args = parser.parse_args()

    print(f"Loading dataset '{args.name}' with config '{args.config}'...")
    dset = load_dataset(args.name, args.config)

    # Export each available split
    for split in ("train", "validation", "test"):
        if split in dset:
            out_file = args.output_dir / f"{split}.jsonl"
            export_split(dset[split], out_file, split)
        else:
            print(f"Split '{split}' not found in this config; skipping.")

    print("Done.")


if __name__ == "__main__":
    main()
