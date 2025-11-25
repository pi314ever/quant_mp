import argparse
import csv
import json
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

# Tasks and human-friendly short names for table headers
TASKS: list[str] = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "piqa",
    "social_iqa",
    "hellaswag",
    "openbookqa",
    "winogrande",
]
TASK_SHORT: dict[str, str] = {
    "arc_easy": "arc_e",
    "arc_challenge": "arc_c",
    "boolq": "boolq",
    "piqa": "piqa",
    "social_iqa": "siqa",
    "hellaswag": "hella",
    "openbookqa": "obqa",
    "winogrande": "wino",
}

# Standardized sort keys
NUMERIC_SORT_KEYS = {
    "arc_e",
    "arc_c",
    "boolq",
    "piqa",
    "siqa",
    "hella",
    "obqa",
    "wino",
    "avg",
    "wiki2",
}
STRING_SORT_KEYS = {
    "model",
    "label",
    "w_dtype",
    "w_block",
    "w_alg",
    "a_dtype",
    "a_alg",
    "a_block",
}
CANONICAL_SORT_KEYS = STRING_SORT_KEYS | NUMERIC_SORT_KEYS

SORT_ALIASES: dict[str, str] = {
    # Weights
    "dtype": "w_dtype",
    "format": "w_dtype",
    "w_format": "w_dtype",
    "block": "w_block",
    "block_size": "w_block",
    "alg": "w_alg",
    # Back-compat for old keys
    "w_dformat": "w_dtype",
    # Activations
    "act_dtype": "a_dtype",
    "act_format": "a_dtype",
    "act_alg": "a_alg",
    # Back-compat for old keys
    "a_dformat": "a_dtype",
    # Metrics
    "perplexity": "wiki2",
    "ppl": "wiki2",
    "average": "avg",
    "mean": "avg",
    # Meta
    "model_name": "model",
    "config": "label",
    "quant": "label",
    "quant_label": "label",
}
SORT_ALLOWED_DISPLAY = ",".join(sorted(CANONICAL_SORT_KEYS | set(SORT_ALIASES.keys())))

# Pretty help text for --sort, grouped by category
SORT_HELP_GROUPED = (
    "Multi-key sort. Prefix with - for desc, + for asc.\n"
    "Default direction: metrics descend; others ascend.\n\n"
    "Keys by group:\n"
    "- Model: model, label\n"
    "- Weights: w_dtype (aliases: dtype, format, w_format, w_dformat), "
    "w_alg (aliases: alg), w_block (aliases: block, block_size)\n"
    "- Activations: a_dtype (aliases: act_dtype, act_format, a_dformat), "
    "a_alg (aliases: act_alg), a_block\n"
    "- Metrics: arc_e, arc_c, boolq, piqa, siqa, hella, obqa, wino, "
    "avg (aliases: average, mean), wiki2 (aliases: perplexity, ppl)"
)

# Default non-metric columns to display (replaces showing only 'label').
META_COLUMNS_DEFAULT: list[str] = [
    "model",
    "w_dtype",
    "w_alg",
    "w_block",
    "a_dtype",
    "a_alg",
    "a_block",
]

# Metrics displayed/handled by the parser/view.
METRIC_COLUMNS: list[str] = [TASK_SHORT[t] for t in TASKS] + ["avg", "wiki2"]

# Hide group aliases for convenience
HIDE_GROUPS: dict[str, list[str]] = {
    "activations": ["a_dtype", "a_alg", "a_block"],
    "activation": ["a_dtype", "a_alg", "a_block"],
    "acts": ["a_dtype", "a_alg", "a_block"],
}


def expand_hide_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for t in tokens:
        if not t:
            continue
        exp = HIDE_GROUPS.get(t, [t])
        for item in exp:
            if item not in seen:
                out.append(item)
                seen.add(item)
    return out


def parse_hide_columns(hide_args: list[str]) -> list[str]:
    hide_raw: list[str] = []
    for h in hide_args:
        if not h:
            continue
        hide_raw.extend([p.strip() for p in h.split(",") if p.strip()])
    return expand_hide_tokens(hide_raw)


def find_model_label_dirs(output_dir: Path) -> Iterable[tuple[str, str, Path]]:
    """Yield (model_short, label, label_dir) found under output_dir.

    Expects layout: <output_dir>/<model_short>/<label>/
    Skips the special directory name 'eval' used for aggregated eval results.
    """
    if not output_dir.exists():
        return []

    for model_dir in (output_dir / "models").iterdir():
        if not model_dir.is_dir():
            continue
        for label_dir in model_dir.iterdir():
            if not label_dir.is_dir():
                continue
            yield model_dir.name, label_dir.name, label_dir


def load_eval_results_from_eval_dir(
    output_dir: Path, model_short: str, label: str
) -> Optional[dict]:
    """Try to load lm-eval accuracy JSON from <output_dir>/eval/*.json if present."""

    acc_path = (
        output_dir / "eval" / model_short / label / "best-model" / "acc_results.json"
    )
    if not acc_path.exists():
        return None
    try:
        with acc_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_wiki_eval_perplexity(
    base_output: Path, model_short: str, label: str
) -> Optional[float]:
    """Load WikiText-2 validation perplexity."""
    perplexity_path = (
        base_output
        / "eval"
        / model_short
        / label
        / "best-model"
        / "perplexity_results.json"
    )
    if not perplexity_path.exists():
        return None
    with perplexity_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "perplexity" not in data:
        return None
    return float(data.get("perplexity"))


def extract_task_acc_percent(results: dict, task: str) -> Optional[float]:
    """Get accuracy (in percent) for a given task from lm-eval results file."""
    try:
        val = results["results"][task]["acc,none"]
    except Exception:
        return None
    if val is None:
        return None
    return float(val) * 100.0


def compute_avg(values: list[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def format_value(v: Optional[float], precision: int) -> str:
    if v is None:
        return "-"
    return f"{v:.{precision}f}"


def parse_label_fields(label: str) -> dict[str, Optional[str]]:
    """Parse common attributes from a label.

    Supported patterns:
      - "Baseline" or any label containing 'baseline' (case-insensitive)
      - "W-<w_dformat>-<w_block>-<w_alg>--A-<a_dformat>-<a_alg>"

    Returns a mapping with keys: w_dtype, w_block, w_alg, a_dtype, a_alg, a_block, dtype (alias of w_dtype).
    Missing/unknown fields are set to None.
    """
    low = label.lower()
    out: dict[str, Optional[str]] = {
        "w_dtype": None,
        "w_block": None,
        "w_alg": None,
        "a_dtype": None,
        "a_alg": None,
        "a_block": None,
        "dtype": None,
    }
    if "baseline" in low:
        return out

    if label.startswith("W-") and "--A-" in label:
        try:
            w_part_full = label[len("W-") :]
            w_part, a_part = w_part_full.split("--A-", 1)
            w_bits = w_part.split("-")
            if len(w_bits) >= 3:
                out["w_dtype"] = w_bits[0] if w_bits[0] != "None" else None
                out["w_block"] = w_bits[1] if w_bits[1] != "None" else None
                out["w_alg"] = (
                    "-".join(w_bits[2:]) if "-".join(w_bits[2:]) != "None" else None
                )
            a_bits = a_part.split("-")
            if len(a_bits) >= 2:
                out["a_dtype"] = a_bits[0] if a_bits[0] != "None" else None
                out["a_alg"] = (
                    "-".join(a_bits[1:]) if "-".join(a_bits[1:]) != "None" else None
                )
        except Exception:
            pass

    out["dtype"] = out["w_dtype"]
    return out


def is_numeric_key(name: str) -> bool:
    return name in NUMERIC_SORT_KEYS


def resolve_sort_specs(sort_arg: Optional[str]) -> list[tuple[str, bool]]:
    """Parse --sort into a list of (key, descending) specs.

    Accepted keys:
      - model, label
      - dtype (alias of w_dformat), w_dformat, w_block (alias block, block_size), w_alg (alias alg)
      - a_dformat (alias act_dtype), a_alg (alias act_alg)
      - metrics: arc_e, arc_c, boolq, piqa, siqa, hella, obqa, wino, avg, wiki2

    Prefix with '-' for descending, '+' for ascending. If no prefix: metrics default to descending, others ascending.
    """
    if not sort_arg:
        return []

    specs: list[tuple[str, bool]] = []
    for raw in [s.strip() for s in sort_arg.split(",") if s.strip()]:
        desc = False
        name = raw
        if raw[0] in "+-":
            desc = raw[0] == "-"
            name = raw[1:]
        name = SORT_ALIASES.get(name, name)
        if name not in CANONICAL_SORT_KEYS:
            raise SystemExit(
                f"Unknown sort key '{name}'. Allowed: {SORT_ALLOWED_DISPLAY}"
            )
        if raw[0] not in "+-":
            # Defaults: metrics -> desc, others -> asc
            desc = is_numeric_key(name)
        specs.append((name, desc))
    return specs


def invert_str_key(s: str) -> str:
    return "".join(chr(0x10FFFF - ord(c)) for c in s)


def build_sort_key_funcs(specs: list[tuple[str, bool]]):
    def key_for_row(row: dict) -> tuple:
        parts = []
        for name, desc in specs:
            # Map name -> value from row
            if name in {
                "model",
                "label",
                "w_dtype",
                "w_block",
                "w_alg",
                "a_dtype",
                "a_alg",
                "a_block",
            }:
                v: Optional[str] = row.get(name)
                if desc:
                    parts.append(
                        (v is None, invert_str_key(v) if v is not None else "")
                    )
                else:
                    parts.append((v is None, v or ""))
            elif name in METRIC_COLUMNS:
                # numeric
                vnum: Optional[float] = get_metric_value(row, name)
                if desc:
                    parts.append((vnum is None, -(vnum if vnum is not None else 0.0)))
                else:
                    parts.append((vnum is None, vnum if vnum is not None else 0.0))
            else:
                # Unknown key, treat as label
                v2: Optional[str] = row.get(name) or row.get("label")
                if desc:
                    parts.append(
                        (v2 is None, invert_str_key(v2) if v2 is not None else "")
                    )
                else:
                    parts.append((v2 is None, v2 or ""))
        return tuple(parts)

    return key_for_row


def get_metric_value(row: dict, name: str) -> Optional[float]:
    if name in {"avg", "wiki2"}:
        return row.get(name)
    return (row.get("metrics") or {}).get(name)


def build_row_for_label(output_dir: Path, model_short: str, label: str) -> dict:
    acc_json = load_eval_results_from_eval_dir(output_dir, model_short, label)
    metrics: dict[str, Optional[float]] = {}
    for task in TASKS:
        short = TASK_SHORT[task]
        metrics[short] = (
            extract_task_acc_percent(acc_json, task) if acc_json is not None else None
        )
    avg_val = compute_avg(list(metrics.values()))
    wiki_val = load_wiki_eval_perplexity(output_dir, model_short, label)
    fields = parse_label_fields(label)
    return {
        "model": model_short,
        "label": label,
        "metrics": metrics,
        "avg": avg_val,
        "wiki2": wiki_val,
        **fields,
    }


def build_row_data(output_dir: Path) -> list[dict]:
    items = list(find_model_label_dirs(output_dir))
    if not items:
        return []

    worker_items = [(output_dir, model_short, label) for model_short, label, _ in items]

    max_workers = min(os.cpu_count() or 1, len(worker_items))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_build_row_from_tuple, worker_items))


def _build_row_from_tuple(args: tuple[Path, str, str]) -> dict:
    output_dir, model_short, label = args
    return build_row_for_label(output_dir, model_short, label)


class ResultsTable:
    """Container that stores parsed rows and handles sorting/export logic."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "ResultsTable":
        return cls(build_row_data(output_dir))

    def sort(self, specs: list[tuple[str, bool]]) -> None:
        if not specs:
            return
        keyfunc = build_sort_key_funcs(specs)
        self.rows = sorted(self.rows, key=keyfunc)

    @staticmethod
    def value_for_column(row: dict, column: str) -> Optional[object]:
        if column in METRIC_COLUMNS:
            return get_metric_value(row, column)
        return row.get(column)

    def iter_csv_rows(self, columns: list[str]) -> Iterable[list[object]]:
        for row in self.rows:
            rendered: list[object] = []
            for column in columns:
                value = self.value_for_column(row, column)
                rendered.append("" if value is None else value)
            yield rendered


class TableView:
    """Renders a ResultsTable using pretty string representations."""

    def __init__(self, table: ResultsTable, columns: list[str], precision: int = 1):
        self.table = table
        self.columns = columns
        self.precision = precision

    @property
    def headers(self) -> list[str]:
        return self.columns

    def iter_rows(self) -> Iterable[list[str]]:
        for row in self.table.rows:
            yield [self._format_cell(row, column) for column in self.columns]

    def _format_cell(self, row: dict, column: str) -> str:
        value = self.table.value_for_column(row, column)
        if column in METRIC_COLUMNS:
            metric_val = value if isinstance(value, (float, int)) else None
            return format_value(metric_val, self.precision)
        if value is None or value == "":
            return "-"
        return str(value)


def compute_col_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    return widths


def print_table(view: TableView) -> None:
    rows = list(view.iter_rows())
    headers = view.headers
    widths = compute_col_widths(headers, rows)

    def fmt_row(vals: list[str]) -> str:
        return " ".join(f"{v:>{w}}" for v, w in zip(vals, widths))

    print(fmt_row(headers))
    for row in rows:
        print(fmt_row(row))


def export_csv(table: ResultsTable, columns: list[str], path: Path) -> None:
    """Export the current table to CSV with raw values (full precision)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for row in table.iter_csv_rows(columns):
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unify parsing of lm-eval accuracies and WikiText-2 perplexity into a single table."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help=(
            "Base output directory that contains <model>/<label>/ and optionally an 'eval' folder."
        ),
    )

    parser.add_argument(
        "--hide",
        action="append",
        default=[],
        help=(
            "Comma-separated list of columns to hide. "
            "Columns: model,label,w_dtype,w_alg,w_block,a_dtype,a_alg,a_block,"
            "arc_e,arc_c,boolq,piqa,siqa,hella,obqa,wino,avg,wiki2. "
            "Groups: activations (hides a_dtype,a_alg,a_block)."
        ),
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Show only a single metric column (plus model and label). "
            "One of: arc_e,arc_c,boolq,piqa,siqa,hella,obqa,wino,avg,wiki2"
        ),
    )

    parser.add_argument(
        "--sort",
        type=str,
        default=None,
        help=SORT_HELP_GROUPED,
    )

    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help=(
            "Optional CSV export path. Writes the same table (after sorting/hiding) "
            "to the specified CSV file."
        ),
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=1,
        help="Number of decimal places to show for metrics in the console output.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Include the label column in the output table.",
    )

    return parser.parse_args()


def resolve_columns(hide_args: list[str], only: Optional[str]) -> list[str]:
    metric_columns = METRIC_COLUMNS.copy()

    if only is not None:
        only = only.strip()
        if only not in metric_columns:
            raise SystemExit(
                f"--only must be one of {','.join(metric_columns)}; got '{only}'."
            )
        return [only]

    # Expand hide lists (support group aliases)
    hide = parse_hide_columns(hide_args)

    # Support hiding model/label as well
    # Build full column order, then filter
    cols = metric_columns.copy()
    for col in ["model", "label"]:
        if col in hide:
            pass  # handled in header composition
    # Only metric columns are controlled here; model/label handled in build_rows
    cols = [c for c in cols if c not in hide]
    return cols


def main() -> None:
    args = parse_args()

    if args.precision < 0:
        raise SystemExit("--precision must be >= 0")

    include_columns = resolve_columns(args.hide, args.only)

    table = ResultsTable.from_output_dir(args.output_dir)
    table.sort(resolve_sort_specs(args.sort))

    hide_columns = set(parse_hide_columns(args.hide))
    meta_columns = META_COLUMNS_DEFAULT.copy()
    if args.show_labels and "label" not in meta_columns:
        meta_columns.insert(1, "label")

    headers = [c for c in (meta_columns + include_columns) if c not in hide_columns]
    view = TableView(table, headers, precision=args.precision)

    # Export CSV if requested
    if args.export is not None:
        export_csv(table, headers, args.export)

    # Always print the pretty table to stdout
    print_table(view)


if __name__ == "__main__":
    main()
