import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import Mapping

import torch
from safetensors.torch import load_file as load_safetensors

QUANT_SUFFIXES = (
    "weight_scale",
    "weight_shift",
    "activation_scale",
    "activation_shift",
)


def resolve_checkpoint_paths(paths: list[Path], recursive: bool) -> list[Path]:
    """Expand provided paths into concrete checkpoint files."""
    resolved: list[Path] = []
    for path in paths:
        if path.is_file():
            resolved.append(path)
            continue
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
        direct_candidate = path / "model.safetensors"
        if direct_candidate.exists():
            resolved.append(direct_candidate)
            continue
        direct_candidate = path / "pytorch_model.bin"
        if direct_candidate.exists():
            resolved.append(direct_candidate)
            continue
        if recursive:
            resolved.extend(sorted(path.rglob("model.safetensors")))
            resolved.extend(sorted(path.rglob("pytorch_model.bin")))
    return resolved


def load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    """Load a checkpoint into CPU memory."""
    if path.suffix == ".safetensors":
        return load_safetensors(path)
    state = torch.load(path, map_location="cpu")
    # Hugging Face checkpoints sometimes wrap state_dict
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def classify_quant_param(name: str) -> str | None:
    """Return quantization param type if a name matches known suffixes."""
    for suffix in QUANT_SUFFIXES:
        if name.lower().endswith(suffix):
            return suffix
    return None


@dataclass
class TensorSummary:
    name: str
    kind: str
    shape: tuple[int, ...]
    numel: int
    finite_frac: float
    zero_frac: float
    nan_frac: float
    inf_frac: float
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    finite_count: int
    zero_count: int
    nan_count: int
    inf_count: int
    sum: float
    sumsq: float


def summarize_tensor(name: str, tensor: torch.Tensor, kind: str) -> TensorSummary:
    data = tensor.detach().float()
    numel = data.numel()
    finite_mask = torch.isfinite(data)
    finite_count = int(finite_mask.sum().item())
    nan_count = int(torch.isnan(data).sum().item())
    inf_count = int(torch.isinf(data).sum().item())
    zero_count = int((data == 0).sum().item())
    min_val = max_val = mean = std = None
    sum_val = sumsq_val = 0.0
    if finite_count > 0:
        finite_vals = data[finite_mask]
        min_val = float(finite_vals.min().item())
        max_val = float(finite_vals.max().item())
        sum_val = float(finite_vals.sum().item())
        sumsq_val = float((finite_vals**2).sum().item())
        mean = sum_val / finite_count
        var = max(sumsq_val / finite_count - mean**2, 0.0)
        std = var**0.5
    finite_frac = finite_count / numel if numel else 0.0
    zero_frac = zero_count / numel if numel else 0.0
    nan_frac = nan_count / numel if numel else 0.0
    inf_frac = inf_count / numel if numel else 0.0
    return TensorSummary(
        name=name,
        kind=kind,
        shape=tuple(data.shape),
        numel=int(numel),
        finite_frac=finite_frac,
        zero_frac=zero_frac,
        nan_frac=nan_frac,
        inf_frac=inf_frac,
        min=min_val,
        max=max_val,
        mean=mean,
        std=std,
        finite_count=finite_count,
        zero_count=zero_count,
        nan_count=nan_count,
        inf_count=inf_count,
        sum=sum_val,
        sumsq=sumsq_val,
    )


@dataclass
class AggregateStats:
    tensors: int = 0
    numel: int = 0
    finite: int = 0
    zero: int = 0
    nan: int = 0
    inf: int = 0
    min: float | None = None
    max: float | None = None
    sum: float = 0.0
    sumsq: float = 0.0

    def update(self, summary: TensorSummary) -> None:
        self.tensors += 1
        self.numel += summary.numel
        self.finite += summary.finite_count
        self.zero += summary.zero_count
        self.nan += summary.nan_count
        self.inf += summary.inf_count
        self.sum += summary.sum
        self.sumsq += summary.sumsq
        if summary.min is not None:
            self.min = summary.min if self.min is None else min(self.min, summary.min)
            self.max = summary.max if self.max is None else max(self.max, summary.max)

    def finalize(self) -> dict[str, float | int | None]:
        mean = None
        std = None
        if self.finite:
            mean = self.sum / self.finite
            var = max(self.sumsq / self.finite - mean**2, 0.0)
            std = var**0.5
        return {
            "tensors": self.tensors,
            "numel": self.numel,
            "finite_frac": self.finite / self.numel if self.numel else 0.0,
            "zero_frac": self.zero / self.numel if self.numel else 0.0,
            "nan_frac": self.nan / self.numel if self.numel else 0.0,
            "inf_frac": self.inf / self.numel if self.numel else 0.0,
            "min": self.min,
            "max": self.max,
            "mean": mean,
            "std": std,
        }


def collect_quant_summaries(state: Mapping[str, torch.Tensor]) -> list[TensorSummary]:
    summaries: list[TensorSummary] = []
    for name, tensor in state.items():
        kind = classify_quant_param(name)
        if kind is None:
            continue
        summaries.append(summarize_tensor(name, tensor, kind))
    summaries.sort(key=lambda item: item.name)
    return summaries


def aggregate_by_kind(summaries: Iterable[TensorSummary]) -> dict[str, AggregateStats]:
    agg: dict[str, AggregateStats] = {}
    for summary in summaries:
        group = agg.setdefault(summary.kind, AggregateStats())
        group.update(summary)
    return agg


def print_summary(
    checkpoint: Path,
    summaries: list[TensorSummary],
    limit: int | None,
) -> None:
    print(f"\n== {checkpoint} ==")
    if not summaries:
        print("No quantization parameters found.")
        return
    aggregate = aggregate_by_kind(summaries)
    print("Per-type stats:")
    for kind, stats in sorted(aggregate.items()):
        final = stats.finalize()
        print(
            f"- {kind}: tensors={final['tensors']} numel={final['numel']} "
            f"finite={final['finite_frac']:.3f} zero={final['zero_frac']:.3f} "
            f"nan={final['nan_frac']:.3f} inf={final['inf_frac']:.3f} "
            f"min={final['min']} max={final['max']} mean={final['mean']} std={final['std']}"
        )

    if limit is not None:
        summaries = summaries[:limit]
    print("\nSample tensors:")
    for summary in summaries:
        print(
            f"  {summary.name} shape={summary.shape} kind={summary.kind} "
            f"min={summary.min} max={summary.max} mean={summary.mean} "
            f"std={summary.std} zero_frac={summary.zero_frac:.3f} "
            f"nan_frac={summary.nan_frac:.3f} inf_frac={summary.inf_frac:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze saved quantization parameters in model checkpoints.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Checkpoint file or directory containing model.safetensors/pytorch_model.bin.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search directories for checkpoints.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of per-tensor entries to display (set to 0 for none).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to save JSON with per-tensor and per-type stats.",
    )
    args = parser.parse_args()

    checkpoints = resolve_checkpoint_paths(args.paths, args.recursive)
    if not checkpoints:
        raise SystemExit("No checkpoint files found.")

    all_results = []
    for ckpt in checkpoints:
        state = load_state_dict(ckpt)
        summaries = collect_quant_summaries(state)
        limit = (
            None if args.limit is None or args.limit > len(summaries) else args.limit
        )
        print_summary(ckpt, summaries, limit)
        if args.json_output:
            aggregate = {
                kind: stats.finalize()
                for kind, stats in aggregate_by_kind(summaries).items()
            }
            all_results.append(
                {
                    "checkpoint": str(ckpt),
                    "per_tensor": [asdict(s) for s in summaries],
                    "per_type": aggregate,
                }
            )

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(all_results, indent=2))
        print(f"\nSaved JSON summary to {args.json_output}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
