#!/usr/bin/env python3
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Optional

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from train_llm_fsdp import (
    add_model_args,
    add_quantization_args,
    load_quant_model,
    set_implicit_args,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_mp.config import QuantModuleConfig

DEFAULT_TASKS = (
    "arc_easy",
    "arc_challenge",
    "boolq",
    "piqa",
    "social_iqa",
    "hellaswag",
    "openbookqa",
    "winogrande",
)


def add_eval_args(parser: ArgumentParser):
    group = parser.add_argument_group("Eval Arguments")
    group.add_argument(
        "--model-path",
        default=None,
        type=Path,
        help="Path to a trained model directory. Defaults to ./output/models/<model>/<label>/best-model",
    )
    group.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Tasks to evaluate on, space or comma separated",
    )
    group.add_argument(
        "--output-dir",
        default=Path("./output"),
        type=Path,
        help="Output directory for evaluation results.",
    )
    group.add_argument(
        "--device",
        default="cuda",
        help="Device passed to model.to(device).",
    )


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    add_quantization_args(parser)
    add_model_args(parser)
    add_eval_args(parser)
    args = parser.parse_args()
    set_implicit_args(args)

    if len(args.tasks) == 1:
        args.tasks = args.tasks[0].split(",")
    args.tasks = [t for t in args.tasks if t]

    model_name_stub = args.model_name.split("/")[-1]
    if args.model_path is None:
        # Mirror finetune output layout: ./output/models/<model>/<label>/best-model
        args.model_path = (
            Path("./output") / "models" / model_name_stub / args.label / "best-model"
        )
    print(f"Eval args: {args}")

    assert args.model_path.exists(), f"Unable to find model at {args.model_path}"
    return args


class QuantizedLLM(HFLM):
    def __init__(
        self,
        model_path: Path,
        model_name: str,
        device: str,
        rconfig: Optional[QuantModuleConfig] = None,
    ):
        # TODO: Maybe pull config and use transformers.dynamic_module_utils.get_class_from_dynamic_module
        # More proper model patching without loading all pretrained weights first
        if rconfig is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True
            )
        else:
            model = load_quant_model(model_path, rconfig)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model.to(device)

        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            backend="causal",
            device=device,
            batch_size="auto:2",
            trust_remote_code=True,
            use_fast_tokenizer=False,
            dtype=torch.bfloat16,
        )


def convert_to_json_serializable(obj):
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {
            key: convert_to_json_serializable(value)
            for key, value in obj.items()
            if not key.startswith("_")
        }  # Skip private attributes
    elif hasattr(obj, "__dict__"):
        return convert_to_json_serializable(obj.__dict__)
    else:
        try:
            # Test if value is JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)  # Convert non-serializable objects to strings


def main():
    args = parse_args()

    # Create the output directory if it doesn't exist
    output_file = (
        args.output_dir
        / "eval"
        / args.model_name.split("/")[-1]
        / args.label
        / "best-model"
        / "acc_results.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        print(
            f"Results already calculated and stored in {output_file}. Skipping this eval run."
        )
        return

    # Initialize the model
    model = QuantizedLLM(
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device,
        rconfig=args.quant_module_config,
    )

    # Run evaluation
    print(f"Evaluating model on tasks: {args.tasks}")
    results = evaluator.simple_evaluate(
        model=model,
        tasks=args.tasks,  # type: ignore
        confirm_run_unsafe_code=True,
    )

    # Return if no results
    if results is None:
        return

    # Save results
    with open(output_file, "w") as f:
        json.dump(convert_to_json_serializable(results), f)
    print(f"Results saved to {output_file}")

    print("Summary:")
    for task_name, task_results in results["results"].items():
        print(f"{task_name}:")
        for metric_name, metric_value in task_results.items():
            print(f"  {metric_name}: {metric_value}")


if __name__ == "__main__":
    main()
