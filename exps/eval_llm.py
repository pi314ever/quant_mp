#!/usr/bin/env python3
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import transformers
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from run_exp_llm import QuantizationArguments, print_once
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

from quant_mp.config import QuantLinearConfig
from quant_mp.utils import patch_model


@dataclass
class EvalArguments:
    model_path: Path
    tasks: List[str] = field(
        default_factory=list,
        metadata={
            "nargs": "+",
            "help": "All tasks to evaluate on, as a comma or space separated list",
        },
    )
    output_dir: str = field(
        default="./output/eval",
        metadata={
            "help": "The output directory where the evaluation results are stored.",
        },
    )
    device: str = field(
        default="cuda",
        metadata={
            "help": "Device model will be loaded on. Passed directly to model.to(device)"
        },
    )

    def __post_init__(self):
        if len(self.tasks) == 1:
            self.tasks = self.tasks[0].split(",")


def parse_args() -> Tuple[QuantizationArguments, EvalArguments]:
    parser = transformers.HfArgumentParser(
        (QuantizationArguments, EvalArguments),  # type: ignore
    )

    quant_args, eval_args = parser.parse_args_into_dataclasses()

    return quant_args, eval_args


class QuantizedLLM(HFLM):
    def __init__(
        self,
        model_path: Path,
        device: str,
        rconfig: Optional[QuantLinearConfig] = None,
    ):
        # TODO: Maybe pull config and use transformers.dynamic_module_utils.get_class_from_dynamic_module
        # More proper model patching without loading all pretrained weights first
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        if rconfig is not None:
            patch_model(model, rconfig)
            # Manually load params for quantized model
            state_dict = {}
            for state_dict_path in model_path.glob("*.safetensors"):
                state_dict.update(load_file(state_dict_path))
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print_once(f"Missing parameters: {missing}")
            if unexpected:
                print_once(f"Unexpected parameters: {unexpected}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model.to(device)

        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            backend="causal",
            device="cuda",
            batch_size=128,
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
    quant_args, eval_args = parse_args()

    # Create the output directory if it doesn't exist
    output_dir = Path(eval_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the model
    model = QuantizedLLM(
        model_path=eval_args.model_path,
        device=eval_args.device,
        rconfig=quant_args.get_rconfig() if quant_args.is_quant else None,
    )

    # Run evaluation
    print(f"Evaluating model on tasks: {eval_args.tasks}")
    results = evaluator.simple_evaluate(
        model=model,
        tasks=eval_args.tasks,  # type: ignore
        confirm_run_unsafe_code=True,
    )

    # Return if no results
    if results is None:
        return

    # Save results
    model_name = os.path.basename(eval_args.model_path)
    output_file = output_dir / f"{model_name}_{quant_args.label}_results.json"
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
