import json
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import transformers
from loguru import logger
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    default_data_collator,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.auto_factory import (
    _get_model_class,  # pyright: ignore[reportPrivateUsage]
)

from quant_mp.algs.template import ALGORITHMS, get_algorithm
from quant_mp.config import QuantConfig, QuantModuleConfig
from quant_mp.datatypes.template import DATA_FORMATS, get_data_format
from quant_mp.utils import patch_model


@dataclass
class QuantizationArguments:
    label: str = field(  # pyright: ignore[reportAssignmentType]
        default=None,
        metadata={
            "help": "Label name for the quantion. Defaults to {activation_qtype}-{activation_format}-{activation_alg}-{weight_qtype}-{weight_format}-{weight_alg}"
        },
    )  # type: ignore
    activation_dformat: Optional[str] = field(
        default=None,
        metadata={
            "choices": DATA_FORMATS.keys(),
            "help": "Data format for activations. Defaults to None (no activation quantization).",
        },
    )
    activation_alg: Optional[str] = field(
        default=None,
        metadata={
            "choices": ALGORITHMS.keys(),
            "help": "Quantization algorithm for activations.",
        },
    )
    activation_alg_kwargs: Optional[str] = field(
        default=None, metadata={"help": "JSON-parsable mapping for algorithm kwargs."}
    )
    activation_init_alg: Optional[str] = field(
        default=None,
        metadata={
            "choices": ALGORITHMS.keys(),
            "help": "Initialization algorithm for activations (defaults to quant algorithm when it supports fit_params).",
        },
    )
    activation_init_alg_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "JSON-parsable mapping for init algorithm kwargs."},
    )
    weight_dformat: Optional[str] = field(
        default=None,
        metadata={
            "choices": DATA_FORMATS.keys(),
            "help": "Data format for weights. Defaults to None (no weight quantization).",
        },
    )
    weight_block_size: Optional[int | str] = field(
        default=None, metadata={"help": "Block size in integer blocks or 'channel'."}
    )
    weight_alg: Optional[str] = field(
        default=None,
        metadata={
            "choices": ALGORITHMS.keys(),
            "help": "Quantization algorithm for activations.",
        },
    )
    weight_alg_kwargs: Optional[str] = field(
        default=None, metadata={"help": "JSON-parsable mapping for algorithm kwargs."}
    )
    weight_init_alg: Optional[str] = field(
        default=None,
        metadata={
            "choices": ALGORITHMS.keys(),
            "help": "Initialization algorithm for weights (defaults to quant algorithm when it supports fit_params).",
        },
    )
    weight_init_alg_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "JSON-parsable mapping for init algorithm kwargs."},
    )

    def __post_init__(self):
        if self.label is None:  # pyright: ignore[reportUnnecessaryComparison]
            if self.is_quant:
                self.label = f"W-{self.weight_dformat}-{self.weight_block_size}-{self.weight_alg}--A-{self.activation_dformat}-{self.activation_alg}"
            else:
                self.label = "Baseline"

    @property
    def activation_qconfig(self):
        if self.activation_dformat is None:
            return None
        assert self.activation_alg is not None, (
            "Alg is required if activation dformat is set."
        )
        qparam_data_format = get_data_format("fp32")

        algorithm = get_algorithm(
            self.activation_alg,
            algorithm_init_kwargs=json.loads(self.activation_alg_kwargs or "{}"),
        )
        init_algorithm = (
            get_algorithm(
                self.activation_init_alg,
                algorithm_init_kwargs=json.loads(
                    self.activation_init_alg_kwargs or "{}"
                ),
            )
            if self.activation_init_alg is not None
            else None
        )
        return QuantConfig(
            qval_data_format=get_data_format(self.activation_dformat),
            qparam_data_format=qparam_data_format,
            algorithm=algorithm,
            init_algorithm=init_algorithm,
        )

    @property
    def weight_qconfig(self):
        if self.weight_dformat is None:
            return None
        assert self.weight_alg is not None, "Alg is required if weight dformat is set."
        qparam_data_format = get_data_format("fp32")

        algorithm = get_algorithm(
            self.weight_alg,
            algorithm_init_kwargs=json.loads(self.weight_alg_kwargs or "{}"),
        )
        init_algorithm = (
            get_algorithm(
                self.weight_init_alg,
                algorithm_init_kwargs=json.loads(self.weight_init_alg_kwargs or "{}"),
            )
            if self.weight_init_alg is not None
            else None
        )
        return QuantConfig(
            qval_data_format=get_data_format(self.weight_dformat),
            qparam_data_format=qparam_data_format,
            algorithm=algorithm,
            init_algorithm=init_algorithm,
            qblock_size=self.weight_block_size,
        )

    @property
    def is_quant(self):
        return self.activation_dformat is not None or self.weight_dformat is not None

    def get_rconfig(self):
        if not self.is_quant:
            return None
        return QuantModuleConfig(
            activation=self.activation_qconfig,
            weight=self.weight_qconfig,
        )


@dataclass
class ModelArguments:
    model_name: str = field(
        default="facebook/MobileLLM-125M",
        metadata={"help": "Model name or path, loaded by AutoModelForCausalLM"},
    )
    tokenizer_name: str = field(
        default=None,
        metadata={
            "help": "Tokenizer name or path, loaded by AutoTokenizer. Defaults to model_name."
        },
    )  # pyright: ignore[reportAssignmentType]
    output_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to save the fine-tuned model. If not set, the model will not be saved."
        },
    )  # pyright: ignore[reportAssignmentType]
    use_cache: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Set model.config.use_cache. Set to False during training with gradient checkpointing to avoid incompatibility warnings."
        },
    )

    def __post_init__(self):
        self.tokenizer_name = self.tokenizer_name or self.model_name


@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max training samples in number of lines. Used for debugging on smaller training set"
        },
    )
    max_valid_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max validation samples in number of lines. Used for debugging on smaller validation set"
        },
    )
    train_ds_path: str = field(
        default="./data/train.jsonl", metadata={"help": "Path to training dataset"}
    )
    valid_ds_path: Optional[str] = field(
        default=None, metadata={"help": "Path to validation dataset"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: Optional[str] = field(
        default="adamw_torch"
    )  # TODO: Determine if this is needed.
    output_dir: str = field(default="/tmp/output/")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    qat: Optional[bool] = field(default=False)


def parse_args() -> Tuple[
    QuantizationArguments, ModelArguments, TrainingArguments, DataArguments
]:
    parser = transformers.HfArgumentParser(
        (QuantizationArguments, ModelArguments, TrainingArguments, DataArguments)  # pyright: ignore[reportArgumentType]
    )
    quant_args, model_args, training_args, data_args = (
        parser.parse_args_into_dataclasses()
    )

    training_args.output_dir = (
        f"./output/{model_args.model_name.split('/')[-1]}/{quant_args.label}"
    )
    # Ensure we always save in safetensors format for HF compatibility
    try:
        # Not all transformers versions expose this in TrainingArguments, so guard it
        setattr(training_args, "save_safetensors", True)
    except Exception:
        pass
    print_once(f"Quant Args: {quant_args}")
    print_once(f"Model Args: {model_args}")
    print_once(f"Data Args: {data_args}")
    print_once(f"Training Args:\n{training_args}")
    return quant_args, model_args, training_args, data_args


class CustomJsonDataset(torch.utils.data.IterableDataset):  # pyright: ignore[reportMissingTypeArgument]
    def __init__(self, dataset, tokenizer, block_size=1024):
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


def read_jsonl_dataset(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    return data


def print_once(*args, **kwargs):
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(*args, **kwargs)


def load_quant_model(quant_model_path: str | Path, rconfig: QuantModuleConfig):
    config = AutoConfig.from_pretrained(quant_model_path, trust_remote_code=True)
    if hasattr(config, "auto_map"):
        model_cls = get_class_from_dynamic_module(
            config.auto_map["AutoModelForCausalLM"], quant_model_path
        )
    elif type(config) in AutoModelForCausalLM._model_mapping.keys():
        model_cls = _get_model_class(config, AutoModelForCausalLM._model_mapping)
    else:
        raise RuntimeError(f"Could not find model class for {quant_model_path}")
    model = model_cls(config)
    patch_model(model, rconfig)
    state_dict = {}
    for state_dict_path in Path(quant_model_path).glob("*.safetensors"):
        state_dict.update(load_file(state_dict_path))
    model.load_state_dict(state_dict, strict=False)
    return model


def main(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    quant_args: QuantizationArguments,
    data_args: DataArguments,
):
    output_path = (
        model_args.output_model_path
        if model_args.output_model_path is not None
        else f"{training_args.output_dir}/best-model"
    )
    quant_config = quant_args.get_rconfig()

    if os.path.exists(output_path):
        print_once(f"Model found at {output_path}")
        if quant_config is not None:
            model = load_quant_model(
                output_path,
                quant_config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                output_path, trust_remote_code=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        if quant_config is not None:
            print_once(f"Patching model with quant config: {quant_config}")
            patch_model(model, quant_config)

    # Respect CLI flag if provided, otherwise disable cache when using gradient checkpointing
    if model_args.use_cache is not None:
        model.config.use_cache = bool(model_args.use_cache)
    elif getattr(training_args, "gradient_checkpointing", False):
        # HF warns and auto-disables if left True; do it explicitly to avoid warning spam
        model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, use_fast=False)

    train_data = read_jsonl_dataset(data_args.train_ds_path)[
        : data_args.max_train_samples
    ]
    train_ds = CustomJsonDataset(
        train_data, tokenizer, block_size=training_args.model_max_length
    )

    valid_ds = None
    if data_args.valid_ds_path is not None:
        valid_data = read_jsonl_dataset(data_args.valid_ds_path)[
            : data_args.max_valid_samples
        ]
        valid_ds = CustomJsonDataset(
            valid_data, tokenizer, block_size=min(training_args.model_max_length, 1024)
        )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=valid_ds if training_args.do_eval else None,
        data_collator=default_data_collator,
    )

    torch.cuda.empty_cache()
    if training_args.do_train and not os.path.exists(output_path):
        trainer.train()
        trainer.save_state()
        trainer.save_model(output_path)
        try:
            tokenizer.save_pretrained(output_path)
        except Exception:
            pass
        torch.cuda.empty_cache()

    if training_args.do_eval and valid_ds is not None:
        metrics = trainer.evaluate()
        max_eval_samples = len(valid_ds)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_ds))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        eval_results_file = Path(training_args.output_dir) / "eval_results.json"
        if not eval_results_file.exists():
            trainer.save_metrics("eval", metrics)

        # Save to eval dir
        if eval_results_file.exists() and os.environ.get("LOCAL_RANK", "0") == "0":
            dest_dir = (
                Path("./output/eval")
                / model_args.model_name.split("/")[-1]
                / quant_args.label
            )
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / "perplexity_results.json"
            shutil.copyfile(eval_results_file, dest_file)


if __name__ == "__main__":
    quant_args, model_args, training_args, data_args = parse_args()
    logger.remove()
    logger.add(f"{training_args.output_dir}/run.log", level="INFO")
    try:
        main(model_args, training_args, quant_args, data_args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
