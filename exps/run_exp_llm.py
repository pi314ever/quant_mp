import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    default_data_collator,
)

from quant_mp.config import qconfig, rconfig
from quant_mp.utils import patch_model


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    qat: Optional[bool] = field(default=False)


class CustomJsonDataset(torch.utils.data.IterableDataset):
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


def read_jsonl_dataset(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    return data


def main(quant_config: rconfig):
    # TODO: Replace with cli args
    model_name = "facebook/opt-350m"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    patch_model(model, quant_config)
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # TODO: Replace with cli args
    training_args = TrainingArguments(
        bf16=True,
        do_eval=True,
        do_train=True,
        fp16=False,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        learning_rate=2e-5,
        logging_dir="./logs",
        logging_steps=1,
        log_on_each_node=False,
        lr_scheduler_type="cosine",
        model_max_length=2048,
        num_train_epochs=1,
        output_dir="./output-" + quant_config.label,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=2,
        qat=True,
        save_steps=2000,
        save_strategy="steps",
        save_total_limit=1,
        tf32=False,
        warmup_ratio=0.0,
        weight_decay=0.0,
    )

    # TODO: Replace with cli args
    train_ds_path = "./train.jsonl"
    valid_ds_path = "./valid.jsonl"

    train_data = read_jsonl_dataset(train_ds_path)
    valid_data = read_jsonl_dataset(valid_ds_path)

    train_ds = CustomJsonDataset(
        train_data, tokenizer, block_size=training_args.model_max_length
    )
    valid_ds = CustomJsonDataset(
        valid_data, tokenizer, block_size=min(training_args.model_max_length, 1024)
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=valid_ds if training_args.do_eval else None,
        data_collator=default_data_collator,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_state()
        # TODO: Replace with cli args
        output_path = f"./output-{quant_config.label}/best-model"
        trainer.save_model(output_path)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        max_eval_samples = len(valid_ds)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_ds))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    qtype = "float"
    format = "e2m1"

    try:
        quant_config = rconfig(
            label="FP32",
            activation=qconfig(qtype=None),
            weight=qconfig(qtype=None),
            grad=qconfig(qtype=None),
        )
        main(quant_config)

        quant_config = rconfig(
            label="FP4-minmax",
            activation=qconfig(qtype=qtype, alg="minmax", format=format),
            weight=qconfig(qtype=qtype, alg="minmax", format=format),
            grad=qconfig(qtype=None),
        )
        main(quant_config)

        quant_config = rconfig(
            label="FP4-analytic",
            activation=qconfig(qtype=qtype, alg="iterative", format=format),
            weight=qconfig(qtype=qtype, alg="normal", format=format),
            grad=qconfig(qtype=None),
        )
        main(quant_config)

        quant_config = rconfig(
            label="FP4-iterative",
            activation=qconfig(qtype=qtype, alg="iterative", format=format),
            weight=qconfig(qtype=qtype, alg="iterative", format=format),
            grad=qconfig(qtype=None),
        )
        main(quant_config)
    finally:
        dist.destroy_process_group()
