import json
import math
import os
import random
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional  # pyright: ignore[reportDeprecated]

import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from safetensors.torch import load_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.auto_factory import (
    _get_model_class,  # pyright: ignore[reportPrivateUsage]
)

from quant_mp.algs.template import ALGORITHMS, get_algorithm
from quant_mp.config import QuantConfig, QuantModuleConfig
from quant_mp.datatypes.template import DATA_FORMATS, get_data_format
from quant_mp.QModules import QConv2d, QLinear
from quant_mp.utils import patch_model


def add_quantization_args(parser: ArgumentParser):
    parser_group = parser.add_argument_group("Quantization Arguments")
    parser_group.add_argument(
        "--label",
        default=None,
        help="Label name for the quantion. Automatically sets if None.",
    )
    parser_group.add_argument(
        "--activation-dformat",
        default=None,
        choices=DATA_FORMATS.keys(),
        help="Data format for activations. Defaults to None (no activation quantization).",
    )
    parser_group.add_argument(
        "--activation-alg",
        default=None,
        choices=ALGORITHMS.keys(),
        help="Quantization algorithm for activations",
    )
    parser_group.add_argument(
        "--activation-alg-kwargs",
        default=None,
        help="JSON-parsable mapping for algorithm kwargs.",
    )
    parser_group.add_argument(
        "--weight-dformat",
        default=None,
        choices=DATA_FORMATS.keys(),
        help="Data format for weights. Defaults to None (no weight quantization).",
    )
    parser_group.add_argument(
        "--weight-block-size",
        default=None,
        help="Block size in integer blocks or 'channel'.",
    )
    parser_group.add_argument(
        "--weight-alg",
        default=None,
        choices=ALGORITHMS.keys(),
        help="Quantization algorithm for activations.",
    )
    parser_group.add_argument(
        "--weight-alg-kwargs",
        default=None,
        help="JSON-parsable mapping for algorithm kwargs.",
    )


def add_model_args(parser: ArgumentParser):
    parser_group = parser.add_argument_group("Model Arguments")
    parser_group.add_argument(
        "--model-name",
        default="facebook/MobileLLM-125M",
        help="Model name or path, loaded by AutoModelForCausalLM",
    )
    parser_group.add_argument(
        "--tokenizer-name",
        default=None,
        help="Tokenizer name or path, loaded by AutoTokenizer. Defaults to model_name.",
    )
    parser_group.add_argument(
        "--shard-layers",
        default=False,
        action="store_true",
        help="Shards the model with FSDP2 into each layer. May not work for some models.",
    )


def add_data_args(parser: ArgumentParser):
    parser_group = parser.add_argument_group("Data Arguments")
    parser_group.add_argument(
        "--max-train-samples",
        default=None,
        type=int,
        help="Max training samples in number of lines. Used for debugging on smaller training set",
    )
    parser_group.add_argument(
        "--max-valid-samples",
        default=None,
        type=int,
        help="Max validation samples in number of lines. Used for debugging on smaller validation set",
    )
    parser_group.add_argument(
        "--max-sequence-length",
        default=2048,
        type=int,
        help="Maximum sequence length for model",
    )
    parser_group.add_argument(
        "--train-ds-path",
        default=Path("./data/train.jsonl"),
        type=Path,
        help="Path to training dataset",
    )
    parser_group.add_argument(
        "--valid-ds-path",
        default=Path("./data/validation.jsonl"),
        type=Path,
        help="Path to validation dataset",
    )


def add_training_args(parser: ArgumentParser):
    parser_group = parser.add_argument_group("Training Arguments")
    parser_group.add_argument(
        "--output-dir",
        default=Path("./output/"),
        type=Path,
        help="Base output directory for model checkpoints and evaluation results.",
    )
    parser_group.add_argument(
        "--save-model",
        default=False,
        action="store_true",
        help="Save model to output directory",
    )
    parser_group.add_argument(
        "--per-device-train-batch-size",
        default=1,
        type=int,
        help="Training batch size per device (data parallel)",
    )
    parser_group.add_argument(
        "--per-device-eval-batch-size",
        default=1,
        type=int,
        help="Validation batch size per device.",
    )
    parser_group.add_argument(
        "--learning-rate", default=2e-5, type=float, help="Optimizer learning rate"
    )
    parser_group.add_argument(
        "--weight-decay", default=0.01, type=float, help="Optimizer weight decay"
    )
    parser_group.add_argument(
        "--num-train-epochs",
        default=1,
        type=int,
        help="Number of training epochs over entire dataset",
    )
    parser_group.add_argument(
        "--do-train",
        default=False,
        action="store_true",
        help="Enable training of the model",
    )
    parser_group.add_argument(
        "--do-eval",
        default=False,
        action="store_true",
        help="Enable evaluation of the model",
    )


def add_profiling_args(parser: ArgumentParser):
    parser_group = parser.add_argument_group("Profiling Arguments")
    parser_group.add_argument(
        "--enable-cuda-memory-dump",
        action="store_true",
        default=False,
        help="Enable CUDA memory dump pkl",
    )
    parser_group.add_argument(
        "--cuda-memory-dump-snapshot-dir",
        default=Path("./memory_dumps"),
        type=Path,
        help="Memory dump directory",
    )
    parser_group.add_argument(
        "--cuda-memory-dump-max-entries",
        default=100_000,
        type=int,
        help="Max entries in cuda memory recording history",
    )


def set_implicit_args(args):
    args.is_quant = (
        args.activation_dformat is not None or args.weight_dformat is not None
    )
    if args.label is None:
        if args.is_quant:
            args.label = f"W-{args.weight_dformat}-{args.weight_block_size}-{args.weight_alg}--A-{args.activation_dformat}-{args.activation_alg}"
        else:
            args.label = "Baseline"

    # Convert numeric block size to integer
    if args.weight_block_size is not None:
        assert (
            args.weight_block_size.isnumeric() or args.weight_block_size == "channel"
        ), f"Invalid weight block size: {args.weight_block_size}"
        if args.weight_block_size.isnumeric():
            args.weight_block_size = int(args.weight_block_size)

    args.quant_module_config = None
    if args.is_quant:
        fp32 = get_data_format("fp32")
        activation_qconfig = None
        if args.activation_dformat is not None:
            assert args.activation_alg is not None, (
                "Activation algorithm is required if activation dformat is set"
            )
            activation_alg = get_algorithm(
                args.activation_alg,
                algorithm_init_kwargs=json.loads(args.activation_alg_kwargs or "{}"),
            )
            activation_qconfig = QuantConfig(
                qval_data_format=get_data_format(args.activation_dformat),
                qparam_data_format=fp32,
                algorithm=activation_alg,
            )
        weight_qconfig = None
        if args.weight_dformat is not None:
            assert args.weight_alg is not None, (
                "Weight algorithm is required if weight dformat is set"
            )
            weight_alg = get_algorithm(
                args.weight_alg,
                algorithm_init_kwargs=json.loads(args.weight_alg_kwargs or "{}"),
            )
            weight_qconfig = QuantConfig(
                qval_data_format=get_data_format(args.weight_dformat),
                qparam_data_format=fp32,
                algorithm=weight_alg,
                qblock_size=args.weight_block_size,
            )
        args.quant_module_config = QuantModuleConfig(
            activation=activation_qconfig, weight=weight_qconfig
        )

    args.tokenizer_name = args.tokenizer_name or args.model_name


def set_seed(seed: int):
    """
    Set random seed for reproducibility across:
    - Python random
    - NumPy
    - PyTorch (CPU & CUDA)
    - CUDNN deterministic behavior

    Parameters
    ----------
    seed : int
        The random seed to use.
    """
    # Python and NumPy seeds
    random.seed(seed)

    # PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For additional reproducibility in dataloader workers
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_jsonl_dataset(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def load_quant_model(quant_model_path: str | Path, rconfig: QuantModuleConfig):
    config = AutoConfig.from_pretrained(
        quant_model_path, trust_remote_code=True, dtype=torch.bfloat16
    )
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


def print_rank0(*args, **kwargs):
    rank_env = os.environ.get("RANK")
    rank = int(rank_env) if rank_env is not None else 0
    if rank == 0:
        print("[rank0 print]:", *args, **kwargs)


class SimpleDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.input_ids[i]),
            "labels": torch.tensor(self.labels[i]),
        }


def get_dataloader(
    ds_path,
    tokenizer,
    max_samples=None,
    max_sequence_length=2048,
    per_device_batch_size=1,
    num_workers=8,
):
    raw_data = read_jsonl_dataset(ds_path)[:max_samples]
    tokenized_data = [tokenizer(example["text"]) for example in raw_data]

    input_ids = []
    for example in tokenized_data:
        input_ids.extend(example["input_ids"])
    total_length = len(input_ids)
    if total_length >= max_sequence_length:
        total_length = (total_length // max_sequence_length) * max_sequence_length
    input_ids_chunked = [
        input_ids[i : i + max_sequence_length]
        for i in range(0, total_length, max_sequence_length)
    ]
    ds = SimpleDataset(input_ids_chunked, input_ids_chunked.copy())
    sampler = DistributedSampler(ds, shuffle=True)
    return DataLoader(
        ds,
        sampler=sampler,
        batch_size=per_device_batch_size,
        num_workers=num_workers,
    )


def collect_quant_params(model):
    ignored = set()
    for m in model.modules():
        if isinstance(m, (QLinear, QConv2d)):
            for name in (
                "weight_scale",
                "weight_shift",
                "activation_scale",
                "activation_shift",
            ):
                if hasattr(m, name):
                    ignored.add(getattr(m, name))
    return ignored


def shard_model(model: nn.Module, mesh, shard_layers: bool = False):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float16,
        reduce_dtype=torch.float32,
    )
    ignored = collect_quant_params(model)
    fully_shard_kwargs: dict[str, Any] = dict(
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=True,
        ignored_params=ignored,
    )
    if shard_layers:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            for layer in model.model.layers:  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
                for module in layer.children():
                    fully_shard(module, **fully_shard_kwargs)
                fully_shard(layer, **fully_shard_kwargs)
        if hasattr(model, "model"):
            for child in model.model.children():  # pyright: ignore[reportAttributeAccessIssue]
                if not isinstance(child, nn.ModuleList):
                    fully_shard(child, **fully_shard_kwargs)
        for module in model.children():
            fully_shard(module, **fully_shard_kwargs)
    fully_shard(model, **fully_shard_kwargs)


@dataclass
class ValidationOutput:
    time_per_batch: list[float]
    total_time: float
    loss: float
    perplexity: float


@dataclass
class TrainingOutput:
    time_per_batch: list[list[float]] = field(default_factory=list)
    time_per_epoch: list[float] = field(default_factory=list)
    total_time: float = 0.0
    loss_per_batch: list[list[float]] = field(default_factory=list)
    loss_per_epoch: list[float] = field(default_factory=list)
    perplexity_per_batch: list[list[float]] = field(default_factory=list)
    perplexity_per_epoch: list[float] = field(default_factory=list)
    validation_outputs: Optional[list[ValidationOutput]] = None  # pyright: ignore[reportDeprecated]

    def log_epoch(
        self,
        batch_timings: list[float],
        batch_losses: list[float],
        batch_perplexities: list[float],
        epoch_time: float,
        epoch_loss: float,
        epoch_perplexity: float,
        validation_output: Optional[ValidationOutput] = None,  # pyright: ignore[reportDeprecated]
    ):
        self.time_per_batch.append(batch_timings)
        self.time_per_epoch.append(epoch_time)
        self.loss_per_batch.append(batch_losses)
        self.loss_per_epoch.append(epoch_loss)
        self.perplexity_per_batch.append(batch_perplexities)
        self.perplexity_per_epoch.append(epoch_perplexity)
        if validation_output is not None:
            if self.validation_outputs is None:
                self.validation_outputs = []
            self.validation_outputs.append(validation_output)

    def print_summary(self):
        if dist.get_rank() == 0:
            print("======== Training Summary ========")
            print(f"Num epochs: {len(self.time_per_epoch)}")
            print(f"Num batch / epoch: {len(self.time_per_batch[0])}")
            print(f"Time (s): {self.time_per_epoch}")
            print(f"Total time (s): {self.total_time}")
            print(f"Train Loss: {self.loss_per_epoch}")
            print(f"Train Perplexity: {self.perplexity_per_epoch}")
            if self.validation_outputs is not None:
                print(f"Validation Loss: {[o.loss for o in self.validation_outputs]}")
                print(
                    f"Validation Perplexity: {[o.perplexity for o in self.validation_outputs]}"
                )

            if self.validation_outputs is not None:
                _, best_epoch = min(
                    zip(
                        [o.perplexity for o in self.validation_outputs],
                        range(len(self.loss_per_epoch)),
                    ),
                    key=lambda i: i[0],
                )
                print("")
                print(f"-------- Best Epoch ({best_epoch}) stats --------")
                print(f"Train Loss: {self.loss_per_epoch[best_epoch]}")
                print(f"Train Perplexity: {self.perplexity_per_epoch[best_epoch]}")
                print(f"Validation Loss: {self.validation_outputs[best_epoch].loss}")
                print(
                    f"Validation Perplexity: {self.validation_outputs[best_epoch].perplexity}"
                )


@dataclass
class Metadata:
    epoch: int = 0
    loss: float = float("inf")
    val_outputs: Optional[list[ValidationOutput]] = None  # pyright: ignore[reportDeprecated]


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metadata: Metadata,
    train_loader: DataLoader[Any],
    valid_loader: Optional[DataLoader[Any]] = None,  # pyright: ignore[reportDeprecated]
    model_output_dir: Optional[Path] = None,  # pyright: ignore[reportDeprecated]
    result_output_dir: Optional[Path] = None,  # pyright: ignore[reportDeprecated]
    save_best_model: bool = False,
    num_epochs: int = 1,
    model_name: str = "",
    qconfig: Optional[QuantModuleConfig] = None,  # pyright: ignore[reportDeprecated]
    device: torch.types.Device = None,
) -> TrainingOutput:
    t0 = time.time()
    output = TrainingOutput()
    rank = dist.get_rank()
    best_loss = metadata.loss
    for epoch in range(metadata.epoch, num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        dist.barrier()
        t1 = time.time()
        local_epoch_loss = torch.zeros(2).to(device)
        batch_timings = []
        batch_losses = []
        batch_perplexities = []
        if rank == 0:
            train_loader = tqdm.tqdm(train_loader, desc=f"[rank0] Epoch {epoch}")  # pyright: ignore[reportAssignmentType]
        for i, batch in enumerate(train_loader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            t2 = time.time()
            optimizer.zero_grad(set_to_none=True)
            model_output = model(**batch)
            loss = model_output["loss"]
            loss.backward()
            optimizer.step()
            t3 = time.time()

            # Stats per batch
            batch_size = batch["input_ids"].size(0)
            local_epoch_loss[0] += loss.item() * batch_size
            local_epoch_loss[1] += batch_size

            local_batch_loss = torch.tensor(
                [loss.item() * batch_size, batch_size], dtype=torch.float32
            ).to(device)
            dist.all_reduce(local_batch_loss, op=dist.ReduceOp.SUM)
            batch_loss = (local_batch_loss[0] / local_batch_loss[1]).item()
            batch_perplexity = math.exp(batch_loss)

            batch_losses.append(batch_loss)
            batch_perplexities.append(batch_perplexity)
            batch_timings.append(t3 - t2)

            if rank == 0:
                tqdm.tqdm.write(
                    f"Epoch {epoch} batch {i}: loss={batch_loss:.4f} perpexity={batch_perplexity:.4f}"
                )
            if math.isnan(batch_loss) or math.isnan(batch_perplexity):
                print_rank0(f"ERROR: NAN detected at {epoch=} batch={i}. Dumping info.")
                print_rank0(f"{batch=}")
                print_rank0(f"{model_output=}")
                raise RuntimeError(
                    "Nan detected in batch loss and/or perplexity. Aborting training."
                )

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        t4 = time.time()

        dist.all_reduce(local_epoch_loss, op=dist.ReduceOp.SUM)
        epoch_loss = (local_epoch_loss[0] / local_epoch_loss[1]).item()
        epoch_perplexity = math.exp(epoch_loss)
        print_rank0(
            f"Epoch {epoch} summary: loss={epoch_loss:.4f} perplexity={epoch_perplexity:.4f}"
        )

        if math.isnan(epoch_loss) or math.isnan(epoch_perplexity):
            raise RuntimeError(
                "Nan detected in epoch loss and/or perplexity. Aborting training."
            )

        val_output = None
        if valid_loader is not None:
            val_output = validate(model, valid_loader, device)
            print_rank0(
                f"Epoch {epoch} validation: loss={val_output.loss:.4f}, perplexity={val_output.perplexity:.4f}"
            )

        # NOTE: This may be off by one
        metadata.epoch = epoch
        metadata.loss = epoch_loss
        if epoch_loss < best_loss:
            print_rank0(
                f"New best epoch loss: {epoch_loss:.4f} (delta: {best_loss - epoch_loss:.4f})"
            )
            best_loss = epoch_loss
            if save_best_model:
                assert model_output_dir is not None
                save_path = model_output_dir / "best-model"
                save_path.mkdir(exist_ok=True, parents=True)
                if rank == 0:
                    save_metadata(metadata, save_path)
                save_model(model, save_path, model_name, qconfig)
            if rank == 0:
                if result_output_dir is not None:
                    save_path = result_output_dir / "best-model"
                    save_path.mkdir(exist_ok=True, parents=True)
                    save_eval_perplexity(epoch, epoch_loss, epoch_perplexity, save_path)
            dist.barrier()

        output.log_epoch(
            batch_timings,
            batch_losses,
            batch_perplexities,
            t4 - t1,
            epoch_loss,
            epoch_perplexity,
            val_output,
        )

    output.total_time = time.time() - t0
    return output


def validate(
    model: nn.Module, valid_loader: DataLoader[Any], device: torch.types.Device = None
) -> ValidationOutput:
    model.eval()
    local_loss = torch.zeros(2).to(device)
    t0 = time.time()
    batch_timings = []
    with torch.no_grad():
        if dist.get_rank() == 0:
            valid_loader = tqdm.tqdm(valid_loader, desc="[rank0] Validation")  # pyright: ignore[reportAssignmentType]
        for batch in valid_loader:
            t1 = time.time()
            for k, v in batch.items():
                batch[k] = v.to(device)
            model_output = model(**batch)
            loss = model_output["loss"].item()
            t2 = time.time()

            # Stats per batch
            batch_size = batch["input_ids"].size(0)
            local_loss[0] += loss * batch_size
            local_loss[1] += batch_size
            batch_timings.append(t2 - t1)

    t3 = time.time()
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    val_loss = (local_loss[0] / local_loss[1]).item()
    val_perplexity = math.exp(val_loss)
    return ValidationOutput(
        time_per_batch=batch_timings,
        total_time=t3 - t0,
        loss=val_loss,
        perplexity=val_perplexity,
    )


def save_model(
    model: nn.Module,
    model_save_dir: Path,
    model_name: str,
    qconfig: QuantModuleConfig | None,
):
    print_rank0(f"Saving model to {model_save_dir}")
    state_dict = model.state_dict()
    rank = dist.get_rank()

    if rank == 0:
        for k, v in state_dict.items():
            if isinstance(v, DTensor):
                state_dict[k] = v.full_tensor().cpu()
            else:
                state_dict[k] = v.cpu()
    else:
        for v in state_dict.values():
            if isinstance(v, DTensor):
                _ = v.full_tensor()
    dist.barrier()

    if rank == 0:
        fresh_model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        if qconfig is not None:
            patch_model(fresh_model, qconfig)
        missing, unexpected = fresh_model.load_state_dict(state_dict, strict=False)
        print(f"{missing=}")
        print(f"{unexpected=}")
        fresh_model.save_pretrained(model_save_dir, save_serialization=True)
    dist.barrier()


def save_metadata(metadata: Metadata, save_dir: Path):
    save_file = save_dir / "training_metadata.json"
    print_rank0(f"Saving metadata to {save_file}")
    json.dump(asdict(metadata), save_file.open("w"))


def save_eval_perplexity(epoch: int, loss: float, perplexity: float, save_dir: Path):
    save_file = save_dir / "perplexity_results.json"
    print_rank0(f"Saving validation results to {save_file}")
    results = {"epoch": epoch, "loss": loss, "perplexity": perplexity}
    json.dump(results, save_file.open("w"))


def main(args):
    print_rank0(f"Training LLM FSDP launched with {args=}")

    local_rank = int(os.environ["LOCAL_RANK"])
    distributed_backend = "nccl"
    device_type = "cuda"
    torch.cuda.set_device(local_rank)
    dist.init_process_group(distributed_backend, device_id=local_rank)
    mesh = init_device_mesh(
        device_type, mesh_shape=(dist.get_world_size(),), mesh_dim_names=("fsdp",)
    )
    device = torch.device(f"cuda:{dist.get_rank()}")

    model_output_dir: Path = (
        args.output_dir / "models" / args.model_name.split("/")[-1] / args.label
    )
    result_output_dir: Path = (
        args.output_dir / "eval" / args.model_name.split("/")[-1] / args.label
    )
    print_rank0(
        f"Initialized {distributed_backend} on {device_type} with {dist.get_world_size()=} and {mesh=}"
    )

    # Load metadata
    metadata_file = model_output_dir / "best-model" / "training_metadata.json"
    if metadata_file.exists():
        metadata = Metadata(**json.loads(metadata_file.read_text()))
    else:
        metadata = Metadata()

    quant_config: Optional[QuantModuleConfig] = args.quant_module_config  # pyright: ignore[reportDeprecated]
    if list((model_output_dir / "best-model").glob("*.safetensors")):
        print_rank0(f"Loading model from {model_output_dir / 'best-model'}")
        if quant_config is not None:
            model = load_quant_model(
                model_output_dir / "best-model",
                quant_config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_output_dir / "best-model",
                trust_remote_code=True,
            )
        print_rank0(
            "Disabling training due to loading best-model. Will evaluate if necessary."
        )
        args.do_train = False
    else:
        print_rank0(f"Loading model from {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=True
        )
        if quant_config is not None:
            patch_model(model, quant_config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)

    dist.barrier()

    print_rank0("Enabling gradient checkpointing")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    print_rank0("Sharding with FSDP2")
    shard_model(model, mesh["fsdp"], args.shard_layers)
    print_rank0(f"Resulting {model=}")

    print_rank0(
        f"Initializing optimizer with {args.learning_rate=}, {args.weight_decay=}"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    train_loader = None
    if args.do_train:
        print_rank0(f"Loading training dataset from {args.train_ds_path}")
        train_loader = get_dataloader(
            args.train_ds_path,
            tokenizer,
            args.max_train_samples,
            args.max_sequence_length,
            args.per_device_train_batch_size,
        )
        print_rank0(f"Loaded {train_loader=}")

    valid_loader = None
    if args.do_eval:
        print_rank0(f"Loading validation dataset from {args.valid_ds_path}")
        valid_loader = get_dataloader(
            args.valid_ds_path,
            tokenizer,
            args.max_valid_samples,
            args.max_sequence_length,
            args.per_device_eval_batch_size,
        )

    if args.do_train:
        assert train_loader is not None

        try:
            train_output = train(
                model,
                optimizer,
                scheduler,
                metadata,
                train_loader,
                valid_loader=valid_loader,
                model_output_dir=model_output_dir,
                result_output_dir=result_output_dir,
                save_best_model=args.save_model,
                num_epochs=args.num_train_epochs,
                model_name=args.model_name,
                qconfig=quant_config,
                device=device,
            )
            train_output.print_summary()
        except RuntimeError as e:
            print_rank0(f"Error occured during training: {e}")
            print_rank0("Soft aborting training.")
            return

    # For eval-only workflows
    if args.do_eval and not args.do_train:
        assert valid_loader is not None, "Must have validation loader for evaluation"
        val_output = validate(model, valid_loader, device=device)
        print_rank0(
            f"Validation: loss={val_output.loss:.4f}, perplexity={val_output.perplexity:.4f}"
        )


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    add_quantization_args(parser)
    add_model_args(parser)
    add_data_args(parser)
    add_training_args(parser)
    add_profiling_args(parser)
    args = parser.parse_args()
    set_implicit_args(args)
    set_seed(0)
    try:
        if args.enable_cuda_memory_dump:
            torch.cuda.memory._record_memory_history(
                max_entries=args.cuda_memory_dump_max_entries
            )
        main(args)
    finally:
        rank_env = os.environ.get("RANK")
        rank = int(rank_env) if rank_env is not None else 0
        if args.enable_cuda_memory_dump:
            args.cuda_memory_dump_snapshot_dir.mkdir(exist_ok=True)
            torch.cuda.memory._dump_snapshot(
                args.cuda_memory_dump_snapshot_dir / f"rank_{rank}_{time.time()}.pkl"
            )
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
