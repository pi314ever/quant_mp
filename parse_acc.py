import json
from pathlib import Path
from exps.run_exp_llm import QuantizationArguments


MODELS = [
    "facebook/MobileLLM-125M",
    "facebook/MobileLLM-600M",
    "meta-llama/Llama-3.2-1B",
]
QUANT_ARGS = [
    QuantizationArguments(label="BF16-baseline"),
    QuantizationArguments(
        weight_qtype="float",
        weight_qbits=4,
        weight_format="e2m1",
        weight_alg="minmax",
        weight_block_size="channel",
    ),
    QuantizationArguments(
        weight_qtype="float",
        weight_qbits=4,
        weight_format="e2m1",
        weight_alg="iterative",
        weight_block_size="channel",
    ),
    QuantizationArguments(
        weight_qtype="float",
        weight_qbits=4,
        weight_format="e2m1",
        weight_alg="normal",
        weight_block_size="channel",
    ),
    QuantizationArguments(
        weight_qtype="float",
        weight_qbits=4,
        weight_format="e2m1",
        weight_alg="lsq",
        weight_block_size="channel",
    ),
    QuantizationArguments(
        weight_qtype="uniform",
        weight_qbits=4,
        weight_alg="minmax",
        weight_block_size="channel",
    ),
    QuantizationArguments(
        weight_qtype="uniform",
        weight_qbits=4,
        weight_alg="iterative",
        weight_block_size="channel",
    ),
    QuantizationArguments(
        weight_qtype="uniform",
        weight_qbits=4,
        weight_alg="normal",
        weight_block_size="channel",
    ),
    QuantizationArguments(
        weight_qtype="uniform",
        weight_qbits=4,
        weight_alg="lsq",
        weight_block_size="channel",
    ),
    QuantizationArguments(
        weight_qtype="float",
        weight_qbits=4,
        weight_format="e2m1",
        weight_alg="minmax",
    ),
    QuantizationArguments(
        weight_qtype="float",
        weight_qbits=4,
        weight_format="e2m1",
        weight_alg="iterative",
    ),
    QuantizationArguments(
        weight_qtype="float",
        weight_qbits=4,
        weight_format="e2m1",
        weight_alg="normal",
    ),
    QuantizationArguments(
        weight_qtype="float",
        weight_qbits=4,
        weight_format="e2m1",
        weight_alg="lsq",
    ),
    QuantizationArguments(
        weight_qtype="uniform",
        weight_qbits=4,
        weight_alg="minmax",
    ),
    QuantizationArguments(
        weight_qtype="uniform",
        weight_qbits=4,
        weight_alg="iterative",
    ),
    QuantizationArguments(
        weight_qtype="uniform",
        weight_qbits=4,
        weight_alg="normal",
    ),
    QuantizationArguments(
        weight_qtype="uniform",
        weight_qbits=4,
        weight_alg="lsq",
    ),
]

LABELS = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "piqa",
    "social_iqa",
    "hellaswag",
    "openbookqa",
    "winogrande",
]


def print_parsed_results(file: Path):
    results = json.load(file.open())
    accuracies = "\t".join(
        str(results["results"][label]["acc,none"]) for label in LABELS
    )
    print(accuracies)


def main():
    print("\t".join(LABELS))
    for model in MODELS:
        for quant_arg in QUANT_ARGS:
            eval_result_file = (
                Path("./output/eval")
                / f"{model.split('/')[-1]}_{quant_arg.label}_results.json"
            )
            if eval_result_file.exists():
                print_parsed_results(eval_result_file)
            else:
                print(f"File does not exist at {eval_result_file}")


if __name__ == "__main__":
    main()
