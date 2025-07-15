import json
from pathlib import Path

from exps.run_exp_llm import QuantizationArguments

MODELS = [
    "facebook/MobileLLM-125M",
    "facebook/MobileLLM-600M",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
]

QUANT_ARGS = [
    # QuantizationArguments(label="BF16-baseline"),
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
        weight_alg="lsq",
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
        weight_alg="minmax",
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
        weight_qtype="uniform",
        weight_qbits=4,
        weight_alg="minmax",
    ),
    QuantizationArguments(
        weight_qtype="uniform",
        weight_qbits=4,
        weight_alg="lsq",
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
        weight_alg="minmax",
        weight_block_size="channel",
    ),
    QuantizationArguments(
        weight_qtype="uniform",
        weight_qbits=4,
        weight_alg="lsq",
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

LABEL_NAMES = ["arc_e", "arc_c", "boolq", "piqa", "siqa", "hella", "obqa", "wino"]


def get_results(eval_results_file: Path, wiki_results_file: Path):
    acc_results = json.load(eval_results_file.open())
    accuracies = [acc_results["results"][label]["acc,none"] * 100 for label in LABELS]
    accuracies.append(sum(accuracies) / len(accuracies))

    wiki_results = json.load(wiki_results_file.open())
    perplexity = wiki_results["perplexity"]
    return (f"{num:.1f}" for num in accuracies + [perplexity])


def main():
    width = 8
    labels = LABEL_NAMES + ["avg", "wiki2"]
    print("".join(f"{item:>{width}}" for item in labels))
    for model in MODELS:
        for quant_arg in QUANT_ARGS:
            eval_result_file = (
                Path("./output/eval")
                / f"{model.split('/')[-1]}_{quant_arg.label}_results.json"
            )
            wiki_results_file = (
                Path("./output")
                / model.split("/")[-1]
                / quant_arg.label
                / "eval_results.json"
            )
            missing_files = []
            if not eval_result_file.exists():
                missing_files.append(eval_result_file)
            if not wiki_results_file.exists():
                missing_files.append(wiki_results_file)
            if missing_files:
                print(f"Missing files: {missing_files}")
                continue
            results = get_results(eval_result_file, wiki_results_file)
            print("".join(f"{item:>{width}}" for item in results))


if __name__ == "__main__":
    main()
