#!/usr/bin/env bash
set -euo pipefail

NPROC="${NPROC:-8}"
CMD="torchrun --nproc_per_node $NPROC ./exps/train_llm_fsdp.py"
DRY_RUN="${DRY_RUN:-0}"
MODEL_FILTER="${MODEL_FILTER:-}"
MODEL_CONFIG_FILE="./configs/model_configs.json"
QUANT_CONFIG_FILE="./configs/quant_configs.json"

# ---- JQ program: merge common + model ---------------------------------------
JQ_PROG='
  . as $root
  | ($root.models[]) as $m
  | $qc[0].configs[] as $q
  | {
      model: $m.name,
      args: ($root.common.args + $m.args + $q.args),
      env:  ($root.common.env + $m.env + $q.env)
    }
'

# ---- Phase 1: Parse all runs into a variable --------------------------------
RUNS=$(jq -c --slurpfile qc "$QUANT_CONFIG_FILE" "$JQ_PROG" "$MODEL_CONFIG_FILE")

# Optional filter (exact match)
if [[ -n "$MODEL_FILTER" ]]; then
	RUNS=$(jq -c --arg mf "$MODEL_FILTER" 'select(.model==$mf)' <<<"$RUNS")
fi

# ---- Phase 2: Execute --------------------------------------------------------
run_with_env() {
	local -n _env="$1"
	shift
	local model="$1"
	shift
	local -a args=("$@")

	if ((DRY_RUN)) && [[ ${#_env[@]} -gt 0 ]]; then
		printf 'DRY env: %s\n' "${_env[*]}"
	fi
	if ((DRY_RUN)); then
		printf 'DRY ▶ %s --model-name %s %s\n' "$CMD" "$model" "${args[*]}"
	else
		if ((${#_env[@]})); then
			printf 'env: %s\n' "${_env[*]}"
			printf '▶ %s --model-name %s %s\n' "$CMD" "$model" "${args[*]}"
			env "${_env[@]}" $CMD --model-name "$model" "${args[@]}"
		else
			printf '▶ %s --model-name %s %s\n' "$CMD" "$model" "${args[*]}"
			$CMD --model-name "$model" "${args[@]}"
		fi
	fi

}

mapfile -t RUN_LINES <<<"$RUNS"

for line in "${RUN_LINES[@]}"; do
	model=$(jq -r '.model' <<<"$line")
	mapfile -t ARGS < <(jq -r '.args[]?' <<<"$line")
	mapfile -t ENV_KV < <(jq -r '.env | to_entries[]? | "\(.key)=\(.value)"' <<<"$line")

	run_with_env ENV_KV "$model" "${ARGS[@]}"
done
