#!/usr/bin/env bash
set -euo pipefail

# One evaluation per GPU; this scheduler uses per-GPU locks to dole out runs.
CMD="torchrun --nnodes=1 --nproc_per_node 1"
DRY_RUN="${DRY_RUN:-0}"
MODEL_FILTER="${MODEL_FILTER:-}"
MODEL_CONFIG_FILE="${MODEL_CONFIG_FILE:-"./configs/model_configs.json"}"
QUANT_CONFIG_FILE="${QUANT_CONFIG_FILE:-"./configs/quant_configs.json"}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-"./output/eval"}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-24500}"
NUM_GPUS="${NUM_GPUS:-}"

if [[ -z "$NUM_GPUS" ]]; then
	if command -v nvidia-smi &>/dev/null; then
		NUM_GPUS=$(nvidia-smi -L | wc -l)
	else
		NUM_GPUS=1
	fi
fi

GPU_STATUS_FILE=$(mktemp /tmp/eval_gpu_status.XXXXXX)
for i in $(seq 0 $((NUM_GPUS - 1))); do
	echo "0" >"${GPU_STATUS_FILE}_$i"
done

terminate_children() {
	# Ensure all background eval jobs (and their children) are cleaned up.
	local pids
	pids=$(jobs -p)
	if [[ -n "$pids" ]]; then
		kill -9 $pids 2>/dev/null || true
	fi
}

cleanup() {
	terminate_children
	for i in $(seq 0 $((NUM_GPUS - 1))); do
		rm -f "${GPU_STATUS_FILE}_$i"
	done
	rm -f "$GPU_STATUS_FILE"
}
trap cleanup EXIT
trap 'terminate_children; exit 1' INT TERM

# ---- JQ program: merge common env + model + quant ---------------------------
JQ_PROG='
  . as $root
  | ($root.models[]) as $m
  | $qc[0].configs[] as $q
  | {
      model: $m.name,
      args: ($root.common.eval_args + ($m.eval_args // []) + ($q.args // [])),
      env:  (($root.common.env // {}) + ($m.env // {}) + ($q.env // {}))
    }
'

# ---- Phase 1: Parse all runs into a variable --------------------------------
RUNS=$(jq -c --slurpfile qc "$QUANT_CONFIG_FILE" "$JQ_PROG" "$MODEL_CONFIG_FILE")

# Optional filter (exact match)
if [[ -n "$MODEL_FILTER" ]]; then
	RUNS=$(jq -c --arg mf "$MODEL_FILTER" 'select(.model==$mf)' <<<"$RUNS")
fi

# ---- GPU helpers -------------------------------------------------------------
get_free_gpu() {
	while true; do
		for i in $(seq 0 $((NUM_GPUS - 1))); do
			if [[ $(cat "${GPU_STATUS_FILE}_$i") -eq 0 ]]; then
				echo "$i"
				return
			fi
		done
		sleep 1
	done
}

run_job() {
	local gpu_id="$1"
	shift
	local model="$1"
	shift

	# Everything up to the first -- is env, the rest is args
	local -a env_ref=()
	while [[ $# -gt 0 ]]; do
		if [[ "$1" == "--" ]]; then
			shift
			break
		fi
		env_ref+=("$1")
		shift
	done
	local -a args_ref=("$@")

	(
		local master_port=$((MASTER_PORT_BASE + gpu_id))
		local -a envs=("${env_ref[@]}" "CUDA_VISIBLE_DEVICES=$gpu_id")
		local -a cmd=(
			$CMD
			--master-port "$master_port"
			./exps/eval_llm.py
			--model-name "$model"
			"${args_ref[@]}"
		)

		if ((DRY_RUN)); then
			if ((${#envs[@]})); then
				printf 'DRY env: %s\n' "${envs[*]}"
			fi
			printf 'DRY ▶ %s\n' "${cmd[*]}"
		else
			if ((${#envs[@]})); then
				printf 'env: %s\n' "${envs[*]}"
			fi
			printf '▶ %s\n' "${cmd[*]}"
			env "${envs[@]}" "${cmd[@]}"
		fi

		echo "0" >"${GPU_STATUS_FILE}_$gpu_id"
	) &
}

# ---- Phase 2: Execute --------------------------------------------------------
mapfile -t RUN_LINES <<<"$RUNS"

for line in "${RUN_LINES[@]}"; do
	model=$(jq -r '.model' <<<"$line")
	mapfile -t ARGS < <(jq -r '.args[]?' <<<"$line")
	mapfile -t ENV_KV < <(jq -r '.env | to_entries[]? | "\(.key)=\(.value)"' <<<"$line")

	gpu=$(get_free_gpu)
	echo "1" >"${GPU_STATUS_FILE}_$gpu"
	printf 'Dispatching %s on GPU %s\n' "$model" "$gpu"
	run_job "$gpu" "$model" "${ENV_KV[@]}" -- "${ARGS[@]}"
done
wait
