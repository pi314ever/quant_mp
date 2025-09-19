set -e

run() {
	model=$1
	quant_config=$2
	do_train=$3

	echo "Running $model with quant config $quant_config"
	accelerate launch ./exps/run_exp_llm.py \
		--model_name "$model" \
		--do_train "$do_train" \
		--deepspeed ds_zero3.json \
		--do_eval True \
		--model_max_length 1024 \
		--fp16 False \
		--bf16 True \
		--log_on_each_node False \
		--num_train_epochs 1 \
		--per_device_train_batch_size 2 \
		--per_device_eval_batch_size 4 \
		--gradient_accumulation_steps 1 \
		--ddp_find_unused_parameters False \
		--save_strategy "no" \
		--learning_rate 2e-5 \
		--weight_decay 0.01 \
		--warmup_ratio 0. \
		--lr_scheduler_type "cosine" \
		--logging_steps 1 \
		--tf32 False \
		--gradient_checkpointing True \
		--use_cache False \
		--qat True \
		--train_ds_path ./data/train.jsonl \
		--valid_ds_path ./data/valid.jsonl \
		$quant_config
}

models=(
	"facebook/MobileLLM-125M"
	"facebook/MobileLLM-600M"
	"meta-llama/Llama-3.2-1B"
	"meta-llama/Llama-3.2-3B"
	"meta-llama/Meta-Llama-3-8B"
)

quant_configs=(
	"--label BF16-baseline"
	"--weight_dformat fp4_e2m1 --weight_alg minmax --weight_block_size channel"
	"--weight_dformat fp4_e2m1 --weight_alg iterative --weight_block_size channel"
	"--weight_dformat fp4_e2m1 --weight_alg analytic --weight_block_size channel"
	"--weight_dformat fp4_e2m1 --weight_alg lsq --weight_block_size channel"
	"--weight_dformat fp4_e2m1 --weight_alg octav --weight_block_size channel"
	"--weight_dformat int4 --weight_alg minmax --weight_block_size channel"
	"--weight_dformat int4 --weight_alg iterative --weight_block_size channel"
	"--weight_dformat int4 --weight_alg analytic --weight_block_size channel"
	"--weight_dformat int4 --weight_alg lsq --weight_block_size channel"
	"--weight_dformat int4 --weight_alg octav --weight_block_size channel"
	"--weight_dformat sf4 --weight_alg minmax --weight_block_size channel"
	"--weight_dformat sf4 --weight_alg iterative --weight_block_size channel"
	"--weight_dformat sf4 --weight_alg analytic --weight_block_size channel"
	"--weight_dformat sf4 --weight_alg lsq --weight_block_size channel"
	"--weight_dformat sf4 --weight_alg octav --weight_block_size channel"
	"--weight_dformat nf4 --weight_alg minmax --weight_block_size channel"
	"--weight_dformat nf4 --weight_alg iterative --weight_block_size channel"
	"--weight_dformat nf4 --weight_alg analytic --weight_block_size channel"
	"--weight_dformat nf4 --weight_alg lsq --weight_block_size channel"
	"--weight_dformat nf4 --weight_alg octav --weight_block_size channel"
	"--weight_dformat fp4_e2m1 --weight_alg minmax"
	"--weight_dformat fp4_e2m1 --weight_alg iterative"
	"--weight_dformat fp4_e2m1 --weight_alg analytic"
	"--weight_dformat fp4_e2m1 --weight_alg lsq"
	"--weight_dformat fp4_e2m1 --weight_alg octav"
	"--weight_dformat int4 --weight_alg minmax"
	"--weight_dformat int4 --weight_alg iterative"
	"--weight_dformat int4 --weight_alg analytic"
	"--weight_dformat int4 --weight_alg lsq"
	"--weight_dformat int4 --weight_alg octav"
	"--weight_dformat sf4 --weight_alg minmax"
	"--weight_dformat sf4 --weight_alg iterative"
	"--weight_dformat sf4 --weight_alg analytic"
	"--weight_dformat sf4 --weight_alg lsq"
	"--weight_dformat sf4 --weight_alg octav"
	"--weight_dformat nf4 --weight_alg minmax"
	"--weight_dformat nf4 --weight_alg iterative"
	"--weight_dformat nf4 --weight_alg analytic"
	"--weight_dformat nf4 --weight_alg lsq"
	"--weight_dformat nf4 --weight_alg octav"
)

set -x
for model in "${models[@]}"; do
	for quant_config in "${quant_configs[@]}"; do
		run "$model" "$quant_config" "True"
	done
done
