set -e

run() {
	model=$1
	quant_config=$2

	echo "Running $model with quant config $quant_config"
	OMP_NUM_THREADS=8 torchrun --nnodes=1 --nproc_per_node=6 ./exps/run_exp_llm.py \
		--model_name "$model" \
		--do_train True \
		--do_eval True \
		--model_max_length 2048 \
		--fp16 False \
		--bf16 True \
		--log_on_each_node False \
		--logging_dir /tmp/output/runs/current \
		--num_train_epochs 1 \
		--per_device_train_batch_size 2 \
		--per_device_eval_batch_size 1 \
		--gradient_accumulation_steps 1 \
		--save_strategy "steps" \
		--save_steps 2000 \
		--save_total_limit 1 \
		--learning_rate 2e-5 \
		--weight_decay 0. \
		--warmup_ratio 0. \
		--lr_scheduler_type "cosine" \
		--logging_steps 1 \
		--tf32 False \
		--gradient_checkpointing False \
		--qat True \
		--train_ds_path ./train.jsonl \
		--valid_ds_path ./valid.jsonl \
		$quant_config
}

models=("facebook/MobileLLM-125M" "facebook/MobileLLM-600M" "meta-llama/Llama-3.2-1B")

quant_configs=(
	"--activation_qtype float --activation_qbits 4 --activation_format e2m1 --activation_alg minmax --weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg minmax"
	"--activation_qtype float --activation_qbits 4 --activation_format e2m1 --activation_alg iterative --weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg normal"
	"--activation_qtype float --activation_qbits 4 --activation_format e2m1 --activation_alg minmax --weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg normal"
	"--activation_qtype float --activation_qbits 4 --activation_format e2m1 --activation_alg minmax --weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg lsq"
	"--activation_qtype float --activation_qbits 4 --activation_format e2m1 --activation_alg lsq --weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg lsq"
	"--activation_qtype float --activation_qbits 4 --activation_format e3m0 --activation_alg minmax --weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg minmax"
	"--activation_qtype float --activation_qbits 4 --activation_format e3m0 --activation_alg iterative --weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg normal"
	"--activation_qtype float --activation_qbits 4 --activation_format e3m0 --activation_alg minmax --weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg normal"
	"--activation_qtype float --activation_qbits 4 --activation_format e3m0 --activation_alg minmax --weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg lsq"
	"--activation_qtype float --activation_qbits 4 --activation_format e3m0 --activation_alg lsq --weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg lsq"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg minmax"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg iterative"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg normal"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg lsq"
	"--activation_qtype uniform --activation_qbits 4 --activation_alg minmax --weight_qtype uniform --weight_qbits 4 --weight_alg minmax"
	"--activation_qtype uniform --activation_qbits 4 --activation_alg iterative --weight_qtype uniform --weight_qbits 4 --weight_alg normal"
	"--activation_qtype uniform --activation_qbits 4 --activation_alg minmax --weight_qtype uniform --weight_qbits 4 --weight_alg normal"
	"--activation_qtype uniform --activation_qbits 4 --activation_alg minmax --weight_qtype uniform --weight_qbits 4 --weight_alg lsq"
	"--activation_qtype uniform --activation_qbits 4 --activation_alg lsq --weight_qtype uniform --weight_qbits 4 --weight_alg lsq"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg minmax"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg iterative"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg normal"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg lsq"
)

set -x
for model in "${models[@]}"; do
	for quant_config in "${quant_configs[@]}"; do
		run "$model" "$quant_config"
	done
done
