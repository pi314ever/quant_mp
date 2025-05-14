set -e

run() {
	model=$1
	quant_config=$2

	echo "Running $model with quant config $quant_config"
	OMP_NUM_THREADS=8 torchrun --nnodes=1 --nproc_per_node=1 --master-port 24501 ./exps/eval_llm.py \
		--tasks arc_easy,arc_challenge,boolq,piqa,social_iqa,hellaswag,openbookqa,winogrande \
		$quant_config
}

models=("facebook/MobileLLM-125M" "facebook/MobileLLM-600M" "meta-llama/Llama-3.2-1B")

quant_configs=(
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg minmax --weight_block_size channel"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg iterative --weight_block_size channel"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg normal --weight_block_size channel"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg lsq --weight_block_size channel"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg minmax --weight_block_size channel"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg iterative --weight_block_size channel"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg normal --weight_block_size channel"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg lsq --weight_block_size channel"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg minmax"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg iterative"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg normal"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg lsq"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg minmax"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg iterative"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg normal"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg lsq"
)

set -x
for model in "${models[@]}"; do
	# Run once without quantconfig and training for bf16 baseline
	run "$model" "--label BF16-baseline"
	for quant_config in "${quant_configs[@]}"; do
		run "$model" "$quant_config"
	done
done
