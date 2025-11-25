CUDA_VISIBLE_DEVICES=0,1,2,3 python exps/run_exp.py --dtype fp
CUDA_VISIBLE_DEVICES=0,1,2,3 python exps/run_exp.py --dtype int


python exps/gen_fig_exp_res.py --dtype fp
python exps/gen_fig_exp_res.py --dtype int