#!/bin/bash


############
# Usage
############

# bash script_main_xx.sh


############
# SBM_PATTERN - 4 RUNS  
# python main_pcg_Tree_classification1.py --gpu_id 3 --seed 0 --config 'conf/PCG_xgb_Classification1.json'
############

seed0=42
seed1=96
seed2=13
seed3=36
code=main_pcg_Tree_classification1.py 
dataset=PCG_signal
tmux new -s t_pcg2 -d
#tmux send-keys "source activate myvenv1" C-m
tmux send-keys "
python $code --gpu_id 3 --seed $seed2 --config 'conf/PCG_xgb_Classification1.json' &
wait" C-m
tmux send-keys "tmux kill-session -t t_pcg2" C-m









