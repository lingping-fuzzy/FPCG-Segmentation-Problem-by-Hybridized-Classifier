#!/bin/bash


############
# Usage
############

# bash script_main_xx.sh


############
# SBM_PATTERN - 4 RUNS  
# python main_pcg_signal_classification.py --gpu_id 0 --seed 0 --config 'conf/PCG_1.json'
############

seed0=42
seed1=96
seed2=13
seed3=36
code=main_pcg_Tree_classification.py 
dataset=PCG_signal
tmux new -s t_pcg1 -d
#tmux send-keys "source activate myvenv1" C-m
tmux send-keys "
python $code --gpu_id 0 --seed $seed0 --config 'conf/PCG_xgb_Classification.json' &
wait" C-m
tmux send-keys "tmux kill-session -t t_pcg1" C-m









