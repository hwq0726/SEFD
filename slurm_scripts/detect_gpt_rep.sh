#!/bin/bash
#SBATCH      --job-name="detectai"
#SBATCH      --mail-user="weiqingh@sas.upenn.edu"
#SBATCH      --time=24:00:00
#SBATCH      --mem=80G
#SBATCH      --gpus=a40
#SBATCH      --mail-type=ALL
#SBATCH      --array=1-4
#SBATCH      --output=slurm_output/output_%A_%a.log




source activate detectai

cd /cbica/home/hewei/projects/detect_framework

# Define data and model parameters based on the task ID
case $SLURM_ARRAY_TASK_ID in
  1)
    DATA="data/gpt-4o-mini.jsonl_pp_pp2_pp3"
    CA="detect_cache/detectgpt_gpt_4o-mini_cache.json"
    P=5
    ;;
  2)
    DATA="data/gpt2_xl.jsonl_pp_pp2_pp3"
    CA="detect_cache/detectgpt_gpt2_xl_cache.json"
    P=5
    ;;
  3)
    DATA="data/gpt3.jsonl_pp_pp2_pp3"
    CA="detect_cache/detectgpt_gpt3_cache.json"
    P=5
    ;;
  4)
    DATA="data/opt_13b.jsonl_pp_pp2_pp3"
    CA="detect_cache/detectgpt_opt_13b_cache.json"
    P=5
    ;;
esac

# Run the Python script with parameters
python sequence_detectgpt_rep.py --data $DATA --detector_cache $CA --pool_size $P --prob_threshold 0.5 --sim_threshold 0.85

