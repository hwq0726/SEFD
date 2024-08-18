#!/bin/bash
#SBATCH      --job-name="detectai"
#SBATCH      --mail-user="weiqingh@sas.upenn.edu"
#SBATCH      --time=4:00:00
#SBATCH      --mem=80G
#SBATCH      --gpus=a40
#SBATCH      --mail-type=ALL
#SBATCH      --array=1-4
#SBATCH      --output=slurm_output/output_%A_%a.log




source activate detectai

cd /cbica/home/hewei/projects/detect_gpt

# Define data and model parameters based on the task ID
case $SLURM_ARRAY_TASK_ID in
  1)
    DATA="data/gpt-4o-mini.jsonl_pp"
    CA="detect_cache/gpt_4o-mini_cache.json"
    POOL=3
    ;;
  2)
    DATA="data/gpt2_xl.jsonl_pp"
    CA="detect_cache/gpt2_xl_cache.json"
    POOL=3
    ;;
  3)
    DATA="data/gpt3.jsonl_pp"
    CA="detect_cache/gpt3_cache.json"
    POOL=3
    ;;
  4)
    DATA="data/opt_13b.jsonl_pp"
    CA="detect_cache/opt_13b_cache.json"
    POOL=3
    ;;
esac

# Run the Python script with parameters
python sequence_detectgpt.py --data $DATA --detector_cache $CA --pool_size $POOL

