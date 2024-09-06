#!/bin/bash
#SBATCH      --job-name="detectai"
#SBATCH      --mail-user="weiqingh@sas.upenn.edu"
#SBATCH      --time=1:00:00
#SBATCH      --mem=80G
#SBATCH      --gpus=a40
#SBATCH      --mail-type=ALL
#SBATCH      --array=1-6
#SBATCH      --output=slurm_output/output_%A_%a.log




source activate detectai

cd /cbica/home/hewei/projects/detect_framework

# Define data and model parameters based on the task ID
case $SLURM_ARRAY_TASK_ID in
  1)
    DATA="data/gpt2_xl_wm.jsonl_pp"
    CA="detect_cache/watermarking_gpt2.json"
    P=3
    ;;
  2)
    DATA="data/opt_13b_wm.jsonl_pp"
    CA="detect_cache/watermarking_opt.json"
    P=3
    ;;
  3)
    DATA="data/gpt2_xl_wm.jsonl_pp"
    CA="detect_cache/watermarking_gpt2.json"
    P=2
    ;;
  4)
    DATA="data/opt_13b_wm.jsonl_pp"
    CA="detect_cache/watermarking_opt.json"
    P=2
    ;;
  5)
    DATA="data/gpt2_xl_wm.jsonl_pp"
    CA="detect_cache/watermarking_gpt2.json"
    P=1.5
    ;;
  6)
    DATA="data/opt_13b_wm.jsonl_pp"
    CA="detect_cache/watermarking_opt.json"
    P=1.5
    ;;
esac

# Run the Python script with parameters
python sequence_watermarking.py --data $DATA --detector_cache $CA --pool_size $P

