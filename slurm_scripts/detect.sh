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

cd /cbica/home/hewei/projects/detect_framework

# Define data and model parameters based on the task ID
case $SLURM_ARRAY_TASK_ID in
  1)
    DATA="data/gpt-4o-mini.jsonl_pp"
    ;;
  2)
    DATA="data/gpt2_xl.jsonl_pp"
    ;;
  3)
    DATA="data/gpt3.jsonl_pp"
    ;;
  4)
    DATA="data/opt_13b.jsonl_pp"
    ;;
esac

# Run the Python script with parameters
python sequence_prob.py --data $DATA

