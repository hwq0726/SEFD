#!/bin/bash
#SBATCH      --job-name="detectai"
#SBATCH      --mail-user="weiqingh@sas.upenn.edu"
#SBATCH      --time=12:00:00
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
    DATA="data/gpt-4o-mini.jsonl_pp_pp2"
    ;;
  2)
    DATA="data/gpt2_xl.jsonl_pp_pp2"
    ;;
  3)
    DATA="data/gpt3.jsonl_pp_pp2"
    ;;
  4)
    DATA="data/opt_13b.jsonl_pp_pp2"
    ;;
esac

# Run the Python script with parameters
python recursive_paraphrase.py --output_file $DATA --para_src "paraphrase_outputs_2" --para_dst "paraphrase_outputs_3"

