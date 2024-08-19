import argparse
import json
import nltk
import numpy as np
import tqdm
import functools
from functools import partial
import pickle
import os
import random
from sentence_transformers import SentenceTransformer, util
import torch

from utils import load_shared_args, detectgpt_detect
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

nltk.download('punkt')


parser = argparse.ArgumentParser()
load_shared_args(parser)
#parser.add_argument('--paraphrase_times', default='paraphrase_outputs', type=str)
parser.add_argument('--prob_threshold', default=0.1, type=float)
parser.add_argument('--sim_threshold', default=0.75, type=float)
parser.add_argument('--total_tokens', default=30, type=int)
parser.add_argument('--mask_model', default="t5-3b", type=str)
parser.add_argument('--pool_size', default=5, type=int)
parser.add_argument('--embedder', default='all-MiniLM-L6-v2', type=str)
parser.add_argument('--data', default='data/gpt-4o-mini.jsonl_pp', type=str)
args = parser.parse_args()