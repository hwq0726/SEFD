import argparse
import json
import functools
import os
import pickle
import numpy as np
import tqdm
import torch
from functools import partial
import random
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

from transformers import AutoTokenizer
from utils import (watermark_detect, get_roc, print_tpr_target, print_accuracies, do_sim_stuff,
                   load_sim_stuff, get_longest_answer)


parser = argparse.ArgumentParser()
parser.add_argument('--threshold', default=4.0, type=float)
parser.add_argument('--sim_threshold', default=0.75, type=float)
parser.add_argument('--total_tokens', default=200, type=int)
parser.add_argument('--paraphrase_no_exist_behavior', default='skip', type=str)
parser.add_argument('--data', default='data/gpt2_xl_wm.jsonl_pp', type=str)
parser.add_argument('--detector_cache', default="detect_cache/watermarking.json", type=str)
parser.add_argument('--embedder', default='all-MiniLM-L6-v2', type=str)
parser.add_argument('--pool_size', default=5, type=int)
args = parser.parse_args()

watermark_fraction = 0.5

if "/gpt2" in args.data:
    tokenizer = AutoTokenizer.from_pretrained(f"gpt2-xl", torch_dtype=torch.float16)
elif "/opt" in args.data:
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-13b", torch_dtype=torch.float16)

if os.path.exists(args.detector_cache):
    with open(args.detector_cache, "r") as f:
        cache = json.load(f)
    # save a copy of cache as a backup
    with open(args.detector_cache + ".bak", "w") as f:
        json.dump(cache, f)
else:
    cache = {}

if "/opt" in args.data:
    vocab_size = 50272
else:
    vocab_size = tokenizer.vocab_size

detect_fn = functools.partial(watermark_detect, watermark_fraction=watermark_fraction, vocab_size=vocab_size)

# get database and inputs
cands = []
truncate_tokens = 10000 #args.total_tokens
corpora_files = [args.data]
for op_file in corpora_files:
    # read args.data
    with open(op_file, "r") as f:
        data = [json.loads(x) for x in f.read().strip().split("\n")]

    # iterate over data and tokenize each instance
    for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
        if isinstance(dd['gen_completion'], str):
            gen_tokens = dd['gen_completion']
        else:
            gen_tokens = dd['gen_completion'][0]
        gen_tokens = gen_tokens.split()
        gold_tokens = dd['gold_completion'].split()

        if len(gen_tokens) <= args.total_tokens:
            continue

        if "paraphrase_outputs" not in dd:
            continue
# gen: ai-generate; gold: human; pp0: paraphrase
        pp0_tokens = dd['paraphrase_outputs']['lex_40_order_40']['output'][0].split()
        min_len = min(len(gold_tokens), len(pp0_tokens), len(gen_tokens))
        if min_len <= args.total_tokens:
            continue

        if len(gold_tokens) >= args.total_tokens and len(pp0_tokens) >= args.total_tokens:
            cands.append({
                "prefix": dd['prefix'],
                "generation": " ".join(gen_tokens[:min_len]),
                "human": " ".join(gold_tokens[:min_len]),
                "paraphrase": " ".join(pp0_tokens[:min_len]),
                "idx": idx
            })

print('Number of questions:', len(cands))
ai_gen = [i["generation"] for i in cands]

# load embedder
embedder = SentenceTransformer(args.embedder)

# Initialize pool
if args.pool_size != 0:

    num_pool = int(len(ai_gen)/args.pool_size)
    pool_text = random.sample(ai_gen, num_pool)
    print('The number of text in the pool is:', len(pool_text))
    pool = []
    for text in pool_text:
        embedding = embedder.encode(text, convert_to_tensor=True).to("cuda")
        pool.append(embedding)

else:
    pool = []
print('Complete initialize pool')

score_list = []
pool_num = []
count = 0
for dd in tqdm.tqdm(cands):
    count += 3
    last_prefix_token = tokenizer(dd['prefix'], add_special_tokens=True)["input_ids"][-1]
    gen_tokens = [last_prefix_token] + tokenizer(dd['generation'], add_special_tokens=False)["input_ids"]
    gold_tokens = [last_prefix_token] + tokenizer(dd['human'], add_special_tokens=False)["input_ids"]
    pp0_tokens = [last_prefix_token] + tokenizer(dd['paraphrase'], add_special_tokens=False)["input_ids"]

    total_tokens = min(len(gen_tokens), len(gold_tokens), len(pp0_tokens))

    if total_tokens < args.total_tokens:
        continue

    gen_tokens = gen_tokens[:total_tokens]
    gold_tokens = gold_tokens[:total_tokens]
    pp0_tokens = pp0_tokens[:total_tokens]

    gen_z, cache1 = detect_fn(gen_tokens, cache)
    gold_z, cache2 = detect_fn(gold_tokens, cache)
    pp0_z, cache3 = detect_fn(pp0_tokens, cache)
    tem_list = [(gen_z, dd['generation']), (gold_z, dd['human']), (pp0_z, dd['paraphrase'])]

    if cache1 or cache2 or cache3:
        # save cache
        with open(args.detector_cache, "w") as f:
            json.dump(cache, f)
    for x in tem_list:
        score = x[0]
        cand = x[1]
        if len(pool) == 0:
            score_list.append((score, 0))
            if score >= args.threshold:
                embedding = embedder.encode(cand, convert_to_tensor=True)
                pool.append(embedding)
        else:
            pool_embeddings = torch.stack(pool, dim=0).to("cuda")
            embedding = embedder.encode(cand, convert_to_tensor=True).to("cuda")
            search_result = util.semantic_search(embedding, pool_embeddings)[0][0]
            index = search_result['corpus_id']
            sim_score = search_result['score']
            score_list.append((score, sim_score))
            if score >= args.threshold and sim_score < args.sim_threshold:
                pool.append(embedding)
            elif score < args.threshold and sim_score >= args.sim_threshold:
                del pool[index]
                pool.append(embedding)
        pool_num.append((count, len(pool)))

save_name = args.data[5:-9]

with open(f"score/score_{save_name}_{args.threshold}_{args.sim_threshold}_{args.pool_size}_watermark.pkl", 'wb') as f:
    pickle.dump(score_list, f)

with open(f"pool/pool_{save_name}_{args.threshold}_{args.sim_threshold}_{args.pool_size}_watermark.pkl", 'wb') as f:
    pickle.dump(pool_num, f)

print('test')

