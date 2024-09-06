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
parser.add_argument('--pool_size', default=5, type=float)
parser.add_argument('--embedder', default='all-MiniLM-L6-v2', type=str)
parser.add_argument('--data', default='data/gpt-4o-mini.jsonl_pp_pp2_pp3', type=str)
args = parser.parse_args()

# tokenizer = AutoTokenizer.from_pretrained(args.base_model)


# load detectgpt
if args.base_model is not None and args.base_model not in ["none", "None", "text-davinci-003"]:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model.eval()
    model.cuda()
elif args.base_model == "text-davinci-003":
    model = "text-davinci-003"
    tokenizer = None
else:
    model = None
    tokenizer = None

print(f"Loading mask model of type {args.mask_model}...")
if args.mask_model is not None and args.mask_model not in ["none", "None"]:
    mask_tokenizer = AutoTokenizer.from_pretrained(args.mask_model)
    mask_model = AutoModelForSeq2SeqLM.from_pretrained(args.mask_model)
    mask_model.eval()
    mask_model.cuda()
else:
    mask_model = None
    mask_tokenizer = None

if os.path.exists(args.detector_cache):
    with open(args.detector_cache, "r") as f:
        cache = json.load(f)
    # save a copy of cache as a backup
    with open(args.detector_cache + ".bak", "w") as f:
        json.dump(cache, f)
else:
    cache = {}
print(f"Cache contains {len(cache)} entries.")

detect_fn = functools.partial(detectgpt_detect, mask_model=mask_model, mask_tokenizer=mask_tokenizer, base_model=model, base_tokenizer=tokenizer)
'''the input of detect_fn should be a string'''
print("detectgpt loaded")

# get database and inputs
cands = []
truncate_tokens = 10000 #args.total_tokens

corpora_files = [args.data]
for op_file in corpora_files:
    # read args.output_file
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

        if "paraphrase_outputs" not in dd or "paraphrase_outputs_2" not in dd or "paraphrase_outputs_3" not in dd:
            continue
# gen: ai-generate; gold: human; pp0: paraphrase
        pp0_tokens = dd['paraphrase_outputs'][args.paraphrase_type]['output'][0].split()
        pp_target_tokens = dd['paraphrase_outputs']['lex_40_order_40']['output'][0].split()
        pp2_tokens = dd['paraphrase_outputs_2']['lex_40_order_40']['output'][0].split()
        pp3_tokens = dd['paraphrase_outputs_3']['lex_40_order_40']['output'][0].split()
        min_len = min(len(gold_tokens), len(pp_target_tokens), len(gen_tokens), len(pp2_tokens), len(pp3_tokens))
        if min_len <= args.total_tokens:
            continue

        if len(gold_tokens) >= args.total_tokens and len(pp_target_tokens) >= args.total_tokens:
            cands.append({
                "generation": " ".join(gen_tokens[:min_len]),
                "human": " ".join(gold_tokens[:min_len]),
                "paraphrase": " ".join(pp0_tokens[:min_len]),
                "paraphrase_2": " ".join(pp2_tokens[:min_len]),
                "paraphrase_3": " ".join(pp3_tokens[:min_len]),
                "idx": idx
            })

print('Number of questions:', len(cands))
cands_squence = []
ai_gen = []
for i in cands:
    cands_squence.append(i["generation"])
    ai_gen.append(i["generation"])
    cands_squence.append(i["human"])
    cands_squence.append(i["paraphrase"])
    cands_squence.append(i["paraphrase_2"])
    cands_squence.append(i["paraphrase_3"])
print('The lens of the input sequence is:', len(cands_squence))

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
for cand in tqdm.tqdm(cands_squence):
    count += 1
    score, cache1 = detect_fn(cand, cache)
    if cache1:
        # save cache
        with open(args.detector_cache, "w") as f:
            json.dump(cache, f)
    if len(pool) == 0:
        score_list.append((score, 0))
        if score >= args.prob_threshold:
            embedding = embedder.encode(cand, convert_to_tensor=True)
            pool.append(embedding)
    else:
        pool_embeddings = torch.stack(pool, dim=0).to("cuda")
        embedding = embedder.encode(cand, convert_to_tensor=True).to("cuda")
        search_result = util.semantic_search(embedding, pool_embeddings)[0][0]
        index = search_result['corpus_id']
        sim_score = search_result['score']
        score_list.append((score, sim_score))
        if score >= args.prob_threshold and sim_score < args.sim_threshold:
            pool.append(embedding)
        elif score < args.prob_threshold and sim_score >= args.sim_threshold:
            del pool[index]
            pool.append(embedding)
    pool_num.append((count, len(pool)))

print('The number of text in pool:', len(pool))

save_name = args.data[5:-17] + '_pp3'

with open(f"score/score_{save_name}_{args.prob_threshold}_{args.sim_threshold}_{args.pool_size}_detectgpt.pkl", 'wb') as f:
    pickle.dump(score_list, f)

with open(f"pool/pool_{save_name}_{args.prob_threshold}_{args.sim_threshold}_{args.pool_size}_detectgpt.pkl", 'wb') as f:
    pickle.dump(pool_num, f)

