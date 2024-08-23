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
from torch.ao.nn.quantized.functional import threshold

from utils import load_shared_args, get_ll, get_rank, get_entropy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

nltk.download('punkt')


parser = argparse.ArgumentParser()
load_shared_args(parser)
parser.add_argument('--prob_threshold', default=[-2.5, -10, -1.25, -3], type=list)
parser.add_argument('--sim_threshold', default=0.75, type=float)
parser.add_argument('--total_tokens', default=30, type=int)
parser.add_argument('--pool_size', default=5, type=int)
parser.add_argument('--embedder', default='all-MiniLM-L6-v2', type=str)
parser.add_argument('--data', default='data/gpt-4o-mini.jsonl_pp', type=str)
args = parser.parse_args()

# load base model and tokenizer
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


likelihood_fn = functools.partial(get_ll, base_model=model, base_tokenizer=tokenizer)
rank_fn = functools.partial(get_rank, base_model=model, base_tokenizer=tokenizer)
ranklog_fn = functools.partial(get_rank, base_model=model, base_tokenizer=tokenizer, log=True)
entropy_fn = functools.partial(get_entropy, base_model=model, base_tokenizer=tokenizer)
function_list = [likelihood_fn, rank_fn, ranklog_fn, entropy_fn]
function_name_list = ['likelihood', 'rank', 'rank_log', 'entropy']
thresholds = args.prob_threshold
save_name = args.data[5:-9]
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

        if "paraphrase_outputs" not in dd:
            continue
# gen: ai-generate; gold: human; pp0: paraphrase
        pp0_tokens = dd['paraphrase_outputs'][args.paraphrase_type]['output'][0].split()
        pp_target_tokens = dd['paraphrase_outputs']['lex_40_order_40']['output'][0].split()
        min_len = min(len(gold_tokens), len(pp_target_tokens), len(gen_tokens))
        if min_len <= args.total_tokens:
            continue

        if len(gold_tokens) >= args.total_tokens and len(pp_target_tokens) >= args.total_tokens:
            cands.append({
                "generation": " ".join(gen_tokens[:min_len]),
                "human": " ".join(gold_tokens[:min_len]),
                "paraphrase": " ".join(pp0_tokens[:min_len]),
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

for i in tqdm.tqdm(range(len(function_name_list))):
    func = function_name_list[i]
    th = thresholds[i]
    cache_path = f'detect_cache/{func}_{save_name}_cache.json'
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)
        # save a copy of cache as a backup
        with open(cache_path + ".bak", "w") as f:
            json.dump(cache, f)
    else:
        cache = {}
    print(f"Cache contains {len(cache)} entries.")

    detect_fn = function_list[i]
    score_list = []
    pool_num = []
    count = 0

    for cand in tqdm.tqdm(cands_squence):
        count += 1
        score, cache1 = detect_fn(cand, cache)
        if cache1:
            # save cache
            with open(cache_path, "w") as f:
                json.dump(cache, f)
        if len(pool) == 0:
            score_list.append((score, 0))
            if func != 'likelihood':
                score = -score
            if score >= th:
                embedding = embedder.encode(cand, convert_to_tensor=True)
                pool.append(embedding)
        else:
            pool_embeddings = torch.stack(pool, dim=0).to("cuda")
            embedding = embedder.encode(cand, convert_to_tensor=True).to("cuda")
            search_result = util.semantic_search(embedding, pool_embeddings)[0][0]
            index = search_result['corpus_id']
            sim_score = search_result['score']
            score_list.append((score, sim_score))
            if func != 'likelihood':
                score = -score
            if score >= th and sim_score < args.sim_threshold:
                pool.append(embedding)
            elif score < th and sim_score >= args.sim_threshold:
                del pool[index]
                pool.append(embedding)
        pool_num.append((count, len(pool)))

    print('The number of text in pool: ', len(pool))


    with open(f"score/score_{save_name}_{th}_{args.sim_threshold}_{args.pool_size}_{func}.pkl",
              'wb') as f:
        pickle.dump(score_list, f)

    with open(f"pool/pool_{save_name}_{th}_{args.sim_threshold}_{args.pool_size}_{func}.pkl",
              'wb') as f:
        pickle.dump(pool_num, f)

