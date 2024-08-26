import argparse
import json
import tqdm
import functools
import pickle
import os
import random
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import RobertaTokenizer, RobertaModel
from utils import preprocess_text, get_mle_single
from skdim.id import MLE

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', default=-11.0, type=float)
parser.add_argument('--sim_threshold', default=0.75, type=float)
parser.add_argument('--total_tokens', default=50, type=int)
parser.add_argument('--data', default='data/gpt-4o-mini.jsonl_pp', type=str)
parser.add_argument('--model', default='FacebookAI/roberta-base', type=str)
parser.add_argument('--tokenizer', default='FacebookAI/roberta-base', type=str)
parser.add_argument('--detector_cache', default="detect_cache/ID_gpt4o.json", type=str)
parser.add_argument('--embedder', default='all-MiniLM-L6-v2', type=str)
parser.add_argument('--pool_size', default=5, type=int)
args = parser.parse_args()

# load model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer)
model = RobertaModel.from_pretrained(args.model).to('cuda')

# load cache
if os.path.exists(args.detector_cache):
    with open(args.detector_cache, "r") as f:
        cache = json.load(f)
    # save a copy of cache as a backup
    with open(args.detector_cache + ".bak", "w") as f:
        json.dump(cache, f)
else:
    cache = {}
print(f"Cache contains {len(cache)} entries.")

# load detector and embedder
detect_fn = functools.partial(get_mle_single, tokenizer=tokenizer, model=model, solver=MLE())
embedder = SentenceTransformer(args.embedder)

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
        pp0_tokens = dd['paraphrase_outputs']['lex_40_order_40']['output'][0].split()
        min_len = min(len(gold_tokens), len(pp0_tokens), len(gen_tokens))
        if min_len <= args.total_tokens:
            continue

        if len(gold_tokens) >= args.total_tokens and len(pp0_tokens) >= args.total_tokens:
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

# begin detection
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

print('The number of text in pool:', len(pool))

save_name = args.data[5:-9]

with open(f"score/score_{save_name}_{args.threshold}_{args.sim_threshold}_{args.pool_size}_ID-MLE.pkl", 'wb') as f:
    pickle.dump(score_list, f)

with open(f"pool/pool_{save_name}_{args.threshold}_{args.sim_threshold}_{args.pool_size}_ID-MLE.pkl", 'wb') as f:
    pickle.dump(pool_num, f)

