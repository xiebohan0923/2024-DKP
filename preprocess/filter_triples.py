import json
import os


from load_bert_data import load_data
from transformers import BertModel
from transformers import BertTokenizerFast
from transformers import AutoConfig
from transformers import LlamaForCausalLM, LlamaTokenizerFast
import torch
import numpy as np
device = torch.device('cuda:0')
from tqdm import tqdm

import random

# BERT_PATH = 'bert-base-uncased'
# tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)
# bert_model = BertModel.from_pretrained(BERT_PATH)

config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
        "output_hidden_states": True
    }

ori_model_path = "/data/llmweights/llama-7b-hf/"

config = AutoConfig.from_pretrained(ori_model_path, **config_kwargs)
tokenizer = LlamaTokenizerFast.from_pretrained(ori_model_path)
tokenizer.pad_token = tokenizer.eos_token

llama_model = LlamaForCausalLM.from_pretrained(
        ori_model_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        revision='main'
    ).to(device)

QandA,Triples = load_data('data_path','data_with_knowledge')

paths = os.listdir('web_data')

for i,q_and_a in enumerate(QandA):
    if 'new_'+paths[i] in os.listdir('filtered data path'):
        print(f'{paths[i]} has been searched!')
        continue
    question = q_and_a['question']
    # ans = list(q_and_a['answer'].values())
    ans = q_and_a['answers']
    if len(ans) == 1 :
        ans = ans[0]+'.'
    elif len(ans) > 0:
        fin_ans = ''
        ans = [fin_ans + sub_ans + ',' for sub_ans in ans]
        ans = fin_ans[:-1]+'.'
    else:
        continue

    triple_score = {}
    if len(Triples[i]) == 0:
        with open(os.path.join('filtered data path','new_'+paths[i]),'w') as f:
            json.dump(triple_score,f)
            print(f'{paths[i]} has been searched!')
        continue
    sen_input = tokenizer.encode_plus(question+ans)
    sentence_vec = llama_model.model.embed_tokens(torch.tensor(sen_input["input_ids"]).unsqueeze(0).to(device))


    tr_to_input = tokenizer.batch_encode_plus(Triples[i],max_length = 20 ,pad_to_max_length=True,return_offsets_mapping=True)
    tr_to_input['sentence'] = Triples[i]


    tr_input_list = [tr_to_input]
    

    for tr_input in tr_input_list:
        tr_id = 0
        triples_vec = llama_model.model.embed_tokens(torch.tensor(tr_input["input_ids"]).to(device))
        sen_vec = torch.mean(sentence_vec,1)
        tri_vec = torch.mean(triples_vec,1)
        cos_sim = torch.matmul(sen_vec,tri_vec.permute(1,0))

        # topk_ids = torch.topk(cos_sim , min(100,cos_sim.shape[1])).indices
        topk_ids = [random.sample(range(0,cos_sim.shape[1]),min(100,cos_sim.shape[1]))]

        # top-k tqdm(topk_ids[0].tolist())
        for j in tqdm(topk_ids[0]):
            logits = llama_model(input_ids = torch.tensor(tr_input['input_ids'][j]).unsqueeze(0).to(device),attention_mask = torch.tensor(tr_input['attention_mask'][j]).unsqueeze(0).to(device)).logits
            score = 0
            for token_id,next_tok in enumerate(tr_input['input_ids'][j]):
                score += logits[:,token_id,next_tok]
            triple_score[tr_input['sentence'][j]] = score.item()

    triple_score = {key: value for key, value in sorted(triple_score.items(), key=lambda item: item[1],reverse=False)}

    with open(os.path.join('filtered data path','new_'+paths[i]),'w') as f:
        json.dump(triple_score,f)
        print(f'{paths[i]} has been searched!')

    




