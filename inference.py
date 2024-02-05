
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import random
 
import json
import torch
import transformers
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from dgl import load_graphs
from utils.prompter import Prompter 
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from gnn import GCNWithAdapter,HeteroRGCN,HeteroRGCNLayer

base_path = "/data/llmweights/llama-2-7b-hf/"
prompt_template_name = "alpaca"
prompter = Prompter(prompt_template_name)

from tqdm import tqdm


def load_test_dataset(path):
    test_dataset = json.load(open(path, "r"))
    return test_dataset

if __name__ == "__main__":
    cuda = "cuda:0"
    lora_weights = " " 
    test_data_path = " "
    data_set = ' '
    embedding_path = "{}/kg_embeddings.pth".format(lora_weights)
    fc_path = "{}/fc.pth".format(lora_weights)
    test_dataset = load_test_dataset(test_data_path)

    fc = torch.load(fc_path,map_location='cuda:0').to(cuda)
    gcn_embedding = torch.load(embedding_path,map_location='cuda:0').to(cuda)
    tokenizer = LlamaTokenizer.from_pretrained(base_path)
    model = LlamaForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16
    ).to(cuda)
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    ).to(cuda)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    tokenizer.padding_side = "left"  # Allow batched inference
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2
    model = model.eval()

    acc = 0
    fail_dict = {}

    for data in tqdm(test_dataset):
        ent = data["input"]
        ans = data["output"]
        ids = data["ids"]

        if os.path.exists(data_set+'_graph/new_q'+str(ids)+'.bin'):  # your graph data
            g = load_graphs(data_set+'_graph/new_q'+str(ids)+'.bin')[0][0].to(cuda)
            prefix = fc(gcn_embedding(g)[0].unsqueeze(0).unsqueeze(0)).half()
            rela_en = torch.zeros([1,768]).to(cuda)
            count = 0
            for stype, etype, dtype in g.canonical_etypes:
            # rela_weight[etype] = {'edge_weight': self.rela_embedding[int(etype)].unsqueeze(0).repeat(G[etype].num_edges(),1)}
                rela_en += torch.mean(g[etype].edata['weight'],dim = 0).unsqueeze(0)
                count+=1
            rela_en = fc((rela_en/count).unsqueeze(0)).half()

            prefix2 = fc(gcn_embedding(g)[-1].unsqueeze(0).unsqueeze(0)).half()

            prefix = torch.cat((prefix, rela_en), dim=1)
            prefix = torch.cat((prefix, prefix2), dim=1)
        else:
            prefix = torch.ones([1,3,4096]).to(cuda).half()
        
        
        
        prompt = prompter.generate_prompt(
            data["instruction"],
            ent)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(cuda)
        token_embeds = model.model.model.embed_tokens(input_ids)
        input_embeds = torch.cat((prefix, token_embeds), dim=1)
        # input_embeds = token_embeds
        generate_ids = model.generate(
            inputs_embeds=input_embeds, 
            max_new_tokens=500
        )
        context = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = response.replace(context, "").strip()
        response = response[:min(len(response),response.find('Input'))]

        with open(f'res/q_{ids}','w') as file:
            file.write(response,ans)
        

    
