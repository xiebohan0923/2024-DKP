
import os
import json
from tqdm import tqdm
import dgl
from dgl import save_graphs
import numpy as np
import torch

en_total = 0
rela_count = 0
r_dict = {}
node_seq = {}

from transformers import BertTokenizer,BertModel
BERT_PATH = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

bert_model = BertModel.from_pretrained(BERT_PATH)
bert_model.eval()

for path in tqdm(os.listdir('filtered data path')):
    node_dict = dict()
    rela_dict = dict()
    node_feat = []

    rela_weight = dict()
    entity_count = 0

    with open(os.path.join('filtered data path',path)) as file:
        data =  json.load(file)
        file.close()
    if len(data) == 0:
        continue
    for triple in data.keys():
        tr_list = triple.split(',')
        h = tr_list[0]
        t = tr_list[-1]
        tr_list.remove(h)
        tr_list.remove(t)

        r=''
        for sub_r in tr_list:
            r+=sub_r

        t = t[:-1]
        if h not in node_dict.keys():
            node_dict[h] = entity_count
            entity_count+=1
            tokens = torch.tensor([tokenizer.encode(h)])
            node_feat.append(bert_model(tokens)[-1].squeeze(0))

        if t not in node_dict.keys():
            node_dict[t] = entity_count
            entity_count+=1
            tokens = torch.tensor([tokenizer.encode(t)])
            node_feat.append(bert_model(tokens)[-1].squeeze(0))

        if r not in r_dict.keys():
            r_dict[r] = rela_count
            triple_str = ('entity',str(r_dict[r]),'entity')
            rela_count += 1
            tokens = torch.tensor([tokenizer.encode(r)])
            weight = bert_model(tokens)[-1].squeeze(0)

            if triple_str not in rela_dict.keys():
                rela_dict[triple_str] = [[],[]]
                rela_dict[triple_str][0].append(node_dict[h])
                rela_dict[triple_str][1].append(node_dict[t])
                rela_weight[triple_str] = [weight]
            else:
                triple_str = ('entity',str(r_dict[r]),'entity')
                rela_dict[triple_str][0].append(node_dict[h])
                rela_dict[triple_str][1].append(node_dict[t])
                rela_weight[triple_str].append(weight)
            
        else:
            triple_str = ('entity',str(r_dict[r]),'entity')
            if triple_str not in rela_dict.keys():
                rela_dict[triple_str] = [[],[]]
                rela_dict[triple_str][0].append(node_dict[h])
                rela_dict[triple_str][1].append(node_dict[t])
                rela_weight[triple_str] = [weight]
            else:
                triple_str = ('entity',str(r_dict[r]),'entity')
                rela_dict[triple_str][0].append(node_dict[h])
                rela_dict[triple_str][1].append(node_dict[t])
                rela_weight[triple_str].append(weight)

    node_seq[path] = [en_total,en_total+len(node_dict)]

    en_total+=len(node_dict)
    for item in rela_dict.keys():
        rela_dict[item] = (rela_dict[item][0],rela_dict[item][1])

    g = dgl.heterograph(rela_dict)

    for etype in g.canonical_etypes:
        g.edges[etype].data['weight'] = torch.stack(rela_weight[etype])
    g.ndata['feat'] = torch.stack(node_feat)
    save_graphs('graph data path'+path+'.bin',g)
    

print(en_total,rela_count)


with open('your relation dict path','w') as f:
    json.dump(r_dict,f)

 

