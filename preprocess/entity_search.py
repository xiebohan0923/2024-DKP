import json
from queue import Queue
from wiki_query import search_rela_entity , qid2label , label2id
from tqdm import tqdm
import time
import os


data_path = 'your data path'
output_dir = 'your outputdir'
D = int('max depth')


with open(data_path,encoding='utf-8') as f:
    datas = json.load(f)

for q_id,data in tqdm(enumerate(datas)):

    entity_num = q_id
    entities =Queue()
    if f'q{entity_num}' in os.listdir(output_dir):
        entity_num+=1
        print(f'question {entity_num} has been solved!')
        continue

    # webquestion
    # en = data['url']
    # label = en[en.rfind('/')+1:]  

    # cwq
    # label = list(data['topic_entity'].values())
    # en = [label2id(label)]

    # qald creak
    en = data['topic_entity'].keys()
    en_search = dict()
    visited = []
    depth = 0
    for x in en:
        if not x:
            continue
        entities.put(x)
        en_search[x] = [] 
        depth = D
    entities.put(' ')
    while(not entities.empty() and depth < 1): 
        current_entity = entities.get()
        if current_entity in visited:
            continue
        visited.append(current_entity)  
        if current_entity == ' ':
            depth+=1
            print(depth)
            entities.put(' ')
            continue
        new_entity , triples = search_rela_entity(current_entity,True)
        if len(new_entity) == 0 or len(triples) == 0:
            continue
        for tr in tqdm(triples):
            if '-' in current_entity or '-' in tr[0] or '-' in tr[1]:
                continue
            head = qid2label(current_entity)
            rela = qid2label(tr[0])
            tail = qid2label(tr[1])
            if rela and tail:
                if [head,rela,tail] not in en_search[current_entity]:
                    en_search[current_entity].append([head,rela,tail]) 
        if depth == 0:
            continue
        for en in new_entity:
            if en not in visited:
                entities.put(en)
                if en not in en_search.keys(): 
                    en_search[en] = []
 
    with open(output_dir+f'\q{entity_num}','w') as f:
        json.dump(en_search,f)
        print(f'question{entity_num} has been searched!')


