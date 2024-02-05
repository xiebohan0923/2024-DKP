import json
import os
from torch.utils.data import Dataset,DataLoader
import torch
from dgl import load_graphs
from sklearn.model_selection import train_test_split
import numpy as np
from copy import deepcopy

def load_data(path1,path2,path3,path4,path5,instruct):

    with open(path1,encoding='utf-8') as f:
        datas = json.load(f)
    with open(path5) as file:
        fail_list = file.read().split(' ')[2:-1]
        fail_list = [int(x) for x in fail_list]

    instruct_data = []

    for qid,data in enumerate(datas):
        path = 'new_q'+str(qid)
        q_and_a = data
        if not os.path.exists(os.path.join(path2,path)):
            continue
        with open(os.path.join(path2,path)) as que:
            triples = list(json.load(que).keys())
            # triples.reverse()
            triples = triples[:min(50,len(triples))]
        
        for i in range(len(triples)):
            triples[i] = triples[i].replace(',','\t')
        sample = {"instruction":instruct}
        sample['input'] = {'input question':q_and_a['question'],'triples':triples}
        # sample['input'] = {'input question':q_and_a['question']}
        # sample['output'] = list(q_and_a['answer'].values()) # qald
        sample['output'] = q_and_a['answers'] # web
        # sample['output'] = [q_and_a['answer']] # cwq,creak
        sample['ids'] = qid
        instruct_data.append(sample)

    # _,new_data = train_test_split(instruct_data,shuffle=True,test_size=1000/len(instruct_data))
    # train_data,test_data = train_test_split(new_data,shuffle=True,test_size=0.2)


    train_data,test_data = train_test_split(instruct_data,shuffle=True,test_size=0.02)
    acc = 0


    temp_train = deepcopy(train_data)
    temp_test = deepcopy(test_data)

    for i in range(len(temp_train)):
        if temp_train[i]['ids'] not in fail_list:
            train_data.remove(temp_train[i])
    for i in range(len(temp_test)):
        if temp_test[i]['ids'] not in fail_list:
            acc+=1
            test_data.remove(temp_test[i])

    
    print(len(train_data),len(test_data),acc) 

    with open(path3,'w') as f:
        json.dump(train_data,f)
    with open(path4,'w') as f:
        json.dump(test_data,f)


if __name__ == "__main__":
    path1='data1/cwq.json'
    path2='cwq_new_data'
    path3 = 'instruct_data/in_cwq/train/instruct_cwq.json'
    path4 = 'instruct_data/in_cwq/instruct_cwq.json'
    path5 = 'success_and fail/fail_cwq.txt'
    instruct = ''