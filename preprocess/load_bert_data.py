import json
from transformers import BertTokenizerFast
import os
from torch.utils.data import Dataset,DataLoader
import torch


BERT_PATH = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)

max_qu_token = 28
max_ans_token = 15
max_triple_token = 400

def load_data(path1,path2):

    with open(path1,encoding='utf-8') as f:
        datas = json.load(f)
    QandA = []
    Triples = []
    for path in os.listdir(path2):
        q_and_a = datas[int(path[1:])]
        with open(os.path.join(path2,path)) as que:
            print(path)
            triples = json.load(que)
            que.close()
        # triple 预处理
        tri = []
        for tr_list in list(triples.values()):
            for h,r,t in tr_list:
                # tri += h+'\t'+r+'\t'+t+'\t.\t'
                if h==None or r==None or t==None:
                    continue
                tri.append(h+','+r+','+t+'.')
        QandA.append(q_and_a)
        Triples.append(tri)
    return QandA,Triples

def token_and_mask(QandA,Triples):
    batch_qu = []
    batch_ans = []
    for q_and_a in QandA:
        question = q_and_a['question']
        ans = list(q_and_a['answer'].values())
        if len(ans):
            ans = ans[0]
        else:
            ans = ''
        batch_qu.append(question)
        batch_ans.append(ans)
    qids = tokenizer.batch_encode_plus(batch_qu,max_length = max_qu_token+2,pad_to_max_length=True,add_special_tokens = False)
    aids = tokenizer.batch_encode_plus(batch_ans,max_length = max_ans_token+2,pad_to_max_length=True)
    tr_ids = tokenizer.batch_encode_plus(Triples,max_length = max_triple_token,pad_to_max_length=True,return_offsets_mapping=True)
    r_mask = []
    t_mask = []
    offsets_map = tr_ids['offset_mapping']

    # mask for relation and tail
    for i,triples in enumerate(Triples):
        temp_r_mask = [0]*max_triple_token
        temp_t_mask = [0]*max_triple_token
        form_loc = 0
        split_loc = triples.find('.')
        while split_loc < len(triples)-1 and len(triples[form_loc:split_loc].split('\t')) == 4:
            head,rela,tail,_ = triples[form_loc:split_loc].split('\t')
            rela_start = form_loc+len(head)+1
            tail_start = form_loc+len(head)+1+len(rela)+1
            rs,re,ts,te = -1,-1,-1,-1
            for k,token_pairs in enumerate(offsets_map[i]):
                if rela_start == token_pairs[0]:
                    rs = k
                if rela_start+len(rela) == token_pairs[1]:
                    re = k
                if tail_start == token_pairs[0]:
                    ts = k
                if tail_start+len(tail) == token_pairs[1]:
                    te = k
                if rs != -1 and re != -1 and ts != -1 and te != -1:
                    break
            temp_r_mask[rs:re] = [1]*(re-rs+1)
            temp_t_mask[ts:te] = [1]*(te-ts+1)
            form_loc = split_loc+1
            split_loc = triples[form_loc:].find('.')

        r_mask.append(temp_r_mask)
        t_mask.append(temp_t_mask)

    return qids,aids,tr_ids,r_mask,t_mask

class bert_dataset(Dataset):
    def __init__(self,qids,aids,tr_ids,r_mask,t_mask) -> None:
        super(bert_dataset).__init__()
        self.qids = torch.stack(qids)
        self.aids = torch.stack(aids)
        self.tr_ids = torch.stack(tr_ids)
        self.r_mask = torch.stack(r_mask)
        self.t_mask = torch.stack(t_mask)
    def __len__(self):
        return self.qids.shape[0]
    def __getitem__(self,index):
        return self.qids[index],self.aids[index],self.tr_ids[index],self.r_mask[index],self.t_mask[index]
    


def prepare_data(val_rate = 0.8):
    QandA,Triples = load_data()
    qids,aids,tr_ids,r_mask,t_mask = token_and_mask(QandA,Triples)
    length = int(val_rate*len(qids))

    train_qids , test_qids = qids[:length] , qids[length:]
    train_aids , test_aids = aids[:length] , aids[length:]
    train_tr_ids , test_tr_ids = tr_ids[:length] , tr_ids[length:]
    train_rmask , test_rmask = r_mask[:length] , r_mask[length:]
    train_tmask , test_tmask = t_mask[:length] , t_mask[length:]

    train_dataset = bert_dataset(train_qids,train_aids,train_tr_ids,train_rmask,train_tmask)
    test_dataset = bert_dataset(test_qids,test_aids,test_tr_ids,test_rmask,test_tmask)

    return DataLoader(train_dataset),DataLoader(test_dataset)










