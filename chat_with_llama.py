from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

import torch
cutoff_len = 512

model_path = '/data/llmweights/llama-2-7b-hf'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.padding_side = "left" 
tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )

device = torch.device('cuda:1')
# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda:1",
    torch_dtype='auto'
).to(device).eval()

# Prompt content: "hi"

su_qu = []
fa_qu = []
fail_ans = dict()
path = 'data path'
with open(path,encoding='utf-8') as f:
    datas = json.load(f)
    i=0
    print(len(datas))
    while i < len(datas):
        sub_que = datas[i]
        question = sub_que['machine_question']
        ans = [sub_que['answer']]
        messages = [
            {"role": "user", "content":"Please answer the following questions.Question:{What is the offical language of Taipei?} Answer:{Chinese}. Question:{How many wars did the Empire of Japan participate in?} Answer:{4}. Question:{"+question+"} Answer:"}
        ]

        input_ids = torch.tensor(tokenizer.encode(messages[0]['content'])).to(device).unsqueeze(0)
        # input_id = torch.tensor(input_ids['input_ids']).to(device).unsqueeze(0)
        # attention_mask = torch.tensor(input_ids['attention_mask']).to(device).unsqueeze(0)

        pad_token_id = tokenizer.eos_token_id
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)

        output_ids = model.generate(input_ids.to(device),max_new_tokens=500,attention_mask = attention_mask, pad_token_id=pad_token_id)
        # output_ids = model.generate(input_id,max_new_tokens=500,attention_mask = attention_mask,pad_token_id=pad_token_id)
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        response = response[response.find(question):]
        response = response[:min(len(response),response.find('Answer')+50)]
        for a in ans:
            a = a.lower()
            if a in response.lower():
                su_qu.append(str(i)+' ')
                print('success: ',i)
                break
        if len(su_qu)==0 or su_qu[-1] != str(i)+' ':
            print('fail: ',i)
            fa_qu.append(str(i)+' ')
            fail_ans[i] = response
        i+=1 

print(len(su_qu),len(fa_qu))

f1=open("success_and fail/success_cwq.txt","w")
f1.writelines(su_qu) 
f1.close()
f2=open("success_and fail/fail_cwq.txt","w")
f2.writelines(fa_qu)
f2.close()
with open(f'success_and fail/fail_ans_cwq','w') as f3:
    json.dump(fail_ans,f3)
