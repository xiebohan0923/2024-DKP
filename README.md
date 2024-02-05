# Paper
DKP-RAG: ADifficulty-Assessed Knowledge Path Retrieval-Augmented Generation Method
# Introduce
> Large language models have achieved remarkable performance across various tasks, but they often suffer from hallucination issues,
especially in scenarios requiring specific domain knowledge and the latest information. These problems can be addressed using the
Retrieval-Augmented Generation method, but blindly introducing information may degrade the quality of responses. In this paper, we
propose a novel difficulty-assessed graph retrieval-augmented generation method that distinguishes tasks based on response difficulty, selectively retrieving external knowledge paths only for complex tasks. We design an effective path filtering method that evaluates path scores from the perspectives of relevance and scarcity, forming a knowledge subgraph. We employ a multi-heterogeneous graph structure adapter to capture neighborhood structural information of the subgraph and project it into the text representation space as a prefix for input prompts.

# Model Architecture
![Model_architecture](figures/model.pdf)

#  Dependencies
Our code is developed based on [alpaca-lora](https://github.com/tloen/alpaca-lora). Please build the Python environment following the instruction in Alpaca-lora.


# Run Model

## prepare data
### step1 
run chat_with llama.py to assess the difficulty of queries

### step2
run preprocess/entity_search.py to get extra information from wikidata
run preprocess/filter_triples.py to filter the paths

### step3
run pre_gnn_data.py to get knowledge subgraphs

## run DKP-RAG tuning
```shell
export WANDB_DISABLED=true
wandb offline
CUDA_VISIBLE_DEVICES=0 nohup python finetune_model.py \
    --base_model 'YOUR LLM PATH' \
    --data_path ' YOUR DATA PATHS' \
    --output_dir 'YOUR SAVE PATH' \
    --data_set 'qald' \ four choices: 'web','cwq','qald','creak'
    --num_epochs 1 \
    --lora_r 16 \
    --learning_rate 3e-4 \
    --batch_size 1 \
    --micro_batch_size 1 \
    --num_prefix 2 \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' > log.txt &
```
You may need to fill the LLM path and save path before running.

## run inference
```shell
CUDA_VISIBLE_DEVICES=0 python inference.py
```
