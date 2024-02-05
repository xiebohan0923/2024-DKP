import dgl
import json
from dgl import load_graphs
from dgl.nn.pytorch import HeteroGraphConv,GraphConv
import torch
import torch.nn as nn
from typing import Optional, List
import os
from transformers import LlamaForCausalLM
from dgl.data import DGLDataset


device = torch.device('cuda:0')

class MyDataset(DGLDataset):
    def __init__(self,data_path1 = '',data_path2 = '',
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(MyDataset, self).__init__(name='dataset_name',
                                        url = url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
        self.graphs = []
        for path in os.listdir(data_path1):
            self.graphs.append(load_graphs(os.path.join(data_path1,path))[0][0].to(device))
        for path in os.listdir(data_path2):
            self.graphs.append(load_graphs(os.path.join(data_path2,path))[0][0].to(device))

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size,data):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        with open(data+'_rela_dict') as file:  # your relation dict
            self.rela_dict = json.load(file)

        self.rgcn = HeteroGraphConv(
            { str(name) : GraphConv(in_size, out_size) for name in self.rela_dict.values()},aggregate='mean')

    def forward(self, g, node_embedding,rela_weight):
        res = self.rgcn(g,{'entity':node_embedding},mod_kwargs=rela_weight)

        return res['entity']

class HeteroRGCN(nn.Module):
    def __init__(self, in_size, out_size,data,hidden_size = 256):
        super(HeteroRGCN, self).__init__()

        self.layer1 = HeteroRGCNLayer(hidden_size, hidden_size,data)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size,data)

        self.node_fc = nn.Linear(in_size,hidden_size)
        self.rela_fc = nn.Linear(in_size,hidden_size)


        # self.act = nn.ReLU()
    def forward(self, G):
        rela_weight = {}
        for stype, etype, dtype in G.canonical_etypes:
            # rela_weight[etype] = {'edge_weight': self.rela_embedding[int(etype)].unsqueeze(0).repeat(G[etype].num_edges(),1)}
            rela_weight[etype] = {'edge_weight': self.rela_fc(G[etype].edata['weight'])}

        emb0 = self.layer1(G, self.node_fc(G.ndata['feat']) ,rela_weight)
        # emb1 = F.elu(emb0)
        # emb1 = self.act(emb0)
        # emb1 = self.layer1_1(G, emb0,rela_weight)
        emb2 = self.layer2(G, emb0,rela_weight)
        return emb2
    

class GCNWithAdapter(nn.Module):
    def __init__(
        self,
        model: LlamaForCausalLM,
        embed_dim = 768,
        data = str
    ) -> None:
        super(GCNWithAdapter, self).__init__()
        self.llama_model = model
        self.device = model.device
        self.embed_dim = embed_dim
        self.gcn_embedding = HeteroRGCN(embed_dim,embed_dim,data).to(self.device)
        self.node_fc = nn.Linear(768,4096) 
        self.set = data
        # self.gcn_embedding = torch.load('my_kg_embedding.pth',map_location='cuda:0')
        # self.gcn_embedding.requires_grad = False
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ids: torch.LongTensor = None,
    ):
  
        if os.path.exists(self.set+'_graph/new_q'+str(ids.item())+'.bin'):  #your graph data path
            g = load_graphs(self.set+'_graph/new_q'+str(ids.item())+'.bin')[0][0].to(self.device)
            en_embed = self.node_fc(self.gcn_embedding(g)[0].unsqueeze(0).unsqueeze(0))
            en_embed2 = self.node_fc(self.gcn_embedding(g)[-1].unsqueeze(0).unsqueeze(0))
            
            rela_en = torch.zeros([1,768]).to(self.device)
            count = 0
            for stype, etype, dtype in g.canonical_etypes:
            # rela_weight[etype] = {'edge_weight': self.rela_embedding[int(etype)].unsqueeze(0).repeat(G[etype].num_edges(),1)}
                rela_en += torch.mean(g[etype].edata['weight'],dim = 0).unsqueeze(0)
                count+=1
            # rela_en = self.node_fc((rela_en/count).unsqueeze(0))
            # en_embed = torch.cat((en_embed, rela_en), dim=1)
            en_embed = torch.cat((en_embed, en_embed2), dim=1)
        else:
            en_embed = torch.ones([1,2,4096]).to(self.device)

        batch_size, seq_len, _ = en_embed.shape
        token_embeds = self.llama_model.model.model.embed_tokens(input_ids)

        input_embeds = torch.cat((en_embed, token_embeds), dim=1)
        prefix_mask = torch.ones((batch_size, seq_len))
        prefix_labels = torch.full((batch_size, seq_len), fill_value=-100, dtype=torch.long)
        new_attention_mask = torch.cat((prefix_mask.cuda(), attention_mask), dim=-1)
        new_labels = torch.cat((prefix_labels.cuda(), labels), dim=-1)
        return self.llama_model(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    