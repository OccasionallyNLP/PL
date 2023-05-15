# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Dict, Optional
from transformers import PreTrainedModel
from dataclasses import dataclass
# pair wise
## pointwise - Sigmoid cross entropy loss
## pairwise - Pairwise logistic loss
## listwise - Softmax loss

# TODO
# point wise - regression version
# point wise - classification
class PointWiseRegressionModel(PreTrainedModel):
    def __init__(self, config, pool, model_class):
        super().__init__(config)
        self.pool = pool
        self.pretrained_model = model_class(config) # T5 Enc model
        self.fc = nn.Linear(config.d_model, 1) # changed

    def init_pretrained_model(self, state_dict):
        self.pretrained_model.load_state_dict(state_dict) 
    
    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.pretrained_model(input_ids, attention_mask) 
        if self.pool=='cls':
            # T5 Encoder만 따로 만들어놔야함.
            if isinstance(self.pretrained_model, T5EncoderModel):
                out = output.last_hidden_state[:,0,:] # bs, seq_len, dim
            else:
                out = output['pooler_output'] # bs, dim
        elif self.pool == 'mean':
            out = output['last_hidden_state'].masked_fill(attention_mask.unsqueeze(2).repeat(1,1,self.config.hidden_size)==0,0) # bs, seq_len, dim
            out = out.sum(dim=1) # bs, dim
            s = attention_mask.sum(-1, keepdim=True) # bs, dim
            out = out/(s)
            
        scores = self.fc(out).unsqueeze(1) # bs, 1 -> bs
        if 'labels' in kwargs:
            loss_fn = nn.MSELoss()
            loss = loss_fn(scores, kwargs['labels'].float())
            return dict(loss=loss, score = scores)
        else:
            if n_docs is not None:
                scores = scores.reshape(bs, n_docs)
            return dict(score = scores)


# point wise - classification
class PointWiseModel(PreTrainedModel):
    def __init__(self, config, pool, model_class, n_ranks):
        super().__init__(config)
        self.pool = pool
        self.pretrained_model = model_class(config) # T5 Enc model
        self.fc = nn.Linear(config.d_model, n_ranks) # changed

    def init_pretrained_model(self, state_dict):
        self.pretrained_model.load_state_dict(state_dict) 
    
    def forward(self, input_ids, attention_mask, **kwargs):
        # input_ids - (bs, n_docs, seq_len) -> (bs*n_docs, seq_len)
        # attention_ids - (bs, n_docs, seq_len) -> (bs*n_docs, seq_len)
        n_docs = None
        if input_ids.dim()==3:
            bs, n_docs, seq_len = input_ids.shape
            input_ids = input_ids.reshape(-1, seq_len)
            attention_mask = attention_mask.reshape(-1, seq_len)
        output = self.pretrained_model(input_ids, attention_mask) 
        if self.pool=='cls':
            # T5 Encoder만 따로 만들어놔야함.
            if isinstance(self.pretrained_model, T5EncoderModel):
                out = output.last_hidden_state[:,0,:] # bs, seq_len, dim
            else:
                out = output['pooler_output'] # bs, dim
        elif self.pool == 'mean':
            out = output['last_hidden_state'].masked_fill(attention_mask.unsqueeze(2).repeat(1,1,self.config.hidden_size)==0,0) # bs, seq_len, dim
            out = out.sum(dim=1) # bs, dim
            s = attention_mask.sum(-1, keepdim=True) # bs, dim
            out = out/(s)
            
        scores = self.fc(out) # bs, n_rank
        
        if 'labels' in kwargs:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(scores, kwargs['labels'])
            return dict(loss=loss, score = scores)
        else:
            if n_docs is not None:
                scores = scores.reshape(bs, n_docs)
            return dict(score = scores)
# pair wise
class PairWiseModel(PreTrainedModel):
    def __init__(self, config, pool, model_class):
        super().__init__(config)
        self.pool = pool
        self.pretrained_model = model_class(config)
        self.fc = nn.Linear(config.d_model, 1)

    def init_pretrained_model(self, state_dict):
        self.pretrained_model.load_state_dict(state_dict) 
    
    def forward(self, sentence_1_input_ids, sentence_1_attention_mask, sentence_2_input_ids, sentence_2_attention_mask, **kwargs):
        # (bs, seq_len) -> (bs*n_docs, seq_len)
        s1_embeds = self.pretrained_model(input_ids=sentence_1_input_ids, attention_mask=sentence_1_attention_mask)
        s2_embeds = self.pretrained_model(input_ids=sentence_2_input_ids, attention_mask=sentence_2_attention_mask)
        if self.pool=='cls':
            # T5 Encoder만 따로 만들어놔야함.
            if isinstance(self.pretrained_model, T5EncoderModel):
                s1_rpr = s1_embeds.last_hidden_state[:,0,:] # bs*n_docs, seq_len, dim -> bs*n_docs, dim
                s2_rpr = s2_embeds.last_hidden_state[:,0,:] 
            else:
                s1_rpr = s1_embeds['pooler_output'] # bs, dim
                s2_rpr = s2_embeds['pooler_output'] # bs, dim
        
        elif self.pool == 'mean':
            s1_rpr = s1_embeds['last_hidden_state'].masked_fill(sentence_1_attention_mask.unsqueeze(2).repeat(1,1,self.config.hidden_size)==0,0) # bs, seq_len, dim
            s1_rpr = s1_rpr.sum(dim=1) # bs, dim
            s = sentence_1_attention_mask.sum(-1, keepdim=True) # bs, 1
            s1_rpr = s1_rpr/(s)
            
            s2_rpr = s2_embeds['last_hidden_state'].masked_fill(sentence_2_attention_mask.unsqueeze(2).repeat(1,1,self.config.hidden_size)==0,0) # bs, seq_len, dim
            s2_rpr = s2_rpr.sum(dim=1) # bs, dim
            s = sentence_2_attention_mask.sum(-1, keepdim=True) # bs, 1
            s2_rpr = s2_rpr/(s)
            
        s1_scores = self.fc(s1_rpr).squeeze(-1)
        s2_scores = self.fc(s2_rpr).squeeze(-1) 
        
        if 'labels' in kwargs: # binary label 1, 3, 2
            # bs, n_docs
            labels = kwargs['labels']
            loss_fn = PairWiseLoss()
            # TODO - reassemble labels
            loss = loss_fn(s1_scores, s2_scores)
            return dict(loss = loss, sentence_1_score = s1_scores, sentence_2_score = s2_scores)
        else:
            return dict(sentence_1_score = s1_scores, sentence_2_score = s2_scores)

# list wise
class ListWiseModel(PreTrainedModel):
    def __init__(self, config, pool, model_class):
        super().__init__(config)
        self.pool = pool
        self.pretrained_model = model_class(config)
        self.fc = nn.Linear(config.d_model, 1)

    def init_pretrained_model(self, state_dict):
        self.pretrained_model.load_state_dict(state_dict) 
    
    def forward(self, input_ids, attention_mask, **kwargs):
        # input_ids - (bs*n_docs, seq_len)
        bs, n_docs, seq_len = input_ids.shape
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)
        output = self.pretrained_model(input_ids, attention_mask)
        if self.pool=='cls':
            # T5 Encoder만 따로 만들어놔야함.
            if isinstance(self.pretrained_model, T5EncoderModel):
                out = output.last_hidden_state[:,0,:] # bs*n_docs, seq_len, dim -> bs*n_docs, dim
            else:
                out = output['pooler_output'] # bs, dim
        elif self.pool == 'mean':
            out = output['last_hidden_state'].masked_fill(attention_mask.unsqueeze(2).repeat(1,1,self.config.hidden_size)==0,0) # bs, seq_len, dim
            out = out.sum(dim=1) # bs, dim
            s = attention_mask.sum(-1, keepdim=True) # bs, 1
            out = out/(s)
            
        # out ~ (bs*n_docs, dim) 
        out = out.reshape(bs, n_docs, -1) # bs, n_docs, dim
        scores = self.fc(out).squeeze(-1) # bs, n_docs, dim -> bs, n_docs, 1 -> bs, n_docs
        
        if 'labels' in kwargs: # binary label 1, 3, 2
            # bs, n_docs
            labels = kwargs['labels']
            loss = -((F.log_softmax(scores,dim=-1)*labels).sum())
            return dict(loss = loss, score = scores)
        else:
            return dict(score = scores)

        
# pairwise 
class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """
    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        return loss
    
