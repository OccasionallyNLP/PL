# -*- coding: utf-8 -*-
# data_utils
import json
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
import random
import copy
from dataclasses import dataclass
from transformers import AutoTokenizer
from itertools import combinations
 
@dataclass
class PointWiseDataset(Dataset):
    data:List[dict]
    tokenizer:AutoTokenizer
    max_length:Optional[int]=None
    def __getitem__(self, index):
        '''
        {'prompt': '번디는 자신이 탐정잡지, 범죄소설 그리고 성범죄 관련 실제 범죄 다큐멘터리들을 탐독했다고 누구에게 말했나?',
         'ranking': [2, 1, 0],
         'completion': ['Allow me to answer your question. I know that you are curious about me.',
          '번디는 다양한 인터뷰자들과 뉴스홍보 담당자들과의 면담 때 밝혔다.',
          '라이언에게 말했다.']}
        '''
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        inputs = []
        labels = []
        for b in batch:
            prompt = b['prompt']
            inputs.extend([prompt+' '+completion for completion in b['completion']])
            if b.get('ranking'):
                labels.extend(b['ranking'])            
        if self.max_length is None:
            inputs = self.tokenizer(inputs, padding='longest',return_tensors = 'pt')
        else:
            inputs = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
        if labels:
            inputs.data['labels']=torch.tensor(labels)
        return inputs

@dataclass
class PairWiseDataset(Dataset):
    data:List[dict]
    tokenizer:AutoTokenizer
    max_length:Optional[int]=None
    def __getitem__(self, index):
        # rank가 낮을수록 goo
        '''
        {'prompt': '번디는 자신이 탐정잡지, 범죄소설 그리고 성범죄 관련 실제 범죄 다큐멘터리들을 탐독했다고 누구에게 말했나?',
         'ranking': [2, 1, 0],
         'completion': ['Allow me to answer your question. I know that you are curious about me.',
          '번디는 다양한 인터뷰자들과 뉴스홍보 담당자들과의 면담 때 밝혔다.',
          '라이언에게 말했다.']}
        '''
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        s1_list = []
        s2_list = []
        labels = []
        for b in batch:
            prompt = b['prompt']
            completion = b['completion']
            if b.get('ranking'):
                ranks = b['ranking']
                comb = list(combinations(ranks,2))
                for i,j in comb:
                    if i>j:
                        m, M = j, i
                    else:
                        m, M = i, j
                    s1 = prompt + ' ' + completion[m]
                    s2 = prompt + ' ' + completion[M]
                    s1_list.append(s1)
                    s2_list.append(s2)
                    labels.append(0)
            else:
                comb = list(combinations(completion,2))
                for i,j in comb:
                    s1 = prompt + ' ' + i
                    s2 = prompt + ' ' + j
                    s1_list.append(s1)
                    s2_list.append(s2)

        if self.max_length is None:
            s1_input = self.tokenizer(s1_list, padding='longest',return_tensors = 'pt')
            s2_input = self.tokenizer(s2_list, padding='longest',return_tensors = 'pt')
        else:
            s1_input = self.tokenizer(s1_list, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
            s2_input = self.tokenizer(s2_list, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
        output = dict(sentence_1_input_ids=s1_input.input_ids, sentence_1_attention_mask=s1_input.attention_mask, 
                     sentence_2_input_ids=s2_input.input_ids,
                     sentence_2_attention_mask=s2_input.attention_mask)
        if labels:
            output['labels']=torch.tensor(labels)
        return output
    
@dataclass 
class ListWiseDataset(Dataset):
    data:List[dict]
    tokenizer:AutoTokenizer
    max_length:Optional[int]=None
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        '''
        {'prompt': '번디는 자신이 탐정잡지, 범죄소설 그리고 성범죄 관련 실제 범죄 다큐멘터리들을 탐독했다고 누구에게 말했나?',
         'ranking': [2, 1, 0],
         'completion': ['Allow me to answer your question. I know that you are curious about me.',
          '번디는 다양한 인터뷰자들과 뉴스홍보 담당자들과의 면담 때 밝혔다.',
          '라이언에게 말했다.']}
        '''
        inputs = []
        labels = []
        bs = len(batch)
        for b in batch:
            n_samples = len(b['completion'])
            prompt = b['prompt']
            inputs.extend([prompt+' '+completion for completion in b['completion']])
            if b.get('ranking'):
                labels.append(b['ranking'])
        if self.max_length is None:
            inputs = self.tokenizer(inputs, padding='longest',return_tensors = 'pt')
        else:
            inputs = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
        inputs.data['input_ids'] = inputs.data['input_ids'].reshape(bs, n_samples, -1) 
        inputs.data['attention_mask'] = inputs.data['attention_mask'].reshape(bs, n_samples, -1)
        if labels:
            inputs.data['labels'] = torch.tensor(labels)
        return inputs
###################################################################################################################
# encoder_decoder
###################################################################################################################
@dataclass
class T5PointWiseDataset(Dataset):
    data:List[dict]
    tokenizer:AutoTokenizer
    max_length:Optional[int]=None
    def __getitem__(self, index):
        '''
        {'prompt': '번디는 자신이 탐정잡지, 범죄소설 그리고 성범죄 관련 실제 범죄 다큐멘터리들을 탐독했다고 누구에게 말했나?',
         'ranking': [2, 1, 0],
         'completion': ['Allow me to answer your question. I know that you are curious about me.',
          '번디는 다양한 인터뷰자들과 뉴스홍보 담당자들과의 면담 때 밝혔다.',
          '라이언에게 말했다.']}
        '''
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        inputs = []
        #decoder_inputs = []
        labels = []
        for b in batch:
            prompt = b['prompt']
            inputs.extend([prompt+' '+completion for completion in b['completion']])
            if b.get('ranking'):
                labels.extend([str(j) for j in b['ranking']])            
        if self.max_length is None:
            inputs = self.tokenizer(inputs, padding='longest',return_tensors = 'pt')
        else:
            inputs = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
        if labels:
            inputs.data['labels']=self.tokenizer(labels, padding=True, return_tensors = 'pt').input_ids
        return inputs

@dataclass
class T5PairWiseDataset(Dataset):
    data:List[dict]
    tokenizer:AutoTokenizer
    max_length:Optional[int]=None
    def __getitem__(self, index):
        # rank가 낮을수록 goo
        '''
        {'prompt': '번디는 자신이 탐정잡지, 범죄소설 그리고 성범죄 관련 실제 범죄 다큐멘터리들을 탐독했다고 누구에게 말했나?',
         'ranking': [2, 1, 0],
         'completion': ['Allow me to answer your question. I know that you are curious about me.',
          '번디는 다양한 인터뷰자들과 뉴스홍보 담당자들과의 면담 때 밝혔다.',
          '라이언에게 말했다.']}
        '''
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        s1_list = []
        s2_list = []
        labels = []
        for b in batch:
            prompt = b['prompt']
            completion = b['completion']
            if b.get('ranking'):
                ranks = b['ranking']
                comb = list(combinations(ranks,2))
                for i,j in comb:
                    if i>j:
                        m, M = j, i
                    else:
                        m, M = i, j
                    s1 = prompt + ' ' + completion[m]
                    s2 = prompt + ' ' + completion[M]
                    s1_list.append(s1)
                    s2_list.append(s2)
                    labels.append(0)
            else:
                comb = list(combinations(completion,2))
                for i,j in comb:
                    s1 = prompt + ' ' + i
                    s2 = prompt + ' ' + j
                    s1_list.append(s1)
                    s2_list.append(s2)

        if self.max_length is None:
            s1_input = self.tokenizer(s1_list, padding='longest',return_tensors = 'pt')
            s2_input = self.tokenizer(s2_list, padding='longest',return_tensors = 'pt')
        else:
            s1_input = self.tokenizer(s1_list, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
            s2_input = self.tokenizer(s2_list, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
        output = dict(sentence_1_input_ids=s1_input.input_ids, sentence_1_attention_mask=s1_input.attention_mask, 
                     sentence_2_input_ids=s2_input.input_ids,
                     sentence_2_attention_mask=s2_input.attention_mask)
        if labels:
            output['labels']=torch.tensor(labels)
        return output
@dataclass 
class ListWiseDataset(Dataset):
    data:List[dict]
    tokenizer:AutoTokenizer
    max_length:Optional[int]=None
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        '''
        {'prompt': '번디는 자신이 탐정잡지, 범죄소설 그리고 성범죄 관련 실제 범죄 다큐멘터리들을 탐독했다고 누구에게 말했나?',
         'ranking': [2, 1, 0],
         'completion': ['Allow me to answer your question. I know that you are curious about me.',
          '번디는 다양한 인터뷰자들과 뉴스홍보 담당자들과의 면담 때 밝혔다.',
          '라이언에게 말했다.']}
        '''
        inputs = []
        labels = []
        bs = len(batch)
        for b in batch:
            n_samples = len(b['completion'])
            prompt = b['prompt']
            inputs.extend([prompt+' '+completion for completion in b['completion']])
            if b.get('ranking'):
                labels.append(b['ranking'])
        if self.max_length is None:
            inputs = self.tokenizer(inputs, padding='longest',return_tensors = 'pt')
        else:
            inputs = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_length, return_tensors = 'pt')
        inputs.data['input_ids'] = inputs.data['input_ids'].reshape(bs, n_samples, -1) 
        inputs.data['attention_mask'] = inputs.data['attention_mask'].reshape(bs, n_samples, -1)
        if labels:
            inputs.data['labels'] = torch.tensor(labels)
        return inputs