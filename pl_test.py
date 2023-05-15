# -*- coding: utf-8 -*-
import os
import json
import pickle
import copy
from tqdm import tqdm
import logging
import time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from transformers import AutoTokenizer, T5EncoderModel, T5Config, T5ForConditionalGeneration, get_linear_schedule_with_warmup, AdamW
import argparse
from utils.data_utils import *
from utils.distributed_utils import *
from utils.utils import *
from utils.metrics import *
from model import *
from utils.evaluation import *
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning.pytorch as pl

def get_args():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, help = 'test_name')
    parser.add_argument('--output_dir', type=str, help = 'output 위치', default = '.')
    # data
    parser.add_argument('--train_data', type=str, help = 'train_data 위치', default = '../data/RM/dev.jsonl')
    parser.add_argument('--val_data', type=str, help='val data 위치', default = '../data/RM/dev.jsonl')
    
    parser.add_argument('--n_ranks', type=int, default = 3)
    parser.add_argument('--rank_type', type=str, default = 'point', choices=['point','regression','pair','list'])
    parser.add_argument('--eval_rank_type', type=str, default = 'point', choices=['point','regression','pair','list'])
    
    # logging 관련
    parser.add_argument('--logging_term', type=int, default = 100)
   
    # 학습 관련
    parser.add_argument('--epochs', default = 10, type=int)
    parser.add_argument('--eval_epoch', type = int, default = 1, help = 'term of evaluation')
    parser.add_argument('--batch_size', default = 2, type=int)
    parser.add_argument('--lr', type=float, default = 5e-5)
    parser.add_argument('--warmup', type=float, default = 1000)
    parser.add_argument('--decay', type=float, default = 0.05)
    parser.add_argument('--fp16', type=str2bool, default = False)
    parser.add_argument('--accumulation_steps', type=int, default = 1) # 221124 추가
    parser.add_argument('--weighted_sampling', type=str2bool, default = False) # 221124 추가
    
    # PTM model
    parser.add_argument('--ptm_path', type=str, default = 'KETI-AIR/ke-t5-small')
    parser.add_argument('--model_path', type=str)
    
    # model input
    parser.add_argument('--max_length', type=int, default = 512)
    
    # distributed 관련
    parser.add_argument('--local_rank', type=int, default = 0)
    parser.add_argument('--distributed', type=str2bool, default = False)
    parser.add_argument('--early_stop', type=str2bool, default = True) # XXX220919
    parser.add_argument('--patience', type=int, default = 3)
    args,_  = parser.parse_known_args()
    return args

def get_tokenizer_and_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.ptm_path)
    config = T5Config.from_pretrained(args.ptm_path)
    if args.rank_type == 'point':
        model = PointWiseModel(config, 'mean', T5EncoderModel, args.n_ranks)
    elif args.rank_type == 'list':
        model = ListWiseModel(config, 'mean', T5EncoderModel)
    elif args.rank_type == 'pair':
        model = PairWiseModel(config, 'mean', T5EncoderModel)
    elif args.rank_type == 'regression':
        model = PointWiseRegressionModel(config, 'mean', T5EncoderModel)
        
    if args.model_path is None:
        t5 = T5EncoderModel.from_pretrained(args.ptm_path)
        model.init_pretrained_model(t5.state_dict())
    else:
        model_state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(model_state_dict)
    return tokenizer, model 

def load_datasets(args, tokenizer):
    # LOAD DATASETS
    train_data = load_jsonl(args.train_data)[:100]
    if args.rank_type == 'point':
        train_dataset = PointWiseDataset(train_data, tokenizer, args.max_length)
    elif args.rank_type == 'regression':
        train_dataset = PointWiseDataset(train_data, tokenizer, args.max_length)    
    elif args.rank_type == 'list':
        train_dataset = ListWiseDataset(train_data, tokenizer, args.max_length)
    elif args.rank_type == 'pair':
        train_dataset = PairWiseDataset(train_data, tokenizer, args.max_length)
    
    if args.distributed:
        # OK - legacy
        val_data = load_data(args.val_data, args.local_rank, args.distributed)[:100]
    else:
        val_data = load_jsonl(args.val_data)[:100]
    
    if args.eval_rank_type == 'point':
        val_dataset = PointWiseDataset(val_data, tokenizer, args.max_length)
    elif args.eval_rank_type == 'regression':
        val_dataset = PointWiseDataset(val_data, tokenizer, args.max_length)
    elif args.eval_rank_type == 'list':
        val_dataset = ListWiseDataset(val_data, tokenizer, args.max_length)
    elif args.eval_rank_type == 'pair':
        val_dataset = PairWiseDataset(val_data, tokenizer, args.max_length)
    return train_dataset, val_dataset
        



# define the LightningModule
class PLModel(pl.LightningModule):
    def __init__(self, args, num_training_steps, logger1, logger2, model):
        super().__init__()
        self.args = args
        self.num_training_steps = num_training_steps
        self.logger1 = logger1
        self.logger2 = logger2
        self.model = model
        self.save_hyperparameters(ignore=['model'])
        self.validation_step_outputs = []
        
    def forward(self, batch):
        out = self.model(**batch)
        return out['score']
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
#        print(batch_idx)
        data = {i:j.cuda() for i,j in batch.items() if i!='labels'}
        labels = batch['labels']
        output = self.model(**batch)['score']
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, labels)
        return loss
        
    def configure_optimizers(self):
        # TODO scheduler
        param_optimizer = list(self.model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
        'params': [
            p for n, p in param_optimizer
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        self.hparams.args.decay
          }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
        }]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.args.lr, weight_decay=self.hparams.args.decay)
        scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=self.hparams.args.warmup,
        num_training_steps=self.num_training_steps
    )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        data = {i:j.cuda() for i,j in batch.items() if i!='labels'}
        labels = batch['labels']
        output = self.model(**batch)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output['score'], labels)
        output = {"loss": loss, "score": output['score'], "actual":labels.cpu().tolist()}
        self.validation_step_outputs.append(output)
        
        # print({"val_loss": loss})
        #self.log('val_loss', loss)
        # return {"loss": val_loss, "preds": preds, "labels": labels}
        return output
        
    def on_validation_epoch_end(self):
        #print(len(self.validation_step_outputs))
        all_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        actuals = [x['actual'] for x in self.validation_step_outputs]
        # score customizing
        score = torch.stack([x['score'] for x in self.validation_step_outputs])
        predict = score.argmax(dim=-1).tolist()
        acc = []
        for i,j in zip(predict, actuals):
            acc.append(i==j)
        acc = sum(acc)/len(acc)
        # do something with all preds
        self.log("val_loss", all_loss, prog_bar=True)
        self.log("acc", acc, prog_bar=True)
        
        self.logger1.info(f'{self.current_epoch} -- loss - {all_loss.item()}')
        self.logger1.info(f'{self.current_epoch} -- acc - {acc}')
        self.logger2.info(f'{self.current_epoch} -- loss - {all_loss.item()}')
        self.logger2.info(f'{self.current_epoch} -- acc - {acc}')
        self.validation_step_outputs.clear()  # free memory
        
if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output_dir, exist_ok = True)
    if args.local_rank in [-1,0]:
        with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    logger1, logger2 = get_log(args)
    if args.local_rank in [-1,0]:
        logger1.info(args)
        logger2.info(args)

    tokenizer, model = get_tokenizer_and_model(args)
    train_dataset, val_dataset = load_datasets(args, tokenizer)
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset) 
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,batch_size = args.batch_size, sampler = train_sampler, collate_fn = train_dataset.collate_fn)
    num_training_steps = len(train_dataloader)*args.epochs
    val_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, sampler = SequentialSampler(val_dataset), collate_fn = val_dataset.collate_fn)
    
    pl_model = PLModel(args, num_training_steps, logger1, logger2, model)
    # Trainer(accelerator="gpu", devices=8, strategy="ddp")
    # TODO
    if args.distributed:
        pass
    else:
        trainer = pl.Trainer(accelerator = 'gpu', precision=16 if args.fp16 else 32, max_epochs = args.epochs, accumulate_grad_batches=args.accumulation_steps,\
                        default_root_dir=args.output_dir, gradient_clip_val=1., log_every_n_steps=args.logging_term,check_val_every_n_epoch = args.eval_epoch, num_nodes=1, enable_progress_bar=True, callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=args.patience)])
        
    trainer.fit(model=pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    if args.local_rank in [-1,0]:
        with open(os.path.join(args.output_dir,'best_model_path'), 'w') as f:
            json.dump(trainer.checkpoint_callback.best_model_path,f)