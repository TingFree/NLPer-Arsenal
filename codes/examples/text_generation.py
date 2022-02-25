r"""
文本生成任务简易教程
"""
import os
import sys
sys.path.extend(['..', '../..'])

import torch
import argparse
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AdamW, get_linear_schedule_with_warmup,
    BertConfig, BertTokenizer, BertModel
)
from prettytable import PrettyTable
from codes.nlper.modules.trainer import Trainer
from codes.nlper.modules.metrics import Metrics, RougeMetric
from codes.nlper.modules.decoders import TransformerDecoder
from codes.nlper.utils.datasets import DatasetGen
from codes.nlper.utils.format_convert import dureaderqg_convert
from codes.nlper.utils import seed_everything, set_devices, Dict2Obj, dataset_names


class Roberta2Transformer(nn.Module):
    def __init__(self, args):
        super(Roberta2Transformer, self).__init__()
        self.max_tgt_len = args.max_tgt_len
        roberta_config = BertConfig.from_pretrained(args.pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        self.encoder = BertModel.from_pretrained(args.pretrained_model)
        self.decoder = TransformerDecoder(d_model=roberta_config.hidden_size, num_decoder_layers=3, dim_feedforward=128)
        self.fc_out = nn.Linear(roberta_config.hidden_size, roberta_config.vocab_size)

    def forward(self, encoded_src, encoded_tgt=None):
        """teacher-forcing

        :param encoded_src: {'input_ids':[batch_size, src_len], 'token_type_ids':[batch_size, src_len],'attention_mask':[batch_size, src_len]}
        :param encoded_tgt: 和encoded_src类似
        :return:
        """
        if not encoded_tgt:
            return self.predict(encoded_src, self.max_tgt_len)
        src_input_ids, src_token_type_ids, src_attention_mask = encoded_src['input_ids'], \
                                                                encoded_src['token_type_ids'], \
                                                                encoded_src['attention_mask']  # 0:mask
        tgt_input_ids, tgt_token_type_ids, tgt_attention_mask = encoded_tgt['input_ids'], \
                                                                encoded_tgt['token_type_ids'], \
                                                                encoded_tgt['attention_mask']  # 0:mask
        # 剔除tgt中的结束符
        batch_size = src_input_ids.size()[0]
        device = src_input_ids.device
        end_index = np.argwhere(tgt_input_ids.cpu().numpy()==self.tokenizer.sep_token_id)[:,1]
        for i in range(len(end_index)):
            end_index[i] += self.max_tgt_len*i
        tgt_input_ids = torch.from_numpy(np.delete(tgt_input_ids.view(-1).cpu().numpy(), end_index)).reshape(batch_size, -1).to(device)
        tgt_token_type_ids = torch.from_numpy(np.delete(tgt_token_type_ids.view(-1).cpu().numpy(), end_index)).reshape(batch_size, -1).to(device)
        tgt_attention_mask = torch.from_numpy(np.delete(tgt_attention_mask.view(-1).cpu().numpy(), end_index)).reshape(batch_size, -1).to(device)

        encode_outputs = self.encoder(src_input_ids, src_attention_mask, src_token_type_ids)
        memory = encode_outputs.last_hidden_state
        # [batch_size, tgt_len, dim]
        embed_tgt = self.encoder.embeddings(input_ids=tgt_input_ids)
        decode_output = self.decoder(embed_tgt, memory, src_attention_mask==0)
        final = decode_output.last_hidden_state
        # [batch_size, tgt_len, voc_size]
        return self.fc_out(final)

    def predict(self, encoded_src, max_len=32):
        batch_size = len(encoded_src['input_ids'])
        device = self.encoder.device
        # 初始化输入[CLS], [batch_size, 1]
        tgt_ids = torch.tensor([[self.tokenizer.cls_token_id] for ex in range(batch_size)], device=device)
        tgt_ids_probs = None
        cur_len = 1
        # get memory
        encode_outputs = self.encoder(encoded_src['input_ids'], encoded_src['attention_mask'], encoded_src['token_type_ids'])
        memory = encode_outputs.last_hidden_state
        while cur_len < max_len:
            encoded_tgt = {
                'input_ids': tgt_ids,
                'token_type_ids': torch.tensor([[0] * cur_len for ex in range(batch_size)], device=device),
                'attention_mask': torch.tensor([[1] * cur_len for ex in range(batch_size)], device=device)
            }
            # [batch_size, cur_len, dim]
            embed_tgt = self.encoder.embeddings(input_ids=encoded_tgt['input_ids'])
            decode_output = self.decoder(embed_tgt, memory, encoded_src['attention_mask']==0)
            # 将生成的结果添加到下一次输入中
            # [batch_size, cur_len, voc_size]
            final = self.fc_out(decode_output.last_hidden_state)
            if tgt_ids_probs == None:
                tgt_ids_probs = final
            else:
                tgt_ids_probs = torch.cat([tgt_ids_probs, final[:,-1].unsqueeze(1)], dim=1)
            tgt_ids = torch.cat([tgt_ids, final[:,-1].argmax(dim=1).unsqueeze(1)], dim=1)
            cur_len += 1
        # [batch_size, max_len-1, voc_size]
        return tgt_ids_probs


class DataCollatorWithPadding():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # batch: list of ['encoded_src':xxx, 'encoded_tgt':xxx] or ['encoded_tgt':xxx], which depends on load_label
        with_tgt = len(batch[0]) == 2
        src = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        tgt = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        for example in batch:
            ex_src = example['encoded_src']
            for k, v in ex_src.items():
                src[k].append(v)
            if with_tgt:
                ex_tgt = example['encoded_tgt']
                for k, v in ex_tgt.items():
                    tgt[k].append(v)
        batch_src = self.tokenizer.pad(
            src,
            padding=True,
            return_tensors='pt'
        )
        if with_tgt:
            batch_tgt = self.tokenizer.pad(
                tgt,
                padding=True,
                return_tensors='pt'
            )
            return {
                'encoded_src': batch_src.data,  # dict
                'encoded_tgt': batch_tgt.data
            }
        else:
            return {
                'encoded_src': batch_src.data
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('text_gen')
    # general
    parser.add_argument('--task_name', default='text_gen', help='for train, do not change it')
    parser.add_argument('--device_ids', default=[0], help='[-1]:cpu')
    parser.add_argument('--seed', default=1000)
    # dataset
    parser.add_argument('--dataset_name', default='text_gen/DuReaderQG', help="if train/val/test can't be found, automatic download")
    parser.add_argument('--dataset_cache_dir', default='../data')
    parser.add_argument('--train_file', default='DuReaderQG/train.json', help='relative path for dataset_cache_dir')
    parser.add_argument('--val_file', default='DuReaderQG/val.json', help='relative path for dataset_cache_dir')
    parser.add_argument('--test_file', default='DuReaderQG/test.json', help='relative path for dataset_cache_dir')
    parser.add_argument('--max_src_len', default=512, help='max length to truncate for src')
    parser.add_argument('--max_tgt_len', default=32, help='max length to truncate for tgt')
    # train
    parser.add_argument('--is_train', default=True)
    parser.add_argument('--is_test', default=True)
    parser.add_argument('--pretrained_model', default='hfl/chinese-roberta-wwm-ext', help='the name or path of pretrained language model')
    parser.add_argument('--checkpoint', default='', help='continue to train from checkpoint')
    parser.add_argument('--num_epochs', default=5)
    parser.add_argument('--train_batch_size', default=8)
    parser.add_argument('--test_batch_size', default=16)
    parser.add_argument('--lr', default=7e-5)
    parser.add_argument('--decoder_lr', default=3e-4)
    parser.add_argument('--warmup_steps', default=0, help="the number of steps to warm up optimizer")
    parser.add_argument('--weight_decay', default=0.01)
    # save
    parser.add_argument('--best_model_path', default='saved/model.bin', help='the path to save model with the highest performance')
    parser.add_argument('--pred_saved', default='saved/pred.txt', help='the path to save prediction of test')

    # print arguments
    args = parser.parse_args()
    seed_everything(args.seed)
    device = set_devices(args.device_ids)
    table = PrettyTable(['Param', 'Value'])
    for item in vars(args):
        table.add_row([item, vars(args)[item]])
    table.align['Param'] = 'l'
    table.align['Value'] = 'l'
    print(table)

    # load data
    print('load data')
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    args.train_file = os.path.join(args.dataset_cache_dir, args.train_file)
    args.val_file = os.path.join(args.dataset_cache_dir, args.val_file)
    args.test_file = os.path.join(args.dataset_cache_dir, args.test_file)
    # 如果三者同时不存在（适用于第一次运行该代码）
    if not (
            os.path.isfile(args.train_file)
            or os.path.isfile(args.val_file)
            or os.path.isfile(args.test_file)
    ):
        # 自动下载数据集
        corpus = dataset_names[args.dataset_name](cache_dir=args.dataset_cache_dir)
        is_over = corpus.prepare_data()
        if not is_over:
            print(f'please download dataset manually, and make sure data file path is correct')
            exit()
    train_data = dureaderqg_convert(args.train_file)
    val_data = dureaderqg_convert(args.val_file)
    test_data = dureaderqg_convert(args.test_file)
    print(f'train length: {len(train_data)}')
    print(f'val length: {len(val_data)}')
    print(f'test length: {len(test_data)}')

    # padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # dataset
    train_dataset = DatasetGen(train_data, tokenizer, args.max_src_len, args.max_tgt_len)
    val_dataset = DatasetGen(val_data, tokenizer, args.max_src_len, args.max_tgt_len)
    test_dataset = DatasetGen(test_data, tokenizer, args.max_src_len, args.max_tgt_len)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=data_collator)
    print(f'number of train batch: {len(train_loader)}')
    print(f'number of val batch: {len(val_loader)}')
    print(f'number of test batch: {len(test_loader)}')
    print('----------------------------------')

    # create model
    print('create model')
    model = Roberta2Transformer(args).to(device)
    # print(model)
    print('----------------------------------')

    # create optimizer
    specials = ['decoder', 'fc_out']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in specials)],
         'weight_decay': args.weight_decay, 'lr': args.decoder_lr},  # decoder 和 fc_output
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in specials)],
         'weight_decay': args.weight_decay, 'lr': args.lr}  # encoder
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        args.warmup_steps,
        args.num_epochs * len(train_loader)
    )

    # create evaluation metric
    rougeL_metric = RougeMetric('rouge-l')
    metrics = Metrics(
        metrics={rougeL_metric.name: rougeL_metric},
        target_metric=rougeL_metric.name  # used for early stop and model saving
    )

    # load config
    trainer_config = Dict2Obj(vars(args))
    trainer_config.update({
        'tokenizer': tokenizer,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'device': device,
        'loss_fn': F.cross_entropy,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'metrics': metrics
    })

    # running
    trainer = Trainer(model, trainer_config)
    if args.is_train:
        trainer.train()
    if args.is_test:
        trainer.test()
