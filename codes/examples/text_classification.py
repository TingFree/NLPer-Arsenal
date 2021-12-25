r"""
文本分类任务之多分类教程，全流程直观展示，适合NLP入门人员
"""
import os.path
import sys
sys.path.append('..')

import argparse
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from prettytable import PrettyTable
from transformers import get_linear_schedule_with_warmup
from transformers import BertModel, BertTokenizer, DataCollatorWithPadding
from codes.nlper.modules.mlp import MLP
from codes.nlper.utils import DatasetCLF
from codes.nlper.modules.trainer import Trainer
from codes.nlper.modules.metrics import Metrics, PMetric, RMetric, F1Metric
from codes.nlper.utils.format_convert import smp2020_ewect_convert
from codes.nlper.utils import seed_everything, set_devices, Dict2Obj, download_dataset


class BertCLF(nn.Module):
    def __init__(self, args):
        super(BertCLF, self).__init__()
        self.bert = BertModel.from_pretrained(args.pretrained_model)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.clf = MLP([self.bert.config.hidden_size, args.num_class],
                       'tanh',
                       dropout=args.dropout)

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.clf(outputs[1])
        return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--pretrained_model',
                        default='bert-base-chinese',
                        help='the name or path of pretrained language model')
    parser.add_argument('--device_ids',
                        default=[0],
                        help='the GPU ID for running model and we only support single gpu now, [-1] means using CPU')
    parser.add_argument('--seed',
                        default=1000,
                        help='random seed for reproduction')
    # dataset
    parser.add_argument('--num_class',
                        default=6,
                        help='the number of class')
    parser.add_argument('--train_file',
                        default='../data/smp2020-ewect-usual/train.tsv')
    parser.add_argument('--val_file',
                        default='../data/smp2020-ewect-usual/dev.tsv')
    parser.add_argument('--test_file',
                        default='../data/smp2020-ewect-usual/test.tsv')
    # train
    parser.add_argument('--is_train',
                        default=True,
                        help='whether to train')
    parser.add_argument('--is_test',
                        default=True,
                        help='whether to test')
    parser.add_argument('--checkpoint',
                        default='',
                        help='continue to train from checkpoint')
    parser.add_argument('--num_epochs',
                        default=2,
                        help='the number of training epochs')
    parser.add_argument('--max_len',
                        default=512,
                        help='the upper bound of input sentence')
    parser.add_argument('--train_batch_size',
                        default=8)
    parser.add_argument('--test_batch_size',
                        default=8)
    parser.add_argument('--lr',
                        default=3e-5)
    parser.add_argument("--warmup_steps",
                        help="the number of steps to warm up optimizer",
                        type=int,
                        default=200)
    parser.add_argument("--weight_decay",
                        help="l2 reg",
                        type=float,
                        default=0.01)
    parser.add_argument('--dropout',
                        default=0.1)
    # save
    parser.add_argument('--best_model_path',
                        default='saved/model.bin',
                        help='the path to save model with the highest performance')
    parser.add_argument('--model_saved',
                        default='saved/model.bin',
                        help='the path of saved model')
    parser.add_argument('--pred_saved',
                        default='saved/pred.txt',
                        help='the path to save prediction of test')
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
    # 如果三者同时不存在
    if not (
            os.path.isfile(args.train_file)
            or os.path.isfile(args.val_file)
            or os.path.isfile(args.test_file)
    ):
        # 自动下载数据集
        is_over = download_dataset('text_clf/smp2020-ewect-usual', cache_dir='../data')
        if not is_over:
            print(f'please download dataset manually, and mask sure data file path is correct')
            exit()
    train_data = smp2020_ewect_convert(args.train_file)
    val_data = smp2020_ewect_convert(args.val_file)
    test_data = smp2020_ewect_convert(args.test_file)
    print(f'sentence number of train: {len(train_data)}')
    print(f'sentence number of dev: {len(val_data)}')
    print(f'sentence number of test: {len(test_data)}')

    # padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # dataset
    train_dataset = DatasetCLF(train_data,
                               tokenizer,
                               max_len=args.max_len)
    val_dataset = DatasetCLF(val_data,
                             tokenizer,
                             max_len=args.max_len)
    test_dataset = DatasetCLF(test_data,
                              tokenizer,
                              max_len=args.max_len)

    # dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              collate_fn=data_collator)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            collate_fn=data_collator)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             collate_fn=data_collator)

    print(f'batch number of train: {len(train_loader)}')
    print(f'batch number of dev: {len(val_loader)}')
    print(f'batch number of test: {len(test_loader)}')
    print('----------------------------------')

    # create model
    print('create model')
    model = BertCLF(args).to(device)
    print(model)
    print('----------------------------------')

    # create optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    # create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        args.warmup_steps,
        args.num_epochs * len(train_loader))

    # create evaluation metric
    precision_metric = PMetric(average='macro')
    recall_metric = RMetric(average='macro')
    f1_metric = F1Metric(average='macro')
    metrics = Metrics(
        metrics={
            precision_metric.name: precision_metric,
            recall_metric.name: recall_metric,
            f1_metric.name: f1_metric
        },
        target_metric=f1_metric.name  # it is used for early stop or model saving
    )

    # load config
    trainer_config = Dict2Obj(vars(args))
    trainer_config.update({
        'task_name': 'text_clf',
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
