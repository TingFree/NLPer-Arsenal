r"""
文本分类任务之多分类教程，全流程直观展示，适合NLP入门人员
"""
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
from nlper.modules.mlp import MLP
from nlper.utils import DatasetCLF
from nlper.modules.trainer import Trainer
from nlper.modules.metrics import Metrics, PMetric, RMetric, F1Metric
from nlper.utils.format_convert import smp2020_ewect_convert
from nlper.utils import seed_everything, set_devices, Dict2Obj


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
                        default=[5],
                        help='the GPU ID for running model, [-1] means using CPU')
    parser.add_argument('--seed',
                        default=1000,
                        help='random seed for reproduction')
    # dataset
    parser.add_argument('--num_class',
                        default=6,
                        help='the number of class')
    parser.add_argument('--train_file',
                        default='../data/smp2020-ewect/usual/train.tsv')
    parser.add_argument('--dev_file',
                        default='../data/smp2020-ewect/usual/dev.tsv')
    parser.add_argument('--test_file',
                        default='../data/smp2020-ewect/usual/test.tsv')
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

    train = smp2020_ewect_convert(args.train_file)
    dev = smp2020_ewect_convert(args.dev_file)
    test = smp2020_ewect_convert(args.test_file)
    print(f'sentence number of train: {len(train)}')
    print(f'sentence number of dev: {len(dev)}')
    print(f'sentence number of test: {len(test)}')

    # padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # dataset
    trainDataset = DatasetCLF(train,
                              tokenizer,
                              max_len=args.max_len)
    devDataset = DatasetCLF(dev,
                            tokenizer,
                            max_len=args.max_len)
    testDataset = DatasetCLF(test,
                             tokenizer,
                             max_len=args.max_len)

    # dataloader
    trainloader = DataLoader(trainDataset,
                             batch_size=args.train_batch_size,
                             shuffle=True,
                             collate_fn=data_collator)
    devloader = DataLoader(devDataset,
                           batch_size=args.test_batch_size,
                           shuffle=False,
                           collate_fn=data_collator)
    testloader = DataLoader(testDataset,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            collate_fn=data_collator)

    print(f'batch number of train: {len(trainloader)}')
    print(f'batch number of dev: {len(devloader)}')
    print(f'batch number of test: {len(testloader)}')
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
        args.num_epochs * len(trainloader))

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
        'train_loader': trainloader,
        'dev_loader': devloader,
        'test_loader': testloader,
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
