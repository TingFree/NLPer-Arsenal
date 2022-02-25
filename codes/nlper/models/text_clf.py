r"""
各种文本分类模型的实现
"""

import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertModel
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup
from codes.nlper.modules import MLP
from codes.nlper.utils import DatasetCLF, Dict2Obj
from codes.nlper.utils import load_nlp_data, save_data
from codes.nlper.modules.modeling_outputs import TextCLFOutput, LightningOutput
from codes.nlper import mini_pytorch_lightning as mpl


class LightningCLF(mpl.StandardModel):
    def __init__(self, model, tokenizer, configs: Dict2Obj, metrics, convert_fn):
        super(LightningCLF, self).__init__(configs, metrics)
        self.configs = configs
        self.aux_configs = Dict2Obj()
        self.metrics = metrics
        self.model = model
        self.tokenizer = tokenizer
        self.convert_fn = convert_fn

    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        outputs = self.model(**batch)
        logits = outputs.logits
        loss = F.cross_entropy(logits.view(-1, self.configs.num_class),
                               labels.view(-1))
        return LightningOutput(loss=loss)

    def validation_step(self, batch, batch_idx):
        labels = batch['labels']
        outputs = self.model(**batch)
        logits = outputs.logits
        loss = F.cross_entropy(logits.view(-1, self.configs.num_class),
                               labels.view(-1))
        batch_preds = logits.argmax(1).cpu().tolist()
        batch_golds = labels.cpu().tolist()
        return LightningOutput(
            loss=loss,
            preds=batch_preds,
            golds=batch_golds
        )

    def validation_epoch_end(self, outputs):
        epoch_preds, epoch_golds = [], []
        for batch_outputs in outputs:
            epoch_preds += batch_outputs.preds
            epoch_golds += batch_outputs.golds
        self.metrics.scores(epoch_golds, epoch_preds)
        self.metrics.print_values()
        return self.metrics.return_target_score()

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        logits = outputs.logits
        # prob, pred
        return LightningOutput(
            probs = F.softmax(logits, dim=-1).cpu().tolist(),
            preds = logits.argmax(1).cpu().tolist()
        )

    def test_epoch_end(self, outputs):
        probs, preds = [], []
        for batch_outputs in outputs:
            probs += [' '.join([str(p) for p in prob]) for prob in batch_outputs.probs]
            preds += batch_outputs.preds
        save_data(probs, os.path.join(self.configs.out_dir, 'test_pred.probs.txt'))
        save_data(preds, os.path.join(self.configs.out_dir, 'test_pred.txt'))

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.configs.weight_decay},
            {'params': [p for n, p in self.named_parameters()if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.configs.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    self.configs.warmup_steps,
                                                    self.configs.trainer_args.max_epochs * self.aux_configs.num_train_batch)
        return optimizer, scheduler

    def prepare_data(self) -> None:
        """ check & load data, the format of each line is 'text label', separated by tab and 'label'
        must be int, such as 0~num_labels-1
        """
        train_file = self.configs.train_file
        val_file = self.configs.val_file
        test_file = self.configs.test_file
        self.collate_fn = DataCollatorWithPadding(tokenizer=self.tokenizer)
        if self.convert_fn:
            self._train_data = self.convert_fn(train_file, load_label=True)
            self._val_data = self.convert_fn(val_file, load_label=True)
            self._test_data = self.convert_fn(test_file, load_label=self.configs.is_eval_test)
        else:
            self._train_data = load_nlp_data(train_file, task_name=self.configs.task_name)
            self._val_data = load_nlp_data(val_file, task_name=self.configs.task_name)
            self._test_data = load_nlp_data(test_file, task_name=self.configs.task_name)

    def train_dataloader(self):
        self.train_data = DatasetCLF(self._train_data,
                                     self.tokenizer,
                                     self.configs.max_len,
                                     load_label=True)
        return DataLoader(self.train_data,
                          batch_size=self.configs.train_batch_size,
                          collate_fn=self.collate_fn,
                          shuffle=True,
                          num_workers=16)

    def val_dataloader(self):
        self.val_data = DatasetCLF(self._val_data,
                                   self.tokenizer,
                                   self.configs.max_len,
                                   load_label=True)
        return DataLoader(self.val_data,
                          batch_size=self.configs.val_batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=16)

    def test_dataloader(self):
        self.test_data = DatasetCLF(self._test_data,
                                    self.tokenizer,
                                    self.configs.max_len,
                                    load_label=self.configs.is_eval_test)
        return DataLoader(self.test_data,
                          batch_size=self.configs.val_batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=16)


class BertCLF(nn.Module):
    def __init__(self, args):
        super(BertCLF, self).__init__()
        self.bert = AutoModel.from_pretrained(args.pretrained_model)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.clf = MLP([self.bert.config.hidden_size, args.num_class],
                       'tanh',
                       dropout=args.dropout)

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.clf(outputs[1])

        return TextCLFOutput(
            logits = logits,
            seqEmb = outputs[1]
        )
