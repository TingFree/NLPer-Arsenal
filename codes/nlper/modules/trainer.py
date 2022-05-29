r"""
examples下各个任务所共用的Trainer
"""

import os
from tqdm import tqdm
import torch
from codes.nlper.models import load_model, save_model
from codes.nlper.utils import Writer, Dict2Obj
from codes.nlper.modules.utils import all_to_device


class TrainerConfig():
    def __init__(self, task_name):
        self.task_name = task_name
        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.metrics = None
        self.device = None

    def update(self, **kwargs):
        for name, value in kwargs.items():
            if hasattr(self, name):
                TrainerConfig.__setattr__(self, name, value)


class Trainer():
    def __init__(self, model, config: Dict2Obj):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = config.train_loader
        self.val_loader = config.val_loader
        self.test_loader = config.test_loader
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.loss_fn = config.loss_fn
        self.metrics = config.metrics

    def train(self):
        # 加载checkpoint
        if os.path.isfile(self.config.checkpoint):
            model, state_dicts = load_model(self.model, self.config.checkpoint, return_state_dict=True)
            self.optimizer.load_state_dict(state_dicts['optimizer'])
            self.scheduler.load_state_dict(state_dicts['scheduler'])
            print(f"load checkpoint from epoch {state_dicts['epoch']}, note: epoch start with 1")
        else:
            model = self.model

        best_score = 0
        for epoch in range(1, self.config.num_epochs + 1):
            model.train()
            total_loss, total = 0, 0
            with tqdm(enumerate(self.train_loader),
                      total=len(self.train_loader)) as pbar:

                for batch_id, data in pbar:
                    data = all_to_device(data, self.config.device)
                    if self.config.task_name == 'text_clf':
                        # logits: [batch_size, n_class]
                        logits = model(**data)
                        labels = data['labels']
                        loss = self.loss_fn(
                            logits.view(-1, self.config.num_class),
                            labels.view(-1)
                        )
                    if self.config.task_name == 'text_gen':
                        # logits: [batch_size, seq_len, voc_size]
                        logits = model(**data)
                        loss = self.loss_fn(
                            logits.reshape(-1, logits.size()[2]),
                            data['encoded_tgt']['input_ids'][:, 1:].reshape(-1),  # 剔除开始符
                            ignore_index=0  # ignore pad id
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    total_loss, total = total_loss + loss.item(), total + 1
                    pbar.set_description(f'training {epoch}')
                    pbar.set_postfix(avg_loss=f'{total_loss/total:.3f}')

            avg_eval_loss = self.eval(self.val_loader)
            print(f'eval epoch {epoch}', end=' ')
            self.metrics.print_values()  # print metric_dicts
            if self.metrics.return_target_score() > best_score:
                best_score = self.metrics.return_target_score()
                save_model(model, self.config.best_model_path)
                print(f'current best model -> {self.config.best_model_path}')

    def eval(self, dataloader=None):
        model = self.model
        model.eval()
        eval_loss, total = 0, 0
        references, predicts = [], []

        with torch.no_grad():
            for data in dataloader:
                data = all_to_device(data, self.config.device)
                if self.config.task_name == 'text_clf':
                    # logits: [batch_size, n_class]
                    logits = model(**data)
                    total += logits.size()[0]
                    labels = data['labels']
                    loss = self.loss_fn(logits.view(-1, self.config.num_class),
                                        labels.view(-1),
                                        reduction='sum')
                    preds = logits.argmax(1)
                    references += labels.cpu().tolist()
                    predicts += preds.cpu().tolist()
                if self.config.task_name == 'text_gen':
                    # logits: [batch_size, seq_len, voc_size]
                    logits = model(encoded_src=data['encoded_src'])  # skip encoded_tgt
                    total += logits.size()[0]
                    golds = data['encoded_tgt']['input_ids'][:,1:]  # 剔除开始符
                    loss = self.loss_fn(logits.reshape(-1, logits.size()[2]),
                                        golds.reshape(-1),
                                        ignore_index=0,
                                        reduction='sum')
                    preds = logits.argmax(2)
                    # str of [batch_size, seq_len]
                    references += self.config.tokenizer.batch_decode(golds, skip_special_tokens=True)
                    predicts += self.config.tokenizer.batch_decode(preds, skip_special_tokens=True)
                eval_loss += loss.item()

        # 计算指标
        self.metrics.scores(references, predicts)
        return eval_loss / total

    def test(self, dataloader=None):
        model = load_model(self.model, self.config.best_model_path)
        dataloader = dataloader if dataloader else self.test_loader
        model.to(self.config.device)
        model.eval()
        references, predicts = [], []
        with_golds = False  # 标记是否存在标签

        with torch.no_grad():
            for data in dataloader:
                data = all_to_device(data, self.config.device)
                if self.config.task_name == 'text_clf':
                    # logits: [batch_size, n_class]
                    logits = model(**data)
                    preds = logits.argmax(1)
                    predicts += preds.cpu().tolist()
                    if 'labels' in data.keys():
                        with_golds = True
                        labels = data['labels']
                        references += labels.cpu().tolist()
                if self.config.task_name == 'text_gen':
                    # logits: [batch_size, seq_len, voc_size]
                    logits = model(encoded_src=data['encoded_src'])  # skip encoded_tgt
                    preds = logits.argmax(2)
                    predicts += self.config.tokenizer.batch_decode(preds, skip_special_tokens=True)
                    if 'encoded_tgt' in data.keys():
                        with_golds = True
                        golds = data['encoded_tgt']['input_ids'][:,1:]  # 剔除开始符
                        references += self.config.tokenizer.batch_decode(golds, skip_special_tokens=True)
        if with_golds:
            self.metrics.scores(references, predicts)
            self.metrics.print_values()
        Writer().write_txt(predicts, self.config.pred_saved)
