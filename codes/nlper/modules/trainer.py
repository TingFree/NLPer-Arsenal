r"""
examples下各个任务所共用的Trainer
"""

import os
from tqdm import tqdm
import warnings
import torch
from nlper.models import load_model, save_model
from nlper.utils import save_data, Dict2Obj
from nlper.modules.utils import all_to_device


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
        self.dev_loader = config.dev_loader
        self.test_loader = config.test_loader
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.loss_fn = config.loss_fn
        self.metrics = config.metrics

    def train(self):
        # 加载checkpoint
        if os.path.isfile(self.config.checkpoint):
            # todo: load optimizer and scheduler
            model = load_model(self.model, self.config.checkpoint)
            warnings.warn("you set checkpoint, but we don't load optimizer and scheduler yet, "
                          "we will repair it later")
        else:
            model = self.model

        best_score = 0
        for epoch in range(1, self.config.num_epochs + 1):
            print(f'epoch:{epoch}')
            model.train()
            with tqdm(enumerate(self.train_loader),
                      total=len(self.train_loader)) as pbar:

                for batch_id, data in pbar:
                    data = all_to_device(data, self.config.device)
                    labels = data['labels']
                    logits = model(**data)
                    if self.config.task_name == 'text_clf':
                        loss = self.loss_fn(
                            logits.view(-1, self.config.num_class),
                            labels.view(-1)
                        )
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    pbar.set_description('training')

            avg_eval_loss = self.eval(self.dev_loader)
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
                labels = data['labels']
                total += labels.shape[0]
                logits = model(**data)
                if self.config.task_name == 'text_clf':
                    loss = self.loss_fn(logits.view(-1, self.config.num_class),
                                        labels.view(-1),
                                        reduction='sum')
                eval_loss += loss.item()

                preds = logits.argmax(1)
                references += labels.cpu().tolist()
                predicts += preds.cpu().tolist()
        # 计算指标
        self.metrics.scores(references, predicts)
        return eval_loss / total

    def test(self, dataloader=None):
        model = load_model(self.model, self.config.best_model_path)
        dataloader = dataloader if dataloader else self.test_loader
        model.to(self.config.device)
        model.eval()
        predicts = []
        with torch.no_grad():
            for data in dataloader:
                data = all_to_device(data, self.config.device)
                logits = model(**data)
                preds = logits.argmax(1)
                predicts += preds.cpu().tolist()
        save_data(predicts, self.config.pred_saved, f_type='txt')
