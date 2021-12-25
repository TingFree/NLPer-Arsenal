r"""
根据本项目需求对pytorch-lightning.Trainer的简易重构
"""

import os
import torch
from tqdm import tqdm
from codes.nlper.utils import set_devices
from codes.nlper.mini_pytorch_lightning.model import StandardModel
from codes.nlper.modules.utils import all_to_device
from codes.nlper.models.io import save_model, load_model


class StandardTrainer():
    def __init__(self, model: StandardModel, gpus):
        self.device = set_devices(gpus)
        self.model = model.to(self.device)
        self.model.prepare_data()
        self.train_loader = self.model.train_dataloader()
        self.model.aux_configs.num_train_batch = len(self.train_loader) if self.train_loader else None
        self.val_loader = self.model.val_dataloader()
        self.test_loader = self.model.test_dataloader()
        self.model.optimizer, self.model.scheduler = self.model.configure_optimizers()

    def fit(self,
            train_loader=None,
            val_loader=None,
            max_epochs=1,
            check_val_n_epoch=1,
            accumulate_step=1,
            early_stop=False,
            patience=3,
            **kwargs):
        self._set_mode('train')

        if not train_loader:
            train_loader = self.train_loader

        cur_patience = patience
        best_score = 0

        outputs = []
        for epoch in range(1, max_epochs + 1):
            total_loss, total = 0, 0
            with tqdm(total=len(train_loader),
                      desc=f'train epoch: {epoch}') as t:
                for batch_idx, batch in enumerate(train_loader):
                    batch = all_to_device(batch, self.device)
                    batch_outputs = list(self.model.training_step(batch, batch_idx))  # tuple to list
                    # update gradient and lr --start
                    loss = batch_outputs[0]
                    if self.model.auto_optimization:
                        loss /= accumulate_step
                        loss.backward()
                        if (batch_idx + 1) % accumulate_step == 0:
                            self.model.optimizer.step()
                            if self.model.scheduler:
                                self.model.scheduler.step()
                            self.model.optimizer.zero_grad()
                    # update gradient and lr --end
                    batch_outputs[0] = loss.item()  # detach
                    batch_outputs = self.model.training_step_end(batch_outputs)
                    outputs.append(batch_outputs)
                    total_loss, total = total_loss + loss.item(), total + 1
                    t.set_postfix(avg_loss=f'{total_loss/total:.3f}')
                    t.update(1)
                self.model.training_epoch_end(outputs)

            # skip eval when check_val_n_epoch equal 0
            if check_val_n_epoch == 0:
                continue
            if epoch % check_val_n_epoch == 0:
                if not val_loader:
                    val_loader = self.val_loader
                target_score = self.eval(val_loader)
                # save best model and evaluate patience --start
                if target_score > best_score:
                    best_score = target_score
                    save_model(
                        self.model,
                        os.path.join(self.model.configs.out_dir, 'best_model.bin'))
                    cur_patience = patience
                else:
                    cur_patience -= 1
                if early_stop and cur_patience == 0:
                    print(
                        f'patience over, current epoch is {epoch}, '
                        f'best score of {self.model.metrics.target} is {best_score} at epoch {epoch-patience*check_val_n_epoch}'
                    )
                    break
                # save best model and evaluate patience --end
                self._set_mode('train')
        save_model(self.model, os.path.join(self.model.configs.out_dir, 'last_model.bin'))

    def eval(self, val_loader=None, checkpoint_path=None):
        self._set_mode('eval')
        if checkpoint_path:
            self.model = load_model(self.model, checkpoint_path)

        if not val_loader:
            val_loader = self.val_loader

        outputs = []
        with tqdm(total=len(val_loader), ncols=80) as t:
            for batch_idx, batch in enumerate(val_loader):
                batch = all_to_device(batch, self.device)
                batch_outputs = self.model.validation_step(batch, batch_idx)
                batch_outputs = self.model.validation_step_end(batch_outputs)
                outputs.append(batch_outputs)
                t.update(1)
        return self.model.validation_epoch_end(outputs)

    def test(self, test_loader=None, load_best=True):
        self._set_mode('eval')
        if load_best:
            self.model = load_model(self.model, os.path.join(self.model.configs.out_dir, 'best_model.bin'))
        else:
            self.model = load_model(self.model, os.path.join(self.model.configs.out_dir, 'last_model.bin'))

        if not test_loader:
            test_loader = self.test_loader

        outputs = []
        with tqdm(total=len(test_loader), ncols=80, desc='test') as t:
            for batch_idx, batch in enumerate(test_loader):
                batch = all_to_device(batch, self.device)
                batch_outputs = self.model.test_step(batch, batch_idx)
                batch_outputs = self.model.test_step_end(batch_outputs)
                outputs.append(batch_outputs)
                t.update(1)
        self.model.test_epoch_end(outputs)

    def _set_mode(self, mode):
        # mode: 'train' or 'eval'
        if mode == 'train':
            self.model.train()
            torch.set_grad_enabled(True)
        if mode == 'eval':
            self.model.eval()
            torch.set_grad_enabled(False)