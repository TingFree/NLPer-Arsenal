r"""
文本分类任务控制器，通过text_clf_xxx.yaml快速配置，包含fit、test、eval_test等功能
"""
import importlib
import sys
sys.path.append('..')

from typing import Union
import torch
from transformers import AutoTokenizer
from nlper.utils import Dict2Obj, format_convert
from nlper.modules.metrics import Metrics
from nlper import mini_pytorch_lightning as mpl

# 根据数据集名称，查找数据转换函数，按标准数据格式读取数据
convert_dataset = {
    'iflytek': format_convert.iflytek_convert,
    'tnews': format_convert.tnews_convert,
    'smp2020-ewect-usual': format_convert.smp2020_ewect_convert,
    'smp2020-ewect-virus': format_convert.smp2020_ewect_convert
}


class TextCLFHandler():
    def __init__(self, configs: Union[dict, Dict2Obj], specialModels=None):
        self.configs = configs if isinstance(configs, Dict2Obj) else Dict2Obj(configs)
        self.specialModels = specialModels
        self._build_metrics()
        self._build_model()
        self._build_trainer()

    def _build_dataloader(self):
        pass

    def _build_optimizer(self):
        pass

    def _build_metrics(self):
        metrics_dict = {}
        for metric_name in self.configs.metrics:
            if metric_name.lower() == 'p':
                from nlper.modules.metrics import PMetric
                metrics_dict['P'] = PMetric(average='macro')
            elif metric_name.lower() == 'r':
                from nlper.modules.metrics import RMetric
                metrics_dict['R'] = RMetric(average='macro')
            elif metric_name.lower() == 'f1':
                from nlper.modules.metrics import F1Metric
                metrics_dict['F1'] = F1Metric(average='macro')
        self.metrics = Metrics(metrics_dict,
                               target_metric=self.configs.target_metric)

    def _build_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.configs.pretrained_model)

        # 根据模型名自动加载在nlper.models.text_clf下的同名模型
        module = importlib.import_module('nlper.models')
        model_name = self.configs.whole_model
        if hasattr(module, model_name):
            self.model = getattr(module, model_name)(self.configs)
        else:
            raise ValueError(
                f'{model_name} not found in nlper.models.text_clf'
            )

        # 对于非标准数据集，必须通过convert_fn转换
        if self.configs.use_convert:
            if self.configs.dataset_name in convert_dataset:
                convert_fn = convert_dataset[self.configs.dataset_name]
            else:
                raise RuntimeError(
                    'use_convert is True, but convert function has not been found'
                )
        else:
            convert_fn = None
        if self.specialModels:
            self.lightningCLF = self.specialModels.CLFModel(
                self.model,
                self.tokenizer,
                self.configs,
                metrics=self.metrics,
                convert_fn=convert_fn)
        else:
            from nlper.models import LightningCLF
            self.lightningCLF = LightningCLF(self.model,
                                             self.tokenizer,
                                             self.configs,
                                             metrics=self.metrics,
                                             convert_fn=convert_fn)

    def _build_trainer(self):
        self._trainer = mpl.StandardTrainer(self.lightningCLF,
                                            self.configs.trainer_args.gpus)

    def fit(self, train_loader=None, val_loader=None):
        self._trainer.fit(train_loader, val_loader, **self.configs.trainer_args.toDict())

    def test(self, test_loader=None, load_best=True):
        self._trainer.test(test_loader, load_best=load_best)

    def eval_test(self, test_loader=None, checkpoint_path=''):
        torch.cuda.empty_cache()
        if test_loader:
            self._trainer.eval(test_loader, checkpoint_path)
        else:
            self._trainer.eval(self._trainer.test_loader, checkpoint_path)
