r"""
根据本项目需求对pytorch-lightning.LightningModule的简易重构
"""
import re
import math
import logging
from typing import Union, List
from collections import OrderedDict
import torch
from accelerate import Accelerator
from transformers import get_scheduler, PreTrainedTokenizerBase
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from evaluate import Metric, CombinedEvaluations
from ..utils.function import create_parentDir


class MplModule(torch.nn.Module):
    def __init__(
            self,
            config,
            model: torch.nn.Module,
            tokenizer: PreTrainedTokenizerBase,
            metrics: Union[Metric, CombinedEvaluations]=None,
            **kwargs
    ):
        super(MplModule, self).__init__()

        # 使用自定义参数覆盖原始config值
        for key, value in kwargs:
            if key in config:
                config.key = value

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = metrics
        if config.seed is not None:
            set_seed(config.seed)

    def __call__(self):
        self.forward()

    def forward(self):
        pass

    # required
    def training_step(self, batch):
        """对训练的封装，处理一个batch，要求返回mean loss，其余自定义

        :param batch: 一个batch_size大小的数据
        :returns: MplOutput(loss)
        """
        raise NotImplementedError()

    # optional
    def training_step_end(self, batch_outputs):
        """对training_step返回值的后处理，默认不处理，在自动计算梯度之后调用，详见mpl.Trainer

        :param batch_outputs: training_step的返回值
        :return: 处理后的batch_outputs
        """
        return batch_outputs

    # optional
    def training_epoch_end(self, outputs:list):
        """对整个train epoch的输出数据进行处理

        :param outputs: 列表，每一个元素都是training_step_end的返回值
        :return: mpl.Trainer不处理返回值
        """
        pass

    # required
    def validation_step(self, batch, accelerator:Accelerator) -> "MplOutput":
        """对验证的封装，处理一个batch， 要求返回loss, predictions, references, 其余可以自定义，
        前三位是必须的（如果没有重写validation_epoch_end的话），预测值和真实值用于计算指标，
        要求符合相应metric的输入

        :param batch: 一个batch_size大小的数据
        :accelerator: 处理predictions和references
        :returns: MplOutput(loss, predictions, references)
        """
        raise NotImplementedError()

    # optional
    def validation_step_end(self, batch_outputs):
        """对validation_step返回值的后处理，默认不处理

        :param batch_outputs: training_step的返回值
        :return: 处理后的batch_outputs
        """
        return batch_outputs

    # optional
    def validation_epoch_end(self, outputs:list, tgt_metric:str) -> "MplOutput":
        """对整个eval epoch的输出数据进行处理

        :param outputs: 元组，每一个元素都是validation_step_end的返回值
        :param tgt_metric: 指定early stop所用的指标
        :return: MplOutput(loss:float, metric_results:Dict[str, float], target_metric_result:Dict[str, float])
        """
        eval_loss, n_examples = 0, 0
        for batch_outputs in outputs:
            n_examples += len(batch_outputs.predictions)
            eval_loss += batch_outputs.loss.item()
            self.metrics.add_batch(
                predictions=batch_outputs.predictions,
                references=batch_outputs.references
            )
        metric_results = self.metrics.compute()

        if tgt_metric in metric_results:
            tgt_metric = {tgt_metric: metric_results[tgt_metric]}
        elif tgt_metric == 'eval_loss':
            tgt_metric = {'eval_loss': eval_loss/n_examples}
        else:
            raise ValueError(f"`tgt_metric` is {tgt_metric}, not in {self.metrics.evaluation_module_names + ['eval_loss']}")
        return MplOutput(
            loss = eval_loss/n_examples,
            metric_results = metric_results,
            target_metric_result = tgt_metric
        )

    # required
    def test_step(self, batch, accelerator:Accelerator) -> "MplOutput":
        """对预测的封装，处理一个batch，推荐返回预测值，但并不强制约束

        :param batch: 一个batch_size大小的数据
        :param accelerator: 处理predictions和references
        :return: MplOutput(predictions)
        """
        raise NotImplementedError()

    # optional
    def test_step_end(self, batch_outputs):
        """对test_step返回值的后处理，默认不处理

        :param batch_outputs: 列表，training_step的返回值
        :return: 处理后的batch_outputs
        """
        return batch_outputs

    # optional
    def test_epoch_end(self, outputs:list):
        """对整个test的输出数据进行处理

        :param outputs: 列表，每一个元素都是training_step_end的返回值
        :return: MplOutput(loss:float, metric_results:Dict[str, float], predictions:list)
        """
        first = outputs[0]
        total_predictions = []
        predict_loss, n_examples = 0, 0
        for batch_outputs in outputs:
            if first.loss is not None:
                predict_loss += batch_outputs.loss
            if first.references is not None:
                self.metrics.add_batch(
                    predictions=batch_outputs.predictions,
                    references=batch_outputs.references
                )
            n_examples += len(batch_outputs.predictions)
            total_predictions.extend(batch_outputs.predictions.tolist())
        return MplOutput(
            loss = predict_loss/n_examples if first.loss is not None else None,
            metric_results = self.metrics.compute() if first.references is not None else None,
            predictions = total_predictions
        )

    # required
    def configure_optimizers(
            self,
            learning_rate: float,
            max_train_steps: int,
            warmup_ratio=0.0,
            num_warmup_steps=0,
            lr_scheduler_type='linear',
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            *args,
            **kwargs
    ):
        """配置optimizer和lr scheduler，在加载数据之后调用

        :param learning_rate: The initial learning rate for AdamW optimizer.
        :param weight_decay: The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.
        :param adam_beta1: The beta1 hyperparameter for the AdamW optimizer.
        :param adam_beta2: The beta2 hyperparameter for the AdamW optimizer.
        :param adam_epsilon: The epsilon hyperparameter for the AdamW optimizer.
        :param lr_scheduler_type: linear\cosine\cosine_with_restarts\polynomial\constant\constant_with_warmup，参考transformers.trainer_utils.SchedulerType
        :param warmup_ratio: Ratio of total training steps used for a linear warmup from 0 to learning_rate.
        :param num_warmup_steps: The number of warmup steps to do for a linear warmup from 0 to learning_rate. Overrides any effect of warmup_ratio.
        :param max_train_steps: The number of training steps to do.
        :returns: optimizer, lr_scheduler
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon
        )
        num_warmup_steps = (
            num_warmup_steps if num_warmup_steps > 0 else math.ceil(warmup_ratio * max_train_steps)
        )
        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        return optimizer, lr_scheduler

    # optional
    def prepare_data(self, **kwargs):
        """ 加载数据前的预处理

        :return: None
        """
        return None

    # optional
    def preprocess_function(self, examples):
        """加载数据后的预处理，用于datasets.map

        :param examples: 参考datasets.map的参数
        :return:
        """
        raise NotImplementedError

    # optional
    def get_train_data(
            self,
            train_file,
            label_key,
            batch_size,
            cache_dir,
            *args,
            **kwargs
    ) -> "MplOutput":
        """返回训练集数据迭代器以及处理后的数据

        :param train_file: 训练数据文件路径
        :param label_key: 数据中的标签关键字
        :param batch_size: 单GPU处理的数据批次大小
        :param cache_dir: 数据处理后的缓存目录
        :return: MplOutput(dataloader:torch.utils.data.DataLoader, dataset)
        """
        return MplOutput(dataloader=None, dataset=None)

    # optional
    def get_eval_data(
            self,
            eval_file,
            batch_size,
            cache_dir,
            *args,
            **kwargs
    ) -> "MplOutput":
        """返回开发集数据迭代器，在self.get_train_data之后调用，label_key和get_train_data中保持一致

        :param eval_file: 评估数据文件路径
        :param batch_size: 单GPU处理的数据批次大小
        :param cache_dir: 数据处理后的缓存目录
        :return: MplOutput(dataloader:torch.utils.data.DataLoader, dataset)
        """
        return MplOutput(dataloader=None, dataset=None)

    # optional
    def get_test_data(
            self,
            test_file,
            label_key,
            batch_size,
            cache_dir,
            *args,
            **kwargs
    ) -> "MplOutput":
        """返回测试集数据迭代器

        :param test_file: 测试数据文件路径
        :param label_key: 数据中的标签关键字，如果不包含标签，可以指定为None
        :param batch_size: 单GPU处理的批次大小
        :param cache_dir: 处理后的数据缓存目录
        :return: MplOutput(dataloader:torch.utils.data.DataLoader, dataset)
        """
        return MplOutput(dataloader=None, dataset=None)


class MplConfig():
    """
    将嵌套字典转换成对象，将关键字访问替换成属性访问

    >>> t = MplConfig()
    >>> t.x1 = 3e-5
    >>> t.x2.x21 = [8]
    >>> t.x2.x22 = 16
    >>> t.update({
    >>>     'x3': 0.1,
    >>>     'x2': {'x22': 32, 'x23': 64},
    >>>     'x4': {'x41':'yyy'}
    >>> })
    >>> t.toDict()  # {'x1': 3e-05, 'x2': {'x21': [8], 'x22': 32, 'x23': 64},
    >>>             #  'x3': 0.1, 'x4': {'x41': 'yyy'}}
    >>> print(t)  # str of t.toDict()
    """
    def __init__(self, init_dict=None):
        if init_dict:
            for key, value in init_dict.items():
                if self._is_valid(key):
                    if type(value) is dict:
                        self.__setattr__(key, MplConfig(value))
                    else:
                        self.__setattr__(key, value)

    def __getattr__(self, key):
        """访问一个不存在的属性时，调用该函数"""
        if self._is_valid(key):
            self.__setattr__(key, MplConfig({}))
            return self.__getattribute__(key)

    def __repr__(self):
        return str(self.toDict())

    def update(self, aux_dict):
        for key, value in aux_dict.items():
            if self._is_valid(key):
                if type(value) is dict:
                    if hasattr(self, key):
                        self.__getattribute__(key).update(value)
                    else:
                        self.__getattr__(key).update(value)
                else:
                    self.__setattr__(key, value)

    def _is_valid(self, key):
        if type(key) is str and re.match(r'[a-zA-Z_][0-9a-zA-Z_]*', key):
            return True
        raise ValueError(f'{key} is not a valid variable, please check manually')

    def toDict(self):
        target = {}
        for key, value in self.__dict__.items():
            if type(value) is not MplConfig:
                target[key] = value
            else:
                target[key] = value.toDict()
        return target


class MplOutput(OrderedDict):
    """修改自transformers.file_utils.ModelOutput，允许下标、索引、属性访问，根据创建时的顺序决定，
    如果访问一个不存在的属性，则返回None

    >>> t = MplOutput(lr=1e-5)
    >>> t.update(n_epochs=2)
    >>> t
    >>> MplOutput([('lr', 1e-05), ('n_epochs', 2)])
    >>> t[0] == t['lr'] == t.lr
    >>> True
    >>> t.lr = 5e-5  # equals t[0]=5e-5 or t['lr']=5e-5
    >>> t.batch_size = 8  # equal t['batch_size']=8
    >>> del t.lr  # equals "del t[0]" or "del t['lr']"
    """

    def __getitem__(self, k):
        """允许索引访问，同时也允许下标"""
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __getattr__(self, key):
        """访问不存在的属性"""
        return None

    def __setattr__(self, name, value):
        """设置属性时，会覆盖同名item"""
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        """设置item的同时，也设置属性"""
        # Will raise a KeyException if needed
        if isinstance(key, int):
            key = list(self.keys())[key]
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __delattr__(self, item):
        """同时删除item和属性，只有通过该模型注册的才能被删除"""
        super().__delattr__(item)
        if item in self.keys():
            self.__delitem__(item)

    def __delitem__(self, key):
        """同时删除item和属性，只有通过该模型注册的才能被删除"""
        if isinstance(key, int):
            key = list(self.keys())[key]
        super().__delitem__(key)
        if key in self.__dict__:
            self.__delattr__(key)

    def pop(self, key):
        result = self[key]
        del self[key]
        return result

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.__setitem__(k, v)

    def __str__(self):
        return str({k: self[k] for k in self.keys()})


class MplLogger():
    """管理输出内容，统一控制终端输出和日志文件中的内容
    """
    def __init__(self, name:str, accelerator:Accelerator, log_path:str, show_terminal=True, save2file=True):
        """

        :param name: name for logger
        :param accelerator: accelerate.Accelerator
        :param log_path: 日志路径
        :param show_terminal: 是否在终端显示
        :param save2file: 是否将输出结果保存在log_path中
        """
        self.logger = get_logger(name)
        self.accelerator = accelerator
        self.show_terminal = show_terminal
        self.save2file = save2file

        create_parentDir(log_path)

        logging.basicConfig(
            filename=log_path,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.INFO
        )

    def filemode(self, mode):
        self.save2file = mode

    def log(self, content, show_terminal=None, save2file=None):
        """打印content到终端和日志文件

        :param content: 内容
        :param show_terminal: True表示在终端显示，若不指定，则默认使用MplLogger创建时的初始值
        :param save2file: True表示保存到日志文件中，若不指定，则使用之前指定的默认值
        """
        save2file = self.save2file if save2file is None else save2file
        show_terminal = self.show_terminal if show_terminal is None else show_terminal

        if save2file:
            self.logger.info(content, main_process_only=True)
        if show_terminal:
            self.accelerator.print(content)
