r"""
根据本项目需求对pytorch-lightning.LightningModule的简易重构
"""

import torch
from codes.nlper.modules.metrics import Metrics
from codes.nlper.utils import Dict2Obj


class StandardModel(torch.nn.Module):
    def __init__(self, configs, metrics: Metrics, **kwargs):
        super(StandardModel, self).__init__()
        self.configs = configs
        self.aux_configs = Dict2Obj()  # store auxiliary config
        self.metrics = metrics
        self.auto_optimization = True  # if False, you must manual update gradient in training_step

    def __call__(self):
        self.forward()

    def forward(self):
        pass

    # required
    def training_step(self, batch, batch_idx):
        """对训练的封装，处理一个batch，要求返回一个元组/列表，第一个是loss，其它自定义

        # :param batch: 一个batch_size大小的数据
        :param batch_idx: 该批次数据在整个数据中的顺序
        :returns: loss,
        """
        raise NotImplementedError()

    # optional
    def training_step_end(self, batch_outputs):
        """对training_step返回值的后处理，默认不处理，在自动计算梯度之后调用，详见mpl.Trainer

        :param batch_outputs: 列表，training_step的返回值
        :return: 处理后的batch_outputs
        """
        return batch_outputs

    # optional
    def training_epoch_end(self, outputs):
        """对整个train epoch的输出数据进行处理

        :param outputs: 列表，每一个元素都是training_step_end的返回值
        :return: mpl.Trainer不处理返回值
        """
        pass

    # required
    def validation_step(self, batch, batch_idx):
        """对验证的封装，处理一个batch，要求返回一个元组/列表，[loss, preds, golds, ...]，
        前三位是固定的（如果没有重写validation_epoch_end的话），预测值和真实值用于计算指标，
        要求符合相应metric的输入

        :param batch: 一个batch_size大小的数据
        :param batch_idx: 一个batch_size大小的数据
        :returns: loss, preds, golds
        """
        raise NotImplementedError()

    # optional
    def validation_step_end(self, batch_outputs):
        """对validation_step返回值的后处理，默认不处理

        :param batch_outputs: 列表，training_step的返回值
        :return: 处理后的batch_outputs
        """
        return batch_outputs

    # optional
    def validation_epoch_end(self, outputs) -> float:
        """对整个eval epoch的输出数据进行处理

        :param outputs: 元组，每一个元素都是validation_step_end的返回值
        :return: 目标指标值，用于early stop以及保存最佳模型，由metric.target而定
        """
        preds, golds = [], []
        for (batch_loss, batch_preds, batch_golds, *batch_others) in outputs:
            preds += batch_preds
            golds += batch_golds
        self.metrics.scores(golds, preds)
        self.metrics.print_values()
        return self.metrics.return_target_score()

    # required
    def test_step(self, batch, batch_idx):
        """对测试的封装，处理一个batch，推荐返回预测值，但并不强制约束

        :param batch: 一个batch_size大小的数据
        :param batch_idx: 一个batch_size大小的数据
        :return: Any
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
    def test_epoch_end(self, outputs):
        """对整个test的输出数据进行处理

        :param outputs: 列表，每一个元素都是training_step_end的返回值
        :return: mpl.Trainer不处理返回值
        """
        preds = []
        for batch_preds in outputs:
            preds += batch_preds
        return preds

    # required
    def configure_optimizers(self):
        """配置optimizer和lr scheduler，在加载数据之后调用

        :returns: optimizer(required), lr_scheduler(optional)
        """
        raise NotImplementedError()

    # optional
    def prepare_data(self):
        """ 加载数据，预处理

        :return: None
        """
        return None

    # optional
    def train_dataloader(self):
        """返回训练集数据迭代器

        :return: torch.utils.data.DataLoader
        """
        return None

    # optional
    def val_dataloader(self):
        """返回训练集数据迭代器

        :return: torch.utils.data.DataLoader
        """
        return None

    # optional
    def test_dataloader(self):
        """返回训练集数据迭代器

        :return: torch.utils.data.DataLoader
        """
        return None