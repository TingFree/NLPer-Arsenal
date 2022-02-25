r"""
封装各种评估指标
"""

import torch
import numpy as np
from typing import Dict, Callable, Optional
from sklearn.metrics import precision_score, recall_score, f1_score
# from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge


def type_transfor(data, target_type, in_type=None):
    """transform data into specific type(list, numpy and tensor)

    :param data: list, np.ndarray, torch.tensor
    :param target_type: list, np.ndarray, torch.tensor
    :param in_type: data[0] type
    :return: new data with target type
    """
    if in_type and isinstance(data, target_type) and isinstance(data[0], in_type):
        return data
    if not in_type and isinstance(data, target_type):
        return data
    elif isinstance(data, list):
        if target_type is np.ndarray:
            return np.array(data)  # list -> numpy
        else:
            return torch.tensor(data, dtype=target_type)  # list -> tensor
    elif isinstance(data, np.ndarray):
        if target_type is list:
            return data.tolist()  # numpy -> list
        else:
            return torch.from_numpy(data)  # numpy -> tensor
    else:
        if target_type is list:
            return data.tolist()  # tensor -> list
        else:
            return data.numpy()  # tensor -> numpy


class MetricBase():
    def __init__(self, name: str, fun: Optional[Callable], data_type, in_type=None, **kwargs):
        """
        封装基础的评估函数

        :param name: 自定义评估函数名
        :param fun: 函数
        :param data_type: 函数接受的数据类型
        :param in_type: data[0]的数据类型
        :param kwargs: 函数的其它参数，非golds和preds
        """
        self.name = name
        self.metric = fun
        self.data_type = data_type
        self.in_type = in_type
        self.kwargs = kwargs

    def score(self, golds, preds):
        golds = type_transfor(golds, self.data_type, self.in_type)
        preds = type_transfor(preds, self.data_type, self.in_type)
        return self.metric(golds, preds, **self.kwargs)

    def score_end(self, scores):
        return scores


class Metrics():
    def __init__(self, metrics: Dict[str, MetricBase], target_metric: str):
        """
        指定模型在训练与测试过程中用来评估性能的指标

        :param metrics: 通过字典的形式添加评估指标，例如{'F1': f1_metric}, f1_metric:MetricBase
        :param target_metric: 用于早停以及保存最佳模型，必须为metrics中的一个，例如'F1'
        """
        self._metrics = metrics
        self.target = target_metric
        if self.target and self.target not in metrics.keys():
            raise ValueError('target_metric must be one of metrics or None')
        self.metric_values = {}

    def print_values(self):
        if not self.metric_values:
            print(
                'No Metric Calculated here, please call Metrics.scores or Metrics.target_score before print_values'
            )
        else:
            for name, value in self.metric_values.items():
                print(f'{name}: {value}')

    def scores(self, golds, preds) -> Dict[str, float]:
        for metric_name in self._metrics.keys():
            scores = self._metrics[metric_name].score_end(
                self._metrics[metric_name].score(golds, preds)
            )
            self.metric_values[metric_name] = scores
        return self.metric_values

    def target_score(self, golds, preds):
        self.metric_values[self.target] = self._metrics[self.target].score(
            golds, preds)
        return self.metric_values[self.target]

    def return_target_score(self):
        if self.target in self.metric_values.keys():
            if self.target == 'rouge-1' or self.target == 'rouge-2':
                return self.metric_values[self.target]['r']
            if self.target == 'rouge-l':
                return self.metric_values[self.target]['f']
            return self.metric_values[self.target]


class PMetric(MetricBase):
    def __init__(self,
                 name='P',
                 fun=precision_score,
                 data_type=np.ndarray,
                 **kwargs):
        super(PMetric, self).__init__(name, fun, data_type, **kwargs)


class RMetric(MetricBase):
    def __init__(self,
                 name='R',
                 fun=recall_score,
                 data_type=np.ndarray,
                 **kwargs):
        super(RMetric, self).__init__(name, fun, data_type, **kwargs)


class F1Metric(MetricBase):
    def __init__(self,
                 name='F1',
                 fun=f1_score,
                 data_type=np.ndarray,
                 **kwargs):
        super(F1Metric, self).__init__(name, fun, data_type, **kwargs)


class RougeMetric(MetricBase):
    def __init__(self, name='rouge-l'):
        """计算rouge-1/2/l指标，golds: List[str], preds:[str]，rouge-1/2侧重召回率，rouge-l侧重F1

        :param name: rouge-1, rouge-2, rouge-l
        """
        assert name=='rouge-1' or name=='rouge-2' or name=='rouge-l'
        rouge = Rouge()
        super(RougeMetric, self).__init__(name, rouge.get_scores, list, str, avg=True)

    def score(self, golds, preds):
        golds = type_transfor(golds, self.data_type, self.in_type)
        preds = type_transfor(preds, self.data_type, self.in_type)
        return self.metric(preds, golds, **self.kwargs)

    def score_end(self, scores):
        # {'p':x, 'r':x, 'f':x}
        return scores[self.name]
#
# class Bleu4Metric(MetricBase):
#     def __init__(self, name='bleu-4'):
#         super(Bleu4Metric, self).__init__(name, corpus_bleu, list)