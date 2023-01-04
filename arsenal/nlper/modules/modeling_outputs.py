from collections import OrderedDict
import torch


class ModelOutput(OrderedDict):
    """修改自transformers.file_utils.ModelOutput，允许下标、索引、属性访问，根据创建时的顺序决定

    >>> t = ModelOutput(lr=1e-5)
    >>> t.update(n_epochs=2)
    >>> t
    >>> ModelOutput([('lr', 1e-05), ('n_epochs', 2)])
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

    def to_tuple(self):
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class MplOutput(ModelOutput):
    """
    封装mpl.StandardModel中training_step、validation_step的输出

    Args:
        loss: 可求导的损失
        preds: list, [batch_size, xxx], 模型的输出值，用于和golds计算指标，具体形式根据不同的任务决定
        golds: list, [batch_size, xxx], 数据的真实标签，用于和preds计算指标
    """
    loss: torch.Tensor = None
    preds: list = []
    golds: list = []
    metric_results: dict = {}
    target_metric_result: dict = {}


class TextCLFOutput(ModelOutput):
    """
    封装TextCLF模型的输出

    Args:
        logits: Tensor, [batch_size, num_labels], 模型最后的输出结果，用于计算损失，非概率值
        seqEmb: Tensor, [batch_size, hidden_size], 最终用于分类的句子级表示
    """
    logits: torch.Tensor = None
    seqEmb: torch.Tensor = None


class EncoderOutput(ModelOutput):
    """
    封装Encoder的输出

    Args:
        seqEmb: Tensor, [batch_size, seq_len, hidden_size]
    """
    seqEmb: torch.Tensor = None


class DecoderOutput(ModelOutput):
    """
    封装Decoder的输出

    Args:
        last_hidden_state: Tensor, [batch_size, seq_len, hidden_size], 解码器的预测结果
    """
    last_hidden_state: torch.Tensor = None


class TextGenOutput(ModelOutput):
    """
    封装TextGen模型的输出

    Args:
        pred: Tensor, [batch_size, seq_len, vocab_size], 用于计算loss

    """