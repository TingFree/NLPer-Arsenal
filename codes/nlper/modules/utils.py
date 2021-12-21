"""
和模型相关的工具函数
"""

from typing import Union, List, Dict
import torch
import torch.nn as nn
import torch.nn.init as init


def initial_parameter(net, initial_method=None):
    r"""source from fastnlp, A method used to initialize the weights of PyTorch models.
    https://github.com/fastnlp/fastNLP/blob/master/fastNLP/modules/utils.py

    :param net: a PyTorch model
    :param str initial_method: one of the following initializations.

            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform

    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(
                m, nn.Conv3d):  # for all the cnn
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)  # weight
                else:
                    init.normal_(w.data)  # bias
        elif m is not None and hasattr(m, 'weight') and \
                hasattr(m.weight, "requires_grad"):
            if len(m.weight.size()) > 1:
                init_method(m.weight.data)
            else:
                init.normal_(m.weight.data)  # batchnorm or layernorm
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias

    net.apply(weights_init)


def all_to_device(data: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
                  device=torch.device('cpu')):
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, (list, dict)) or hasattr(data, 'keys'):
        keys = range(len(data)) if isinstance(data, list) else data.keys()
        for k in keys:
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device)
    return data
