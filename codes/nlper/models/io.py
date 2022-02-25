r"""
加载模型与保存模型，来自uer
"""

import torch
from codes.nlper.utils.io import create_parentDir


def load_model(model, model_path, return_state_dict=False):
    state_dicts = torch.load(model_path, map_location='cpu')
    if hasattr(model, "module"):  # DataParallel封装过的多卡训练模型
        model.module.load_state_dict(state_dicts['model'], strict=False)
    else:
        model.load_state_dict(state_dicts['model'], strict=False)
    if return_state_dict:
        return model, state_dicts
    return model


def save_model(model, model_path, optimizer=None, scheduler=None, epoch=None):
    create_parentDir(model_path)
    state_dicts = {'epoch':epoch, 'model':None, 'optimizer':None, 'scheduler':None}
    if optimizer:
        state_dicts['optimizer'] = optimizer.state_dict()
    if scheduler:
        state_dicts['scheduler'] = scheduler.state_dict()
    if hasattr(model, "module"):  # DataParallel封装过的多卡训练模型
        state_dicts['model'] = model.module.state_dict()
    else:
        state_dicts['model'] = model.state_dict()
    torch.save(state_dicts, model_path)
    print(f'model -> {model_path} over')
