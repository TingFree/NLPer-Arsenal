r"""
加载模型与保存模型，来自uer
"""

import torch
from codes.nlper.utils.io import create_parentDir


def load_model(model, model_path):
    if hasattr(model, "module"):  # DataParallel封装过的多卡训练模型
        model.module.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    return model


def save_model(model, model_path):
    create_parentDir(model_path)
    if hasattr(model, "module"):  # DataParallel封装过的多卡训练模型
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    print(f'model -> {model_path} over')
