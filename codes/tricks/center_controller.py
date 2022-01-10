r"""
加载配置文件，快速运行，方便复现
"""
import sys
sys.path.append('..')
import os
import multiprocessing as mp
import argparse
import importlib
from codes.nlper.utils import (
    read_data,
    Dict2Obj,
    seed_everything,
    ProcessStatus
)
from text_clf_handler import TextCLFHandler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_config',
                        default='default_configs/text_clf_smp2020_ewect_usual.yaml')
    parser.add_argument('--trick_name',
                        default='',
                        choices=['fgm', 'eight_bit', 'unsup_simcse'],
                        help='the subdir name of tricks, which contains specialModels.py in this subdir')
    # 以下设置将覆盖task_config中的参数
    parser.add_argument('--whole_model',
                        default='BertCLF',
                        choices=['BertCLF'],
                        help='the model to run')
    parser.add_argument('--gpu',
                        default=0,
                        type=int,
                        help='-1: cpu, device id to select GPU')
    parser.add_argument('--out_dir',
                        default='saved/')
    parser.add_argument('--pretrained_model',
                        default='bert-base-chinese')
    args = parser.parse_args()

    task_config = Dict2Obj(read_data(args.task_config))
    task_config.trainer_args.gpus = [args.gpu]
    task_config.out_dir = args.out_dir
    task_config.pretrained_model = args.pretrained_model
    task_config.whole_model = args.whole_model

    # 开启系统监控
    if task_config.trainer_args.gpus[0] != -1:
        processStatus = ProcessStatus(task_config.trainer_args.gpus[0])
    else:
        processStatus = ProcessStatus()
    monitor_process = mp.Process(target=processStatus.record_running_status)
    monitor_process.start()

    # 固定随机数
    if task_config.seed is not None:
        seed_everything(task_config.seed)

    # 选择trick
    if args.trick_name:
        special_models = importlib.import_module(args.trick_name + '.specialModels')
    else:
        special_models = None

    # 加载trick到指定任务中
    if task_config.task_name == 'text_clf':
        taskHandler = TextCLFHandler(task_config, special_models)
    else:
        raise ValueError(f'your task name is {task_config.task_name} which is not supported yet')

    # 训练，保存在验证集上指标最高的参数（best_model.bin），以及最后一轮的参数（last_model.bin）
    if task_config.is_train:
        taskHandler.fit()
    # 预测，推理测试集的标签
    if task_config.is_test:
        taskHandler.test(load_best=task_config.load_best)
    # 计算模型在测试集上的指标
    if task_config.is_eval_test:
        # last_model.bin or best_model.bin
        taskHandler.eval_test(checkpoint_path=os.path.join(task_config.out_dir, 'best_model.bin'))

    # 结束系统监控，打印监控结果
    monitor_process.terminate()
    processStatus.print_statisticAnalysis()
    # processStatus.plot_running_info()
