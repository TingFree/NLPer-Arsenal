import warnings
from typing import List
import os
import re
import random
import time
import multiprocessing as mp
import psutil
import pynvml
import numpy as np
import torch
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def seed_everything(seed=1000):
    """seed everything to reproduce your experiments

    :param int seed: default 1000
    :return: None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def set_devices(device_ids: List[int]):
    """setting the global environment of CUDA

    :param device_ids: list of device id, [-1] is cpu
    :return: torch.device
    """
    if type(device_ids) != list:
        raise TypeError(f'the gpus type should be List[int], not {type(device_ids)}')
    if len(device_ids) > 1:
        warnings.warn(f'we only support cpu or single gpu now, '
                      f'but you input {len(device_ids)} device id, and only the first will be used')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])

    if device_ids[0] != -1:
        print(f'Training on GPU {device_ids}')
        return torch.device('cuda')
    else:
        print('Training on CPU')
        return torch.device('cpu')


def count_params(model, show=False):
    num_params = 0
    if show:
        for name, p in model.named_parameters():
            print(f'{name}: {str(p.size())}')
            num_params += p.numel()
    else:
        for name, p in model.named_parameters():
            num_params += p.numel()
    return num_params


def format_runTime(seconds:float):
    """format running time to `day hours:minutes:seconds`

    :param seconds: 通常来说是两次time.time()的差值
    :return: format string
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    h = '0' + str(int(h)) if h < 10 else str(int(h))
    m = '0' + str(int(m)) if m < 10 else str(int(m))
    s = '0' + str(int(s)) if s < 10 else str(int(s))
    if d == 0:
        return f'{h}:{m}:{s}'
    else:
        return f'{d}d {h}:{m}:{s}'


class ProcessStatus():
    """记录程序运行过程中GPU/CPU/内存的全局使用情况（不一定是主进程的实际使用情况，暂未实现进程跟踪功能）

    >>> gpu = 0  # 指定0号GPU，或者为None，不指定GPU
    >>> processStatus = ProcessStatus(gpu)
    >>> p = mp.Process(target=processStatus.record_running_status, args=(1,))
    >>> p.start()  # 开始执行监控进程
    >>> # 执行主进程，例如运行程序
    >>> p.terminate()  # 终结监控进程
    >>> processStatus.print_statisticAnalysis()  # 打印表信息
    >>> processStatus.plot_running_info()  # 打印图信息
    """
    def __init__(self, gpu:int=None):
        self.start = time.time()
        self.running_info = mp.Manager().list()
        self.gpu = gpu
        if gpu:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.device_total_memory = round(gpu_info.total/1024**2)  # MiB
            self.driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            self.device_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            pynvml.nvmlShutdown()

    def record_running_status(self, interval=1):
        """供多进程调用，监控程序运行过程中的GPU、CPU、内存变化

        :param interval: 记录间隔，默认 1s 记录一次
        :return: 不间断运行，直至主进程内结束该子进程
        """
        pynvml.nvmlInit()
        start = self.start
        if self.gpu != None:  # 指定GPU的情况下
            while True:
                cur_time = time.time()
                if cur_time - start >= interval:
                    start = cur_time
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu)
                    gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem = psutil.virtual_memory()
                    self.running_info.append({
                        'cur_time': cur_time,
                        'gpu_used': round(gpu_info.used / 1024 ** 2, 2),  # GPU显存占用量（MiB）
                        'gpu_util': pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,  # GPU使用率（0~100）
                        'cpu_util': psutil.cpu_percent(),  # CPU使用率（0.0~100.0）
                        'mem_util': mem.percent,  # 内存使用率（0.0~100.0）
                        'mem_used': round(mem.used / 1024 ** 2)  # 内存占用量（MiB）
                    })
        else:  # 不指定GPU的情况下
            while True:
                cur_time = time.time()
                if cur_time - start >= interval:
                    start = cur_time
                    mem = psutil.virtual_memory()
                    self.running_info.append({
                        'cur_time': cur_time,
                        'cpu_util': psutil.cpu_percent(),  # CPU使用率（0.0~100.0）
                        'mem_util': mem.percent,  # 内存使用率（0.0~100.0）
                        'mem_used': round(mem.used / 1024 ** 2)  # 内存占用量（MiB）
                    })

    def print_statisticAnalysis(self):
        """统计分析程序运行时间以及GPU/CPU/内存使用情况，以表格形式呈现
        """
        start = self.start
        table = PrettyTable(['Param', 'Value'])
        if self.gpu != None:  # 指定GPU的情况下
            table.add_row(['cuda version', torch.version.cuda])
            table.add_row(['driver version', self.driver_version])
            table.add_row(['device', self.device_name])
            table.add_row(['device id', self.gpu])
            table.add_row(['start time', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))])
            table.add_row(['end time', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())])
            table.add_row(['running time', format_runTime(time.time() - start)])
            table.add_row(['device total memory', f'{self.device_total_memory} MiB'])
            table.add_row(['device max used memory', f"{round(np.max([t['gpu_used'] for t in self.running_info]), 2)} MiB"])
            table.add_row(['device avg util ratio', f"{round(np.mean([t['gpu_util'] for t in self.running_info]), 2)}%"])
        else:  # 不指定GPU的情况下
            table.add_row(['start time', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))])
            table.add_row(['end time', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())])
            table.add_row(['running time', format_runTime(time.time() - start)])
        table.add_row(['cpu avg util ratio', f"{round(np.mean([t['cpu_util'] for t in self.running_info]), 2)}%"])
        table.add_row(['memory max used', f"{round(np.max([t['mem_used'] for t in self.running_info]), 2)} MiB"])
        table.add_row(['memory avg util ratio', f"{round(np.mean([t['mem_util'] for t in self.running_info]), 2)}%"])
        table.align['Param'] = 'l'
        table.align['Value'] = 'l'
        print(table)

    def plot_running_info(self, show=False, saved_path='./status.png'):
        """以图表形式展现程序运行过程中的GPU/CPU/内存使用情况，默认不显示，只保存在'./status.png'

        :param show: 是否调用plt.show()画出该图
        :param saved_path: 将图保存在指定位置
        """
        font = FontProperties()
        font.set_family('serif')
        font.set_name('Times New Roman')
        font.set_style('normal')
        font.set_size(12)
        plt.style.use(['science', 'no-latex'])
        plt.figure(figsize=(12, 12), dpi=300)

        cur_time = [item['cur_time']-self.start for item in self.running_info]
        cpu_util = [item['cpu_util'] for item in self.running_info]
        mem_util = [item['mem_util'] for item in self.running_info]
        mem_used = [item['mem_used'] for item in self.running_info]

        if self.gpu != None:
            gpu_used = [item['gpu_used'] for item in self.running_info]
            gpu_util = [item['gpu_util'] for item in self.running_info]
            
            ax = plt.subplot(2, 1, 1)
            ax.plot(cur_time, gpu_util, label='gpu_util')
            ax.plot(cur_time, cpu_util, label='cpu_util')
            ax.plot(cur_time, mem_util, label='mem_util')
            plt.xticks(font_properties=font)
            plt.yticks(font_properties=font)
            plt.gca().set_ylabel('percentage', font_properties=font, fontsize=16)
            plt.legend()
            ax = plt.subplot(2, 1, 2)
            ax.plot(cur_time, gpu_used, label='gpu_used')
            ax.plot(cur_time, mem_used, label='mem_used')
            plt.xticks(font_properties=font)
            plt.yticks(font_properties=font)
            plt.gca().set_xlabel('time', font_properties=font, fontsize=16)
            plt.gca().set_ylabel('capacity', font_properties=font, fontsize=16)
            plt.legend()
            plt.title("status", font_properties=font, fontsize=20)
        else:
            ax = plt.subplot(2, 1, 1)
            ax.plot(cur_time, cpu_util, label='cpu_util')
            ax.plot(cur_time, mem_util, label='mem_util')
            plt.xticks(font_properties=font)
            plt.yticks(font_properties=font)
            plt.gca().set_ylabel('percentage', font_properties=font, fontsize=16)
            plt.legend()
            ax = plt.subplot(2, 1, 2)
            ax.plot(cur_time, mem_used, label='mem_used')
            plt.xticks(font_properties=font)
            plt.yticks(font_properties=font)
            plt.gca().set_xlabel('time', font_properties=font, fontsize=16)
            plt.gca().set_ylabel('capacity', font_properties=font, fontsize=16)
            plt.legend()
            plt.title("status", font_properties=font, fontsize=20)

        if show:
            plt.show()
        if saved_path:
            plt.savefig('./status.png')


class Timer(object):
    """Computes elapsed time."""
    def __init__(self, name):
        self.name = name
        self.running = True
        self.total = 0
        self.start = round(time.time(), 2)
        self.intervalTime = round(time.time(), 2)
        print("<> <> <> Starting Timer [{}] <> <> <>".format(self.name))

    def reset(self):
        self.running = True
        self.total = 0
        self.start = round(time.time(), 2)
        return self

    def interval(self, intervalName=''):
        intervalTime = self._to_hms(round(time.time() - self.intervalTime, 2))
        print("<> <> Timer [{}] <> <> Interval [{}]: {} <> <>".format(
            self.name, intervalName, intervalTime))
        self.intervalTime = round(time.time(), 2)
        return intervalTime

    def stop(self):
        if self.running:
            self.running = False
            self.total += round(time.time() - self.start, 2)
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = round(time.time(), 2)
        return self

    def time(self):
        if self.running:
            return round(self.total + time.time() - self.start, 2)
        return self.total

    def finish(self):
        if self.running:
            self.running = False
            self.total += round(time.time() - self.start, 2)
            elapsed = self._to_hms(self.total)
        print("<> <> <> Finished Timer [{}] <> <> <> Total time elapsed: {} <> <> <>".format(self.name, elapsed))
        return elapsed

    def _to_hms(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return "%dh %02dm %02ds" % (h, m, s)


class Dict2Obj():
    """
    将嵌套字典转换成对象，将关键字访问替换成属性访问

    >>> t = Dict2Obj()
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
                        self.__setattr__(key, Dict2Obj(value))
                    else:
                        self.__setattr__(key, value)

    def __getattr__(self, key):
        if self._is_valid(key):
            self.__setattr__(key, Dict2Obj({}))
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
            if type(value) is not Dict2Obj:
                target[key] = value
            else:
                target[key] = value.toDict()
        return target
