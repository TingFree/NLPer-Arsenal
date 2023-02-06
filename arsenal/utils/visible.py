"""
dataset explore for each task
"""
from collections import Counter
from typing import Union, List, Dict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from arsenal.utils import Reader, seed_everything


seed_everything()
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
font.set_style('normal')
font.set_size(12)


class Visual():
    """对matplotlib.pyplot的进一步封装

    >>> visual = Visual(2,2, figsize=(8,8))  # 创建一个2x2的空白图
    >>> visual.plot(x=[1,2,3], y=[1,2,3], label='line1', pos=(1,1))  # 在右下图中添加线条
    >>> visual.add_info(xlabel='x', ylabel='y', pos=(1,1))  # 在右下图中添加横纵标签
    >>> visual.show()  # 打印所有图
    >>> visual.savefig('pic.jpg')  # 保存图片
    """
    def __init__(
            self,
            nrows=1,
            ncolumns=1,
            sharex=False,
            sharey=False,
            figsize:tuple=None,
            title=None,
            wspace=0.25,
            hspace=0.7
    ):
        """ 初始化创建空白图，一共nrows行，ncolumns列个子图

        :param nrows: 每行子图数
        :param ncolumns: 每列子图数
        :param sharex: 是否共享横坐标
        :param sharey: 是否共享纵坐标
        :param figsize: 完整图大小，指定（长，宽）
        :param title: 整图标题,
        :param wspace: 子图间隔横向相对距离
        :param hspace: 子图间隔纵向相对距离
        """
        self._nrows = nrows
        self._ncolumns = ncolumns
        self._sharex = sharex
        self._sharey = sharey
        self._figsize = figsize
        self._title = title
        self._wspace = wspace
        self._hspace = hspace

        self._fig = plt.figure(figsize=figsize)
        self._axes = self._fig.subplots(nrows, ncolumns, sharex=sharex, sharey=sharey)
        self._fig.subplots_adjust(wspace=wspace, hspace=hspace)
        self._fig.suptitle(title)

    def plot(
            self, x, y, color='black', linestyle='-', marker=None, alpha=0.8, linewidth=1, label=None,
            show_value=False, fontsize=12, pos:Union[tuple, int]=(0,0), **kwargs
    ):
        """添加单条折线，通过指定不同的pos可以将该折线添加在不同子图上

        :param x: 横坐标
        :param y: 纵坐标
        :param color: 线条颜色
        :param linestyle: 线条样式，['-', '--', '-.', ':']其中之一, -:实线，--:短线，-.:短点相间，`:`:虚点线
        :param marker: 折点样式，['o', 's', '^', '*', 'p'], o:圆形，s:方形，^:上三角，p:五角
        :param alpha: 线条透明度
        :param linewidth: 线条宽度
        :param label: 线条标签
        :param pos: 元组，确定子图位置，类下标访问，从下标0开始
        :param kwargs: 其它参数
        :return:
        """
        ax = self._select_ax(pos)
        ax.plot(
            x, y, color=color, linestyle=linestyle, marker=marker, alpha=alpha, linewidth=linewidth, label=label, **kwargs
        )
        if show_value:
            for xx, yy in zip(x, y):
                ax.text(xx, yy, yy, fontdict={'fontsize':fontsize})

    def scatter(self, *args, **kwargs):
        pos = kwargs['pos'] if 'pos' in kwargs else (0,0)
        ax = self._select_ax(pos)
        ax.scatter(*args, **kwargs)

    def bar(self, x, height, *args, **kwargs):
        """
        >>> visual = Visual()
        >>> x = np.arange(5)
        >>> a = np.random.random(5)
        >>> b = np.random.random(5)
        >>> width = 0.4
        >>> visual.bar(x-0.5*width, a,  width=width, label='bar1')
        >>> visual.bar(x+0.5*width, b, width=width, label='bar2')
        >>> visual.add_info(xticks=x, xticks_label=['a', 'b', 'c', 'd','e'])
        >>> visual.show()

        :param x: 横轴值
        :param height: 纵轴值
        :param args: plt.bar其它位置参数
        :param kwargs: plt.bar其它关键字参数
        :return:
        """
        pos = kwargs.pop('pos') if 'pos' in kwargs else (0,0)
        show_value = kwargs.pop('show_value') if 'show_value' in kwargs else False
        ax = self._select_ax(pos)
        ax.bar(x, height, *args, **kwargs)
        if show_value:
            for xx, yy in zip(x, height):
                ax.text(xx, yy, yy, fontdict={'fontsize': 12})

    def add_info(
            self,
            xlabel='x',
            ylabel='y',
            title=None,
            loc='best',
            xticks=None,
            xticks_label=None,
            xlim:tuple=None,
            ylim:tuple=None,
            pos:Union[tuple, int]=(0,0)
    ):
        """ 为指定子图设置外围标签

        :param xlabel: 横轴标签
        :param ylabel: 纵轴标签
        :param title: 标题
        :param loc: 内部点线标签位置
        :param xticks: 设置横轴位置
        :param xticks_label: 设置横轴位置标签
        :param xlim: (bottom, top)，约束横轴显式范围
        :param ylim: (bottom, top)，约束纵轴显式范围
        :param pos: 子图坐标
        :return:
        """
        ax = self._select_ax(pos)
        ax.legend(loc=loc)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xticks is not None: ax.set_xticks(xticks, xticks_label)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def _select_ax(self, pos:Union[tuple, int]=(0,0)):
        """ 通过下标选择特定子图

        :param pos: 子图坐标
        :return: 子图
        """
        if self._nrows == 1 and self._ncolumns == 1:
            return self._axes
        elif isinstance(pos, int):
            return self._axes[pos]
        elif self._nrows == 1 or self._ncolumns == 1:
            return self._axes[max(pos)]
        else:
            return self._axes[pos[0], pos[1]]

    def show(self, with_clear=False):
        """ 显示图片

        :param with_clear: 是否清空图信息
        :return:
        """
        self._fig.show()
        if with_clear: self.reset()

    def savefig(self, saved_path='pic.jpg'):
        self._fig.savefig(saved_path)

    def reset(self, title:str=None):
        """清空所有图信息"""
        self._fig = plt.figure(figsize=self._figsize)
        self._axes = self._fig.subplots(
            self._nrows, self._ncolumns, sharex=self._sharex, sharey=self._sharey
        )
        self._fig.subplots_adjust(wspace=self._wspace, hspace=self._hspace)
        self._fig.suptitle(title if title else self._title)

def CharTokenizer(s: str):
    return list(s)


def WhitespaceTokenizer(s: str):
    return s.split(' ')


def len_distribution(data: Union[List[str], List[List[str]]], title: str, tokenizer=None):
    if tokenizer:
        instance_lens = [len(tokenizer(ins)) for ins in data]
    else:
        instance_lens = [len(ins) for ins in data]
    print(f'min: {np.min(instance_lens)}, max: {np.max(instance_lens)},'
          f'median: {np.median(instance_lens)}, mean: {np.mean(instance_lens):.2f}')

    # 默认只覆盖99%的数据分布
    p = 99
    boundary = np.percentile(instance_lens, p)
    print(f'set max_length to {int(boundary)} can cover {p}% data instances')
    plot_instance_lens = [len(ins[:int(boundary)]) for ins in data]

    plt.figure(figsize=(8, 5), dpi=300)
    plt.style.use(['science', 'no-latex'])
    sns.distplot(plot_instance_lens)
    # todo: add text
    # plt.text(0.8, 0.8, f'cover {p}%: {int(boundary)}')

    plt.xticks(font_properties=font)
    plt.yticks(font_properties=font)

    plt.gca().set_xlabel('instance length', font_properties=font, fontsize=16)
    plt.gca().set_ylabel('frequency', font_properties=font, fontsize=16)

    plt.title(f"length distribution of {title}", font_properties=font, fontsize=20)
    plt.show()


def label_distribution(data: List[Union[str, int]], title: str):
    label_counter = Counter(data)
    # 固定画图顺序，保持颜色一致
    sorted_item = sorted(label_counter.items(), key=lambda x: (x[0]), reverse=True)
    keys, values = zip(*sorted_item)

    plt.figure(figsize=(8, 8), dpi=300)
    plt.style.use(['science', 'no-latex'])
    plt.pie(values, labels=keys, autopct='%1.1f%%')

    plt.title(f"label distribution of {title}", font_properties=font, fontsize=20)
    plt.show()


if __name__ == '__main__':
    raw_data = Reader().read_jsonl('DuReaderQG/test.json')
    src, tgt = [], []
    for example in raw_data:
        src.append(example['answer'] + example['context'])
        tgt.append(example['question'])
    len_distribution(src, title='DuReaderQG test src')
    len_distribution(tgt, title='DuReaderQG test tgt')
