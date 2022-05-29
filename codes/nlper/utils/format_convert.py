r"""
读取非标准数据集为标准输出格式
"""

import json
from codes.nlper.utils.io import Reader


def iflytek_convert(datafile, load_label=True):
    raw_data = [
        json.loads(instance) for instance in Reader().read_txt(datafile)
    ]
    target_data = []
    if load_label:
        for instance in raw_data:
            target_data.append([instance['sentence'], int(instance['label'])])
    else:
        for instance in raw_data:
            target_data.append([instance['sentence']])
    return target_data


def tnews_convert(datafile, load_label=True):
    label_convert = {
        '100': 0,
        '101': 1,
        '102': 2,
        '103': 3,
        '104': 4,
        '106': 5,
        '107': 6,
        '108': 7,
        '109': 8,
        '110': 9,
        '112': 10,
        '113': 11,
        '114': 12,
        '115': 13,
        '116': 14
    }
    raw_data = [
        json.loads(instance) for instance in Reader().read_txt(datafile)
    ]
    target_datat = []
    if load_label:
        for instance in raw_data:
            target_datat.append([
                instance['sentence'] + instance['keywords'],
                label_convert[instance['label']]
            ])
    else:
        for instance in raw_data:
            target_datat.append([instance['sentence'] + instance['keywords']])
    return target_datat


def smp2020_ewect_convert(datafile, load_label=True):
    """
    convert custom dataset to standard format, users should specialize it

    :param datafile: custom datafile
    :param load_label: 是否加载标签
    :return: list of [text, label], label should be integer
    """
    raw_data = Reader().read_table(datafile, f_type='tsv')
    target = []
    if load_label:
        for idx, row in raw_data.iterrows():
            target.append([row['text_a'], row['label']])
    else:
        for idx, row in raw_data.iterrows():
            target.append([row['text_a']])
    return target


def dureaderqg_convert(datafile, load_label=True, sep='[SEP]'):
    """将DuReaderQG数据转换成标准数据，返回list of [src, tgt]，如果没有tgt，则返回list of [src]

    :param datafile: custom datafile, train/val/test.json path
    :param load_label: 是否加载目标序列
    :param sep: answer与context之间的分隔符
    :return: list of [src, tgt] or [src]
    """
    raw_data = Reader().read_jsonl(datafile)
    target = []
    if load_label:
        for example in raw_data:
            src = example['answer'] + ' ' + sep + ' ' + example['context']
            tgt = example['question']
            target.append([src, tgt])
    else:
        for example in raw_data:
            src = example['answer'] + ' ' + sep + ' ' + example['context']
            target.append([src])
    return target