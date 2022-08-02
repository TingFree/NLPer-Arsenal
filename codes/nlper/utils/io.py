r"""
对数据样例的读取与保存
"""

import os
import json
import yaml
import pandas as pd
from types import FunctionType
from typing import Union, List, Tuple


def select_kwargs(kwargs:dict, keys:Union[list, tuple, FunctionType]):
    """过滤传参，挑选指定参数，便于输入到特定函数，避免报错

    :param kwargs: 参数字典
    :param keys: 函数或指定参数
    :return: 过滤后的参数字典
    """
    if isinstance(keys, FunctionType):
        return {key: value for key, value in kwargs.items() if key in keys.__code__.co_varnames}
    return {key:value for key, value in kwargs.items() if key in keys}


class Reader:
    def check(self, filepath, delete=False):
        """检查文件是否存在，若存在且delete=True，则删除该文件

        :param filepath: 文件路径
        :param delete: 是否删除该文件
        :return:
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'不存在{filepath}, 请仔细检查该文件路径')
        elif delete:
            os.remove(filepath)

    def read_txt(self, filepath)->List[str]:
        """按行读取文件，去除样本两侧空格和换行符

        :param filepath: 文件路径
        :return: 列表
        """
        self.check(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = [line.strip() for line in f]
        print(f'{filepath} -> data, {len(data)} examples')
        return data

    def get_loader(self, filepath, batch_size=1):
        """按行读取文件，每次读取batch_size行，返回一个迭代器，自动去除末尾换行符

        :param filepath: 文件路径
        :param batch_size: 每次读取的样本数
        :return: 生成器
        """
        self.check(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            while True:
                examples = [f.readline() for _ in range(batch_size)]
                examples = [ex.strip() for ex in examples if ex]
                if len(examples) != batch_size:  # EOF
                    if examples:
                        yield examples
                    break
                if batch_size == 1:
                    yield examples[0]
                else:
                    yield examples

    def read_json(self, filepath):
        """读取一个json文件

        :param filepath: 文件路径
        :return: 类型和json文件内容相关
        """
        self.check(filepath)
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        print(f'{filepath} -> data, {len(data)} examples')
        return data

    def read_jsonl(self, filepath)->list:
        """每一行是一个json字符串，按行读取，返回列表

        :param filepath: 文件路径
        :return: 列表
        """
        self.check(filepath)
        data = []
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        print(f'{filepath} -> data, {len(data)} examples')
        return data

    def read_table(self, filepath, f_type=None, **kwargs):
        """读取表格，支持csv/tsv/xls/xlsx，如果要返回xls/xlsx中的所有页面，需要指定sheet_name=None

        :param filepath: 文件路径
        :param f_type: ['csv', 'tsv', 'xls', 'xlsx']中的一个，如果不指定，则自动识别文件名后缀
        :return: 表格
        """
        self.check(filepath)
        f_type = os.path.splitext(filepath)[-1].replace('.', '') if not f_type else f_type
        assert f_type in ['csv', 'tsv', 'xls', 'xlsx']
        if f_type == 'csv':
            csv_kwargs = select_kwargs(kwargs, pd.read_csv)
            data = pd.read_csv(filepath, sep=',', **csv_kwargs)
        elif f_type == 'tsv':
            tsv_kwargs = select_kwargs(kwargs, pd.read_csv)
            data = pd.read_csv(filepath, sep='\t', **tsv_kwargs)
        else:
            excel_kwargs = select_kwargs(kwargs, pd.read_excel)
            data = pd.read_excel(filepath, **excel_kwargs, engine='openpyxl')
        print(f'{filepath} -> data, {len(data)} examples')
        return data

    def read_yaml(self, filepath)->dict:
        """读取yaml

        :param filepath: 文件路径
        :return: 字典
        """
        self.check(filepath)
        with open(filepath, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data

    def load_nlp_data(self, file, task_name='text_clf'):
        """读取不同任务的标准数据

        :param file: data file
        :param task_name: one of ['text_clf']
        :return: target data
        """
        if task_name == 'text_clf':
            # 数据格式为每行一个样本，text, label，中间用制表符分隔
            raw_data = self.read_txt(file)
            target_data = []
            for raw_instance in raw_data:
                split_instance = raw_instance.split('\t')
                if len(split_instance) == 2:  # with label
                    split_instance[1] = int(split_instance[1])  # int(label)
                target_data.append(split_instance)
        else:
            raise ValueError(f'load {task_name} failed, we only support load text_clf data now')
        return target_data


class Writer:
    def create_parentDir(self, path:str, exist_ok=True):
        """递归创建path的父目录，例如path=L1/L2/L3/tmp.txt，则创建L1/L2/L3三级目录

        :param path: 文件地址
        :param exist_ok: True
        :return:
        """
        head, tail = os.path.split(path)
        if head and not os.path.exists(head):
            print(f'create {head} directory')
            os.makedirs(head, exist_ok=exist_ok)

    def write_txt(self, data:Union[List, Tuple], saved_path):
        """将数据逐行写入到文件

        :param data: 列表或元组
        :param saved_path: 保存路径
        :return:
        """
        self.create_parentDir(saved_path)
        with open(saved_path, 'w', encoding='utf-8') as f:
            for example in data:
                f.write(str(example) + '\n')
        print(f"data -> {saved_path}, {len(data)} examples")

    def write_json(self, data, saved_path):
        """将数据作为json保存"""
        self.create_parentDir(saved_path)
        with open(saved_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print(f"data -> {saved_path}")

    def write_jsonl(self, data:Union[List, Tuple], saved_path):
        """将数据中的每一个元素作为json保存

        :param data: 列表或元组
        :param saved_path: 保存路径
        :return:
        """
        self.create_parentDir(saved_path)
        with open(saved_path, 'w', encoding='utf-8') as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        print(f"data -> {saved_path}, {len(data)} examples")

    def write_excel(self, data:List[List], names:list, saved_path, sheet_name='default', mode='w', index=True):
        """保存数据到指定页面，当mode='w'时，其它页面会被删除，只保留当前页面，当mode='a'时，同名页面会被重命名写入。
        如果要写入多个页面，第一个页面指定mode为'w'，其它页面指定mode为'a'即可

        >>> writer = Writer()
        >>> writer.write_excel(
        >>>     data=[[1, 2], [3, 4]],
        >>>     names=['a', 'b'],
        >>>     saved_path='tmp.xlsx',
        >>>     sheet_name='s1',
        >>>     mode='w'
        >>> )
        >>> # 追加页面s2
        >>> writer.write_excel(
        >>>     data=[[5, 6], [7, 8]],
        >>>     names=['a', 'b'],
        >>>     saved_path='tmp.xlsx',
        >>>     sheet_name='s2',
        >>>     mode='a'
        >>> )

        :param data: List[List]，每一个元素都是页面中的一行内容
        :param names: 列名列表，每一列都需要指定列名
        :param saved_path: 保存路径
        :param sheet_name: 页名
        :param mode: 'w'表示覆盖写入，'a'表示追加页面
        :param index: 是否添加行号
        :return:
        """
        self.create_parentDir(saved_path)
        assert len(data[0]) == len(names)
        pd_writer = pd.ExcelWriter(saved_path, mode=mode, engine='openpyxl')
        df = pd.DataFrame(data, columns=names)
        df.to_excel(pd_writer, sheet_name=sheet_name, index=index, encoding='utf-8')
        pd_writer.save()
        pd_writer.close()
        print(f"data -> {saved_path}, sheet: {sheet_name}, {len(data)} examples")
