r"""
封装处理NLP各个任务的数据
"""
import torch
from torch.utils.data import Dataset


class DatasetCLF(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 max_len=512,
                 load_label=True,
                 **kwargs):
        """封装文本分类数据集，现在仅支持二分类和多分类，暂不支持多标签分类

        :param data: 标准数据格式为List[List[str, int]]，例如[[’今天天气很好‘, 1], ['我心情不好', 0]]
        :param tokenizer: transformers.xxxTokenizer
        :param max_len: 分词后的文本长度上限，default=512
        :param load_label: 是否加载标签，default=True
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data
        self.load_label = load_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index][0]
        encode_inputs = self.tokenizer(text,
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')

        example = {}
        for key, value in encode_inputs.items():
            example[key] = value[0]
        if self.load_label:
            label = self.data[index][1]
            example['labels'] = torch.tensor(label, dtype=torch.long)
            return example
        return example


class DatasetGen(Dataset):
    def __init__(self, data, tokenizer, max_src_len=512, max_tgt_len=512, load_label=True):
        """封装文本生成数据

        :param data: 标准数据格式为List[List[str, str]]，例如[["北京冬奥会是什么时候？", "2022年2月4日 – 2022年2月20日"], ["今天天气怎么样？", "晴"]]。如果tgt为空，则格式为List[List[str]]
        :param tokenizer: transformers.xxxTokenizer
        :param max_src_len: 分词后的源序列长度上限，default=512
        :param max_tgt_len: 分词后的目标序列长度上限，default=512
        :param load_label: 是否加载目标序列，default=True
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.load_label = load_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        src = self.data[index][0]
        encoded_src = self.tokenizer(src,
                                    truncation=True,
                                    max_length=self.max_src_len)
        if self.load_label:
            tgt = self.data[index][1]
            encoded_tgt = self.tokenizer(tgt,  # padding到最大长度，便于非自回归解码输入等长序列后的loss计算
                                        padding='max_length',
                                        truncation=True,
                                        max_length=self.max_tgt_len)
            return {
                'encoded_src': encoded_src,
                'encoded_tgt': encoded_tgt
            }
        return {
            'encoded_src': encoded_src
        }
