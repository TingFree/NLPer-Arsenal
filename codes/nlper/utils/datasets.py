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
        :param model_type:
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
