r"""
配置下载数据集
"""
import os
import warnings
import requests
import zipfile, gzip, tarfile
from tqdm import tqdm
from codes.nlper.utils.io import read_data, save_data
import random
random.seed(1000)


class BaseCorpus():
    def __init__(self, cache_dir):
        self.task_name = None  # 任务说明
        self.dataset_name = None  # 数据集名称
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)  # 递归创建缓存目录
        self.structure = {}  # 解压后的文件树形结构，无需手动指定

    def prepare_data(self):
        """下载好所有需要的文件，并处理成train_file、dev_file、test_file，处理完成，返回True，否则返回False
        """
        raise NotImplementedError

    def download_singleFile(self, url, filename=None):
        """下载单个文件至缓存目录（cache_dir）

        :param url: 远程下载地址
        :param filename: 如果指定，则重命名下载后的文件，否则自动使用url中的文件名
        :return: 如果文件存在或完整地下载好，返回True，否则False
        """
        # 如果没有指定重命名，则自动使用url中的文件名
        filename = filename if filename else os.path.split(url)[1]
        cache_dir = self.cache_dir
        # 如果文件已经存在，跳过下载
        if os.path.isdir(cache_dir) and filename in os.listdir(cache_dir):
            print(f"{filename} is exist in '{cache_dir}' dir, skip download")
            return True
        # 开始下载
        file_path = os.path.join(cache_dir, filename)
        try:
            res = requests.get(url, stream=True, timeout=7)  # 设置超时7s报错
            total_length = int(res.headers.get('content-length'))
            with open(file_path, 'wb') as f:
                for trunk in tqdm(res.iter_content(chunk_size=1024), total=total_length // 1024, ncols=80,
                                  desc=filename):
                    if trunk:
                        f.write(trunk)
            # 检查下载是否正常
            res.raise_for_status()
        except TimeoutError:
            # 超时
            warnings.warn(f"timeout, you can manual download dataset from: {url} to '{os.path.abspath(file_path)}'")
            return False
        except Exception as exc:
            # 其它状况
            warnings.warn(f"There was a problem: {exc}, please check manually")
            return False
        print(f"download {url} over, saved in '{os.path.abspath(file_path)}'")
        return True

    def download_compressedFile(self, url, filename=None, is_decompression=True, keep_alive=True):
        """下载单个压缩包至缓存目录（cache_dir）

        :param url: 远程下载链接
        :param filename: 如果指定，则重命名下载后的文件，否则自动使用url中的文件名
        :param is_decompression: 是否解压，支持解压['zip', 'tar', 'gz', 'tar.gz']格式，默认为True。暂不支持递归解压
        :param keep_alive: True：处理后保留原文件；False：处理后删除。默认为True
        :return: 处理完成，True，否则False
        """
        if not self.download_singleFile(url, filename):
            return False
        filename = filename if filename else os.path.split(url)[1]
        cache_dir = self.cache_dir
        file_path = os.path.join(cache_dir, filename)
        if is_decompression:
            split_filename = filename.split('.')
            if split_filename[-1] == 'zip':
                names = self.unzip(file_path)
            elif '.'.join(split_filename[-2:]) == 'tar.gz':
                self.ungzip(file_path)
                tmp_file_path = os.path.join(cache_dir, filename[:-3])
                names = self.untar(tmp_file_path)
                os.remove(tmp_file_path)
            elif split_filename[-1] == 'gz':  # 默认是单个文件的压缩包
                names = self.ungzip(file_path)
            elif split_filename[-1] == 'tar':  # 解包
                names = self.untar(file_path)
            else:
                warnings.warn(f"we only support decompress zip/tar/gz/tar.gz, "
                              f"and can't decompress your file {filename} now.")
                warnings.warn("decompress failed, so we keep original file")
                return False
            print(f'decompress over')
            self.names_transform(names)
            self.show_structure(self.structure)
        if not keep_alive:
            os.remove(file_path)
            print(f"delete '{file_path}'")
        return True

    def download_multiFiles(self, urls, filenames:list=None):
        """下载多个文件至缓存目录（cache_dir），允许对每个文件重命名，但要确保filenames和urls一一对应，无需重命名的可以指定为None

        :param urls: list，url列表
        :param filenames: list，重命名列表
        :return: True：所有文件都已准备好；False：至少一个文件下载失败
        """
        if not filenames:
            filenames = [None] * len(urls)
        assert len(urls) == len(filenames)
        tags = []  # 标记每个文件的状态，True：下载完成/已存在；False：下载失败
        for url, filename in zip(urls, filenames):
            tag = self.download_singleFile(url, filename)
            tags.append(tag)
        return all(tags)

    def unzip(self, file_path):
        zip_file = zipfile.ZipFile(file_path)
        for subfile in zip_file.namelist():
            zip_file.extract(subfile, self.cache_dir)
        names = zip_file.namelist()
        zip_file.close()
        return names

    def ungzip(self, file_path):
        """ 默认是单个文件的压缩包
        """
        gz_file = gzip.GzipFile(file_path, mode='rb')
        filename = os.path.split(file_path)[1]
        with open(os.path.join(self.cache_dir, filename[:-3]), 'wb') as f:  # 去掉.gz后缀
            f.write(gz_file.read())
        gz_file.close()
        return {filename[:-3]:None}

    def untar(self, file_path):
        tar_file = tarfile.TarFile(file_path)
        for subfile in tar_file.getnames():
            tar_file.extract(subfile, self.cache_dir)
        names = tar_file.getnames()
        tar_file.close()
        return names

    def names_transform(self, names):
        """将解压后得到的文件名列表格式化，记录在self.structure中

        >>> names = ['bq', 'bq/train.txt', 'bq/dev.txt', 'bq/test.txt']
        >>> self.names_transform(names)
        >>> {
        >>>     'bq': {'train.txt', 'dev.txt', 'test.txt'}
        >>> }
        :param names: 解压后的文件名列表，例如['bq', 'bq/train.txt', 'bq/dev.txt', 'bq/test.txt']
        """
        structure = self.structure
        for name in names:
            file_path = os.path.join(self.cache_dir, name)
            dirnames, basename = os.path.split(name.strip('/'))
            if dirnames:
                now_level = structure
                for dirname in dirnames.split('/'):
                    if dirname not in now_level:
                        now_level[dirname] = {}
                    now_level = now_level[dirname]
                now_level[basename] = None if os.path.isfile(file_path) else {}
            elif os.path.isdir(file_path):
                if basename not in structure:
                    structure[basename] = {}
            else:
                structure[basename] = None

    def show_structure(self, structure, n=0):
        """ 可视化文件结构

        >>> structure = {'bq':{'train.txt':None, 'dev.txt':None, 'test.txt':None}}
        >>> self.show_structure(structure)
        >>> .
        >>> |---- bq
        >>> |    |---- train.txt
        >>> |    |---- dev.txt
        >>> |    |---- test.txt
        :param structure: 类似于{'bq':{'train.txt':None, 'dev.txt':None, 'test.txt':None}}，叶子为文件时，值为None，否则为{}
        :param n: 无需手动指定，递归时用到
        """
        prefix = '|    '
        if n == 0:
            print(self.cache_dir + ':')
        for key, value in structure.items():
            if value:
                print(f"{prefix * n}{'|----'} {key}")
                self.show_structure(value, n + 1)
            elif value is {}:
                print(f"{prefix * n}{'|----'} {key}/")
            else:
                print(f"{prefix * n}{'|----'} {key}")


class Ewect20Usual(BaseCorpus):
    """
    task_name: text_clf  \n
    dataset_name: smp2020-ewect-usual  \n
    cache_dir: ../../data/smp2020-ewect-usual
    """
    def __init__(self, task_name='text_clf', dataset_name='smp2020-ewect-usual',
                 cache_dir='../../data/smp2020-ewect-usual'):
        super(Ewect20Usual, self).__init__(cache_dir)
        self.task_name = task_name
        self.dataset_name = dataset_name

    def prepare_data(self):
        train_url = "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/usual/train.tsv"
        dev_url = "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/usual/dev.tsv"
        test_url = "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/usual/test.tsv"
        label_url = "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/usual/labels.json"
        tag = self.download_multiFiles([train_url, dev_url, test_url, label_url])
        if tag:
            print(f"downloads {self.dataset_name} -> '{os.path.abspath(self.cache_dir)}' over")
        else:
            print(f"downloads {self.dataset_name} -> '{os.path.abspath(self.cache_dir)}' failed, some files lost, please check manually! source data: https://github.com/dbiir/UER-py/tree/master/datasets/smp2020-ewect/usual")
        return tag


class Ewect20Virus(BaseCorpus):
    """
    task_name: text_clf  \n
    dataset_name: smp2020-ewect-virus  \n
    cache_dir: ../../data/smp2020-ewect-virus
    """
    def __init__(self, task_name='text_clf', dataset_name='smp2020-ewect-virus',
                 cache_dir='../../data/smp2020-ewect-virus'):
        super(Ewect20Virus, self).__init__(cache_dir)
        self.task_name = task_name
        self.dataset_name = dataset_name

    def prepare_data(self):
        train_url = "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/virus/train.tsv"
        dev_url = "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/virus/dev.tsv"
        test_url = "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/virus/test.tsv"
        label_url = "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/virus/labels.json"
        tag = self.download_multiFiles([train_url, dev_url, test_url, label_url])
        if tag:
            print(f"prepare {self.dataset_name} -> '{os.path.abspath(self.cache_dir)}' over")
        else:
            print(f"prepare {self.dataset_name} -> '{os.path.abspath(self.cache_dir)}' failed, some files lost, please check manually! source data: https://github.com/dbiir/UER-py/tree/master/datasets/smp2020-ewect/virus")
        return tag


class DuReaderQG(BaseCorpus):
    """
    task_name: text_gen \n
    dataset_name: DuReaderQG \n
    cache_dir: ../../data
    """
    def __init__(self, task_name='text_gen', dataset_name='DuReaderQG', cache_dir='../../data'):
        super(DuReaderQG, self).__init__(cache_dir)
        self.task_name = task_name
        self.dataset_name = dataset_name

    def prepare_data(self):
        dataset_url = "https://dataset-bj.cdn.bcebos.com/qianyan/DuReaderQG.zip"
        tag = self.download_compressedFile(dataset_url)
        if tag:
            dev_data = read_data(os.path.join(self.cache_dir, self.dataset_name, 'dev.json'), f_type='line_json')
            random.shuffle(dev_data)
            val_data = dev_data[:492]
            test_data = dev_data[492:]
            print(f'split dev (984 sample) to val (492 sample) and test (492 sample)')
            save_data(val_data, os.path.join(self.cache_dir, self.dataset_name, 'val.json'), f_type='line_json')
            save_data(test_data, os.path.join(self.cache_dir, self.dataset_name, 'test.json'), f_type='line_json')
        else:
            print(f"prepare {self.dataset_name} -> '{os.path.abspath(self.cache_dir)}' failed, some files lost, please check manually! source data: https://dataset-bj.cdn.bcebos.com/qianyan/DuReaderQG.zip")
            print(f"when you download {self.dataset_name} manually, split dev.json (984 sample) to val.json (492 sample) "
                  f"and test.json (492 sample) with random seed 1000, code snippet as follow: \n"
                  f"random.shuffle(dev) \n"
                  f"val = dev[:492] \n"
                  f"test = dev[492:]")
        return tag

dataset_names = {
    "text_clf/smp2020-ewect-usual": Ewect20Usual,
    "text_clf/smp2020-ewect-virus": Ewect20Virus,
    "text_gen/DuReaderQG": DuReaderQG,
}
