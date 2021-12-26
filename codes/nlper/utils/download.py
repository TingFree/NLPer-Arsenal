"""
用于下载本项目所使用到的数据集
"""
import os
import requests
from tqdm import tqdm
import shutil

dataset_names = {
    'text_clf': {
        'smp2020-ewect-usual':[
            "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/usual/train.tsv",
            "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/usual/dev.tsv",
            "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/usual/test.tsv",
            "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/usual/labels.json"
        ],  # UER处理后的数据
        'smp2020-ewect-virus':[
            "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/virus/train.tsv",
            "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/virus/dev.tsv",
            "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/virus/test.tsv",
            "https://github.com/dbiir/UER-py/raw/master/datasets/smp2020-ewect/virus/labels.json"
        ]  # UER处理后的数据
    }
}
supported_task_dataset_names = [task_name+'/'+dataset_name
                                for task_name in dataset_names.keys() for dataset_name in dataset_names[task_name]]


def download_dataset(task_dataset_name, cache_dir='../../data'):
    """ 下载受支持的数据集

    :param task_dataset_name: task_name/dataset_name，例如text_clf/smp2020-ewect-usual
    :param cache_dir: 保存数据的目录，会在cache_dir下新建一个dataset_name文件夹，如果已经存在，则覆盖
    :return 如果数据集存在或完整地下载好，返回True，否则返回False
    """
    task_name, dataset_name = task_dataset_name.strip().split('/')
    if task_name not in dataset_names.keys() or dataset_name not in dataset_names[task_name]:
        raise ValueError(f"{task_dataset_name} is invalid, mask sure it's format by 'task_name/dataset_name', "
                         f"and be supported in nlper/utils/download.dataset_names")
    root = os.path.join(cache_dir, dataset_name)
    # 检查数据集是否已经存在
    filenames = []
    for url in dataset_names[task_name][dataset_name]:
        filenames.append(os.path.split(url)[1])
    if os.path.isdir(root) and filenames == os.listdir(root):
        print(f'{task_dataset_name} is exist in {os.path.abspath(root)}, skip download')
        return True

    if os.path.exists(root):  # 删除cache_dir下的dataset_name同名文件夹
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)

    print(f'start to download {task_dataset_name}')
    for i, url in enumerate(dataset_names[task_name][dataset_name]):
        filename = filenames[i]
        filepath = os.path.join(root, filename)
        try:
            res = requests.get(url, stream=True, timeout=7)  # 7s
            total_length = int(res.headers.get('content-length'))
            with open(filepath, 'wb') as f:
                for trunk in tqdm(res.iter_content(chunk_size=1024), total=total_length//1024, ncols=80, desc=filename):
                    if trunk:
                        f.write(trunk)
        except:
            print(f"timeout, you can manual download dataset from: {url} to '{os.path.abspath(filepath)}'")
    download_files = os.listdir(root)
    lost = len(filenames) - len(set(filenames)&set(download_files))
    print(f"download over, {task_dataset_name} has been saved to {os.path.abspath(root)}, {lost} file lost")
    if lost == 0:
        return True
    else:
        return False
