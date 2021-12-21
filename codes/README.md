# NLPer-Arsenal-Code


## 目录
- [NLPer-Arsenal-Code](#nlper-arsenal-code)
  - [目录](#目录)
  - [项目简介](#项目简介)
  - [项目优势](#项目优势)
  - [使用实例](#使用实例)
  - [基础模型](#基础模型)
  - [竞赛策略](#竞赛策略)
  - [代码框架](#代码框架)


## 项目简介

本项目基于pytorch生态实现NLP各个任务的baseline及不同的竞赛策略，希望可以帮助竞赛选手快速入手并取得可观的成绩。

适用对象：
- 入门者：通过`examples`文件夹快速了解NLP中的各个任务及基础的代码框架  
- 资深者：在`tricks`文件夹下挑选不同的竞赛策略并将其适配至自己的模型以提升竞赛性能  

## 项目优势

* `examples`提供NLP各任务的baseline实现，简洁明了，注释详细，帮助NLPer快速了解各个任务
* 实现了各种竞赛策略，`tricks/trick_name/README.md`中提供了详细的解耦实现，可以快速将策略应用到NLPer自己的模型中，上分利器
* 对所有竞赛策略，`tricks/trick_name/README.md`中提供了详细的实验验证，帮助NLPer快速选取合适的策略

## 任务介绍

本项目现已支持的NLP任务，以及默认的数据处理格式

|                             Task                             |                      Data Format                       |                Description                 |
| :----------------------------------------------------------: | :----------------------------------------------------: | :----------------------------------------: |
| text_classification <br />（文本分类，支持二分类与多分类，暂不支持多标签分类） | [ [text1, label1], [text2, label2], ... ]，label为数值 | 文件格式如`data/examples/text_clf.txt`所示 |



## 使用实例

> 如果想要进一步修改本项目代码，请参考`Developer.md`（开发者指南），包括设置个性化数据集、迁移竞赛策略、提交策略/模型到本项目等。

安装依赖环境

```shell
pip install -r requirements.txt
```

`examples`下的实例

```shell
cd expamples
python text_classification.py > examples.text_classification.log 2>&1 &
```

`tricks`下的实例

```shell
cd tricks
python center_controller.py 
    --whole_model BertCLF 
    --trick_name fgm 
    --task_config default_configs/text_clf_smp2020_ewect_usual.yaml > tricks.text_classification.log 2>&1 &
```

## 基础模型

本项目涉及且已代码实现的基础模型（所有实验均在1块1080Ti上进行）

|   Task   | Model Name |                          Description                           |  Acc  |
| :------: | :--------: | :------------------------------------------------------------: | :---: |
| 文本分类 |  BertCLF   | Bert + MLP，和`transformers.BertForSequenceClassification`类似 |  80   |


## 竞赛策略

本项目涉及且已实现的竞赛策略


|   Trick   |   Target   |           Description            | Text CLF | ToDo |
| :-------: | :--------: | :------------------------------: | :------: | :--: |
| eight-bit |  加速训练  | 8bit，降低显存开销，加快训练速度 |    √     |      |
|    fgm    | 增强鲁棒性 |  对抗训练，在embedding上加扰动   |    √     |      |

## 代码框架

```angular2html
.
├── data  # 数据集
│   ├── readme.md  # 数据集的详细介绍
│   └── smp2020-ewect  # 多分类数据集
├── examples  # NLP各任务实例
│   └── text_classification.py  # 文本分类
├── nlper  # 核心代码
│   ├── models
│   │   ├── io.py
│   │   └── text_clf.py
│   ├── modules
│   │   ├── metrics.py
│   │   ├── mlp.py
│   │   ├── trainer.py
│   │   └── utils.py
│   └── utils
│       ├── fn.py
│       ├── format_convert.py
│       ├── io.py
│       └── datasets.py
├── tricks  # 竞赛策略
│   ├── center_controller.py  # 加载任务配置 & trick运行
│   ├── default_configs  # 各任务默认配置
│   │   └── task_name_dataset_name.yaml  # task_name在dataset_name上的配置
│   ├── trick_name  # 策略名
│   │   ├── README.md  # 该策略的解耦实现，以及在不同任务不同数据集上的消融实验
│   │   └── specialModels.py  # 应用策略
│   ├── README.md
│   └── task_name_handler.py  # 解析task_name任务配置
├── README.md
└── requirements.txt  # 环境要求
```

