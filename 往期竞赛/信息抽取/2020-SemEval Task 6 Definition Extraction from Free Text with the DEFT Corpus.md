## 2020-SemEval Task 6: Definition Extraction from Free Text with the DEFT Corpus

* 任务简介

  * 根据给定的术语-定义对，从文本中抽取相关定义，包含三个子任务，（1）句子分类；（2）序列标注；（3）关系抽取。
  * 官网：https://competitions.codalab.org/competitions/22759#learn_the_details

* 时间：2019.8~2020.3

* 数据示例

  1. 句子分类：给定一个句子，判断该句子里是否包含定义
  2. 序列标注：根据给定的tag schema用BIO标记每个词。已知前四列，预测第五列Tag。
  3. 关系抽取：给定relation schema和序列标注结果，标记出tag之间的关系。已知前六列信息，预测Root_ID和relation。

  ![数据示例](https://pic1.zhimg.com/v2-a825e526e8bf959ea1660f1278e307d0_b.jpg?raw=true)

  * Token：句子里的单词
  * Source：标识当前句子来源于哪篇文章
  * Start/End：单词在文章中的起始位置
  * Tag：tag schema中的标签，符合BIO标注格式
  * Tag_ID：Tag标签的唯一标识，如果是O标签，则为-1
  * Root_ID：当前Tag_ID所关联的Tag_ID
  * Relation：relation schema中的关系

* 数据说明

  > 数据总共有215个文件，包含26552个句子

  | train | dev  | test |                        下载                        |
  | :---: | :--: | :--: | :------------------------------------------------: |
  |  80   |  68  |  67  | [√](https://github.com/adobe-research/deft_corpus) |

  

* 竞赛方案

  |                 task1 方案 / rank                 |   F1   |                             说明                             |                            代码                             |
  | :-----------------------------------------------: | :----: | :----------------------------------------------------------: | :---------------------------------------------------------: |
  | [5](https://aclanthology.org/2020.semeval-1.59/)  | 0.8444 |                       Multi-task BERT                        |                              ×                              |
  | [6](https://aclanthology.org/2020.semeval-1.58/)  | 0.8304 |            RoBERTa + Stochastic Weight Averaging             |                              ×                              |
  | [12](https://aclanthology.org/2020.semeval-1.96/) | 0.8077 | Joint classification and sequence labeling pre-trained model with MLP and CRF layer |                              ×                              |
  | [16](https://aclanthology.org/2020.semeval-1.44/) | 0.8007 |                BERT with two-step fine tuning                |                              ×                              |
  | [18](https://aclanthology.org/2020.semeval-1.90/) | 0.7971 |                 BERT with BiLSTM + attention                 |                              ×                              |
  | [26](https://aclanthology.org/2020.semeval-1.94/) | 0.7885 |                            XLNet                             |                              ×                              |
  | [32](https://aclanthology.org/2020.semeval-1.97/) | 0.7772 |                   RoBERTa with finetuning                    | [√](https://github.com/avramandrei/UPB-SemEval-2020-Task-6) |
  | [40](https://aclanthology.org/2020.semeval-1.93/) | 0.7593 |             BERT with fine-tuned language model              |    [√](https://github.com/dsciitism/SemEval-2020-Task-6)    |
  |                        41                         | 0.7555 |                              *                               |   [√](https://github.com/mukesh-mehta/SemEval2020-Task6)    |
  | [46](https://aclanthology.org/2020.semeval-1.91/) | 0.7109 |        FastText and ELMo embeddings with RNN ensemble        |                              ×                              |
  | [47](https://aclanthology.org/2020.semeval-1.95/) | 0.6851 | Concatenated GloVe and on-the-fly POS embeddings with BiLSTM and 1D-Conv + MaxPool layers |                              ×                              |

  |                 task2 方案 / rank                 | Macro-F1 |             说明              |                            代码                             |
  | :-----------------------------------------------: | :------: | :---------------------------: | :---------------------------------------------------------: |
  | [23](https://aclanthology.org/2020.semeval-1.59/) |  0.5233  |             BERT              |                              ×                              |
  | [27](https://aclanthology.org/2020.semeval-1.92/) |  0.4968  |          CRF tagger           |            [√](https://github.com/DFKI-NLP/defx)            |
  | [34](https://aclanthology.org/2020.semeval-1.94/) |  0.4589  |         XLNet - large         |                              ×                              |
  | [37](https://aclanthology.org/2020.semeval-1.97/) |  0.4398  | RoBERTa + CRF with finetuning | [√](https://github.com/avramandrei/UPB-SemEval-2020-Task-6) |
  |                        46                         |  0.2577  |               *               |   [√](https://github.com/mukesh-mehta/SemEval2020-Task6)    |

  |                      task3 方案 / rank                       | Macro-F1 |           说明            | 代码 |
  | :----------------------------------------------------------: | :------: | :-----------------------: | :--: |
  | 1（[知乎](https://zhuanlan.zhihu.com/p/189831468)  、[paper](https://aclanthology.org/2020.semeval-1.96/) ） |   1.0    | BERT + hand crafted rules |  ×   |
  |       [4](https://aclanthology.org/2020.semeval-1.58/)       |  0.9943  |       Random Forest       |  ×   |

  

* 推荐资料

  [官方总结](https://aclanthology.org/2020.semeval-1.41/) 