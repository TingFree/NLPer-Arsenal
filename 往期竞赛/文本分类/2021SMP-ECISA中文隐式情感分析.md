# 2021SMP-ECISA中文隐式情感分析

* 任务简介

  * 我们将隐式情感定义为：“**不含有显式情感词，但表达了主观情感的语言片段**”，并将其划分为**事实型隐式情感**和**修辞型隐式情感**。其中，修辞型隐式情感又可细分为**隐喻/比喻型**、**反问型**以及**反讽型**。本次评测任务中，**仅针对隐式情感的识别与情感倾向性分类**。
  * 官网：https://github.com/sxu-nlp/ECISA2021

* 时间：2021.6~2021.8

* 数据示例

  ```xml
  <Doc ID="5">
  
  <Sentence ID="1">因为你是老太太</Sentence>
  
  <Sentence ID="2" label="1">看完了，满满的回忆，很多那个时代的元素</Sentence>
  
  </Doc>
  ```

  注：  

  1. 带有label="1"标记的标注句子，含有完整的上下文，标签为：0-不含情感，1-褒义隐式情感，2-贬义隐式情感。
  2. 一个Doc中含有多个句子，有些是没有标注的

* 数据说明

  > 本次评测A榜数据集为smp2019-ecisa数据，B榜不公开，以下为A榜数据

  |       | 句子数 | 褒义 | 贬义 | 中性 |                         下载                         |
  | :---: | :----: | :--: | :--: | :--: | :--------------------------------------------------: |
  | train | 60102  | 3828 | 3957 | 7003 | [√](https://github.com/FoVNull/ECISA/tree/main/data) |
  |  dev  | 20592  | 1233 | 1358 | 2554 |                         同上                         |
  | test  | 26483  | 919  | 979  | 1902 |                         同上                         |

  

* 竞赛方案

  |                          方案/rank                           | A榜 Macro F | B榜 Macro F |                      代码                       |
  | :----------------------------------------------------------: | :---------: | :---------: | :---------------------------------------------: |
  | 1（[ppt](https://github.com/sxu-nlp/ECISA2021/blob/main/WinningTeamProject/SMP2021-ECISA-gsdata.pdf) ） |   0.8127    |   0.8617    |                        ×                        |
  | 2（[ppt](https://github.com/sxu-nlp/ECISA2021/blob/main/WinningTeamProject/SMP2021-ECISA-BERT4EVER.pptx) ） |   0.8094    |   0.8581    |                        ×                        |
  |                              3                               |   0.7994    |   0.8523    | [√](https://github.com/Wchoward/SMP2021-ECISA)  |
  |                              4                               |   0.8187    |   0.8520    | [√](https://github.com/myeclipse/SMP2021-ECISA) |
  | 5（[ppt](https://github.com/sxu-nlp/ECISA2021/blob/main/WinningTeamProject/SMP2021-ECISA-highfive.pptx) ） |   0.8052    |   0.8468    |                        √                        |

  

* 推荐资料

  以下资料来源于：[here](https://zhuanlan.zhihu.com/p/361698109) 

  * 2008《情感词汇本体的构造》
  * 2020《基于混合神经网络的中文隐式情感分析》
  * 2020《基于图注意力神经网络的中文隐式情感分析》
  * 2020《一种融合上下文特征的中文隐式情感分析模型》
  * 2021《融合上下文信息的隐式情感句判别方法》
  * 2021《基于ERNIE2.0-BiLSTM-Attention的隐式情感分析方法》
  * 2021《融合关键对象识别与深层自注意力的Bi_LSTM情感分析模型》

