# 2021NLPCC-FewCLUE

* 任务简介

  * 考察模型在情感分析、自然语言推理、多种文本分类、文本匹配和成语阅读理解任务上的小样本学习能力。
  * 官网：https://www.cluebenchmarks.com/NLPCC.html

* 时间：2021.4~2021.7

* 数据示例

  > 分类任务：EPRSTMT电商产品评论情感分析数据集、CSLDCP中文科学文献学科分类数据集、TNEWS今日头条中文新闻（短文本）分类数据集、IFLYTEK长文本分类数据集

  ```json
  {
      "id": 23,
      "sentence": "外包装上有点磨损，试听后感觉不错",
      "label": "Positive"
  }
  ```

  > 匹配任务：OCNLI 中文原版自然语言推理数据集、BUSTM 小布助手对话短文本匹配数据集

  ```json
  {
  	"level":"medium",
  	"sentence1":"身上裹一件工厂发的棉大衣,手插在袖筒里",
  	"sentence2":"身上至少一件衣服",
      "label":"entailment",
  	"label0":"entailment",
      "label1":"entailment",
      "label2":"entailment",
      "label3":"entailment",
      "label4":"entailment",
      "genre":"lit",
      "prem_id":
      "lit_635",
      "id":0
  }
  ```

  > 阅读理解任务：ChID 成语阅读理解填空、CSL 论文关键词识别、CLUEWSC WSC Winograd模式挑战中文版

  ```json
  {
      "id": 1421, 
      "content": "当广州憾负北控,郭士强黯然退场那一刻,CBA季后赛悬念仿佛一下就消失了,可万万没想到,就在时隔1天后,北控外援约瑟夫-杨因个人裁决案(拖欠上一家经纪公司的费用),导致被禁赛,打了马布里一个#idiom#,加上郭士强带领广州神奇逆转天津,让...", 
      "candidates": ["巧言令色", "措手不及", "风流人物", "八仙过海", "平铺直叙", "草木皆兵", "言行一致"],
      "answer": 1
  }
  ```

  

* 数据说明

  > 数据下载：https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets

  |             数据集             | train | dev  | open test | test | no label |
  | :----------------------------: | :---: | :--: | :-------: | :--: | :------: |
  |      电商产品评论情感分析      |  32   |  32  |    610    | 753  |  19565   |
  |      中文科学文献学科分类      |  536  | 536  |   1784    | 2999 |    67    |
  | 今日头条中文新闻（短文本）分类 |  240  | 240  |   2010    | 1500 |  20000   |
  |           长文本分类           |  928  | 690  |   1749    | 2279 |   7558   |
  |      中文原版自然语言推理      |  32   |  32  |   2520    | 3000 |  20000   |
  |     小布助手对话短文本匹配     |  32   |  32  |   1772    | 2000 |   4251   |
  |        成语阅读理解填空        |  42   |  42  |   2002    | 2000 |   7585   |
  |         论文关键词识别         |  32   |  32  |   2828    | 3000 |  19841   |
  |   WSC Winograd模式挑战中文版   |  32   |  32  |    976    | 290  |    0     |

  

* 竞赛方案

  > 评测总结论文：[《Few-shot Learning for Chinese NLP tasks》](https://link.springer.com/chapter/10.1007/978-3-030-88483-3_33) 

  | 方案/ rank final | 复赛 ACC |                           答辩PPT                            |                           评测论文                           |                             代码                             |
  | :--------------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  |        1         |  65.334  |                                                              | [√](https://link.springer.com/chapter/10.1007/978-3-030-88483-3_34) |                                                              |
  |        2         |  63.112  | [√](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/ppt/FewCLUE%E5%A4%8D%E8%B5%9B%E9%80%89%E6%89%8B%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88/%E5%A7%9C%E6%B1%81%E6%9F%A0%E6%AA%AC%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88.pptx) | [√](https://link.springer.com/chapter/10.1007/978-3-030-88483-3_31) |                                                              |
  |        3         |  64.525  | [√](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/ppt/FewCLUE%E5%A4%8D%E8%B5%9B%E9%80%89%E6%89%8B%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88/%E7%9A%AE%E7%9A%AE%E8%99%BE%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88.pptx) |                                                              |                                                              |
  |        4         |  60.525  | [√](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/ppt/FewCLUE%E5%A4%8D%E8%B5%9B%E9%80%89%E6%89%8B%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88/MLPfans%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88.pdf) |                                                              |                                                              |
  |        5         |  60.024  | [√](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/ppt/FewCLUE%E5%A4%8D%E8%B5%9B%E9%80%89%E6%89%8B%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88/%E5%A7%9C%E6%B1%81%E5%8F%AF%E4%B9%90%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88.pdf) |                                                              |                                                              |
  |        6         |    52    | [√](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/ppt/FewCLUE%E5%A4%8D%E8%B5%9B%E9%80%89%E6%89%8B%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88/%E4%B8%8A%E5%B1%B1%E6%B2%A1%E8%80%81%E8%99%8E%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88.pptx) |                                                              |                                                              |
  |        7         |  55.24   | [√](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/ppt/FewCLUE%E5%A4%8D%E8%B5%9B%E9%80%89%E6%89%8B%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88/%E5%8D%B7%E5%BF%83%E8%8F%9C%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88.pptx) |                                                              |                                                              |
  |        8         |  55.233  |                                                              |                                                              |                                                              |
  |        9         |  48.083  | [√](https://github.com/CLUEbenchmark/FewCLUE/blob/main/resources/ppt/FewCLUE%E5%A4%8D%E8%B5%9B%E9%80%89%E6%89%8B%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88/paht_sjtu%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88.pptx) |                                                              |                                                              |
  |     baseline     |    43    |                                                              |                                                              | [√](https://github.com/CLUEbenchmark/FewCLUE/tree/main/baselines) |

* 推荐资料

  * [FewCLUE: A Chinese Few-shot Learning Evaluation Benchmark](https://arxiv.org/abs/2107.07498) 
  * [FewCLUE: 小样本学习最新进展(EFL)及中文领域上的实践](https://meeting.tencent.com/user-center/shared-record-info?id=899d9236-9630-47c8-8b15-2caad162ecb9&is-single=true) 访问密码：8BK0wLZ8
  * [FewCLUE: 小样本学习最新进展(ADAPET)及中文领域上的实践](https://meeting.tencent.com/user-center/meeting-record/info?meeting_id=1396073604659040256&id=11129074844230672382&from=0) 访问密码：sJVuH39l
  * 更多资料详见FewCLUE的[Github页面](https://github.com/CLUEbenchmark/FewCLUE#%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99) 