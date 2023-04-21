# 2022CCF BDCI小样本数据分类任务

## 任务简介

* 比赛方公开958条专利数据，包括专利权人、专利标题、专利摘要和分类标签，其中分类标签经过脱敏处理，共36类。要求选手设计一套算法，完成测试数据的分类任务。
* 官网：https://www.datafountain.cn/competitions/582
* 时间：2022.8.29-2022.12.20

## 数据示例

```json
{
    "id": "aaf98d6bfe1932cf1a262812ca59d1ba",
    "title": "一种测试方法及电子设备",
    "assignee": "腾讯科技(北京)有限公司",
    "abstract": "本发明提供了一种测试方法及电子设备，该方法包括：基于选取的测试任务确定目标测试用例，根据所述目标测试用例确定对应的至少一个测试步骤；从至少一个测试环境中选取得到目标测试环境，根据所述目标测试环境确定目标域名信息以及目标地址信息；基于所述目标域名信息以及所述目标地址信息，构建所述测试步骤对应的测试请求；发送测试请求至服务器侧，获取到所述服务器侧反馈的测试结果。",
    "label_id": 0
}
```



## 数据说明

|           | 样本数 | 下载                                                         |
| --------- | ------ | ------------------------------------------------------------ |
| 训练集    | 958    | [√](https://aistudio.baidu.com/aistudio/datasetdetail/167177) |
| A榜测试集 | 20839  | 同上                                                         |
| B榜测试集 | 20890  | [√](https://github.com/zhangzhao219/CCF-BDCI-fewshot-classification/blob/main/data/fewshot/testB.json) |



## 竞赛方案

| 方案/rank                                                 | A榜   | B榜   | 代码                                                         |
| --------------------------------------------------------- | ----- | ----- | ------------------------------------------------------------ |
| B榜第9                                                    | 0.652 | 0.595 | [√](https://github.com/zhangzhao219/CCF-BDCI-fewshot-classification) |
| [其它](https://mp.weixin.qq.com/s/YV7TPh6yjzKLFQi3vXLHkQ) | ？    | ？    | ×                                                            |
| baseline1                                                 | 0.501 | -     | [√](https://aistudio.baidu.com/aistudio/projectdetail/4482450) |
| baseline2                                                 | 0.556 | -     | [√](https://discussion.datafountain.cn/articles/detail/3604) |
| baseline3                                                 | 0.557 | -     | [√](https://discussion.datafountain.cn/articles/detail/2503) |
| baseline4                                                 | 0.566 | -     | [√](https://discussion.datafountain.cn/articles/detail/2513) |



## 推荐资料

无

