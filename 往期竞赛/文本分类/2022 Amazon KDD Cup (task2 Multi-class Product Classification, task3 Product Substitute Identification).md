# 2022 Amazon KDD Cup (task2 Multi-class Product Classification, task3 Product Substitute Identification)

## 任务简介

* task2：给定一个query和该query检索的产品列表，这项任务的目标是将每个产品分类为 Exact, Substitute, Complement, Irrelevant match.
* task3：识别query检索的产品列表中存在替代产品（二分类）
* 官网：https://www.aicrowd.com/challenges/esci-challenge-for-improving-product-search
* 时间：2022.3.15-2022.7.15

## 数据示例

**Task2 input**

| example_id | query            | product_id | query_locale |
| :--------- | :--------------- | :--------- | :----------- |
| example_1  | 11 degrees       | product0   | us           |
| example_2  | 11 degrees       | product1   | us           |
| example_3  | 針なしほっちきす | product2   | jp           |
| example_4  | 針なしほっちきす | product3   | jp           |

The metadata about each of the products will be available in `product_catalogue-v0.3.csv` which will have the following columns : `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color_name`, `product_locale`

**Task2 output**

| example_id | esci_label |
| :--------- | :--------- |
| example_1  | exact      |
| example_2  | complement |
| example_3  | irrelevant |
| example_4  | substitute |

注：exact、substitute、complement、irrelevant的类别占比分别为65.17%、21.91%、2.89%、10.04%

**Task3 input**

| example_id | query   | product  | query_locale |
| :--------- | :------ | :------- | :----------- |
| example_1  | query_1 | product0 | us           |
| example_2  | query_2 | product1 | us           |
| example_3  | query_3 | product2 | jp           |
| example_4  | query_4 | product3 | jp           |

The metadata about each of the products will be available in `product_catalogue-v0.3.csv` which will have the following columns: `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color_name`, `product_locale`

**Task3 output**

| example_id | substitute_label |
| :--------- | :--------------- |
| example_1  | no_substitute    |
| example_2  | no_substitute    |
| example_3  | substitute       |
| example_4  | substitute       |

## 数据说明

| Total         | Total     | Total        | Train      | Train     | Train        | Test       | Test      | Test         |            |
| ------------- | --------- | ------------ | ---------- | --------- | ------------ | ---------- | --------- | ------------ | ---------- |
| Language      | # Queries | # Judgements | Avg. Depth | # Queries | # Judgements | Avg. Depth | # Queries | # Judgements | Avg. Depth |
| English (US)  | 97,345    | 1,818,825    | 18.68      | 74,888    | 1,393,063    | 18.60      | 22,458    | 425,762      | 18.96      |
| Spanish (ES)  | 15,180    | 356,410      | 23.48      | 11,336    | 263,063      | 23.21      | 3,844     | 93,347       | 24.28      |
| Japanese (JP) | 18,127    | 446,053      | 24.61      | 13,460    | 327,146      | 24.31      | 4,667     | 118,907      | 25.48      |
| Overall       | 130,652   | 2,621,288    | 20.06      | 99,684    | 1,983,272    | 19.90      | 30,969    | 638,016      | 20.60      |

数据集：[下载](https://github.com/amazon-science/esci-data) 、[paper](https://arxiv.org/abs/2206.06588 ) 

注：如果要使用比赛中的数据`product_catalogue-v0.3.csv`，则需要通过以下步骤获取

```shell
# 注册AICrowd账号，https://www.aicrowd.com/
# 安装aicrowd-cli包
pip install aicrowd-cli
# 账号授权
aicrowd login
# 下载数据
aicrowd dataset download -c esci-challenge-for-improving-product-search
```



## 竞赛方案

| task2 rank                                                   | task2 micro F1 | task3 rank | task3 F1 | 代码                                                         |
| ------------------------------------------------------------ | -------------- | ---------- | -------- | ------------------------------------------------------------ |
| [1](https://discourse.aicrowd.com/t/my-solution-good-good-study-day-day-up/7965) | 0.8326         | 1          | 0.8790   | ×                                                            |
| [2](https://discourse.aicrowd.com/t/ets-lab-our-solution/7961) ([paper](https://arxiv.org/abs/2208.00108) ) | 0.8325         | 2          | 0.8771   | [training code](https://github.com/wufanyou/KDD-Cup-2022-Amazon) <br />[code summission](https://gitlab.aicrowd.com/wufanyou/kdd_task_2) |
| [3](https://discourse.aicrowd.com/t/uni-our-competition-experience/7967) | 0.8273         | 3          | 0.8754   | [√](https://github.com/FizzerYu/AmazoneKddCup2022Top3)       |
| [7](https://discourse.aicrowd.com/t/solution-zhichunroad-5th-task1-7th-task2-and-8th-task3/8006) ([paper](https://amazonkddcup.github.io/slides/4952.pdf) ) | 0.8194         | 8          | 0.8686   | [√](https://github.com/cuixuage/KDDCup2022-ESCI)             |
| ？                                                           | ？             | -          | -        | [√](https://github.com/guijiql/kddcup2022)                   |
| [baseline](https://github.com/amazon-science/esci-data)      | 0.62           | -          | 0.76     | [√](https://github.com/amazon-science/esci-data)             |



## 推荐资料

评测论文：https://amazonkddcup.github.io/#papers