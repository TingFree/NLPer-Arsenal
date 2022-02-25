# 数据集介绍

未来将增加的benchmark
1. https://github.com/thu-coai/LOT-Benchmark
2. https://www.cluebenchmarks.com/index.html
3. https://gluebenchmark.com/
3. https://tianchi.aliyun.com/muge

目录

* 一、文本分类
* 二、文本生成
* 三、句法分析
* 四、阅读理解

## 一、文本分类

|       dataset       | language | num class |    type     | train  |  dev  | test  |                            source                            |
| :-----------------: | :------: | :-------: | :---------: | :----: | :---: | :---: | :----------------------------------------------------------: |
| smp2020-ewect-usual |    zh    |     6     | multi-class | 26,226 | 2,000 | 5,000 | [link](https://github.com/dbiir/UER-py/tree/master/datasets/smp2020-ewect) |
| smp2020-ewect-virus |    zh    |     6     | multi-class | 8,606  | 2,000 | 3,000 | [link](https://github.com/dbiir/UER-py/tree/master/datasets/smp2020-ewect) |



### 1. [smp2020-ewect-usual](https://smp2020ewect.github.io/) 

```json
{
    0: "neutral",
    1: "angry",
    2: "happy",
    3: "sad",
    4: "fear",
    5: "surprise"
}
```



|                            text_a                            | label |
| :----------------------------------------------------------: | :---: |
|     无论是心情多么低沉的夜晚，天光大亮后都是崭新的开始。     |   0   |
|               所以注定我这辈子是做不了商人妈蛋               |   1   |
|      小紧张~不过美美羊和圆圆媛要去看我表演，好开森啊！       |   2   |
|                帽子怎么就变绿色幸好只是试一下                |   3   |
| 尼玛吓死我了，人家剪个头发回来跟劳改犯一样短的可怕，后面什么鬼[黑线] [黑线] [黑线] [白眼] [白眼] |   4   |
|            棉花一枝独秀？难道看不见服装业的形势？            |   5   |

<table><tr>
    <td> <img src=../assets/smp2020-ewect/length_distribution_usual_train.png></td>
    <td> <img src=../assets/smp2020-ewect/length_distribution_usual_dev.png></td>
    <td> <img src=../assets/smp2020-ewect/length_distribution_usual_test.png></td>
    </tr>
    <tr>
    <td> <img src='../assets/smp2020-ewect/label_distribution_usual_train.png'> </td>
    <td> <img src='../assets/smp2020-ewect/label_distribution_usual_dev.png'> </td>
    <td> <img src='../assets/smp2020-ewect/label_distribution_usual_test.png'> </td>
    </tr>
</table>




### 2. [smp2020-ewect-virus](https://smp2020ewect.github.io/) 

```json
{
    0: "neutral",
    1: "angry",
    2: "happy",
    3: "sad",
    4: "fear",
    5: "surprise"
}
```

|                            text_a                            | label |
| :----------------------------------------------------------: | :---: |
|                      除夕夜，他们在武汉                      |   0   |
| 没错！平时说上海样样不好，出事了都往上海跑，上海都往外输出救助资源。等事情过了也不会对上海人改变看法……上海人有做什么事上不了台面了？！ |   1   |
|        [心] [鲜花] [作揖] [赞]//@针灸匠张宝旬:榜样。         |   2   |
|                        最后一句心疼了                        |   3   |
| 我们公司怎么还没发有关这次的通知啊，就希望湖北回来的员工先自行隔离两周，没事的话再来上班不然这样我也很害怕… ?? |   4   |
| 万万没想到出来玩居然是买这些东西回家！！！#新型冠状病毒##武汉加油# http://t.cn/RyhQMjB ??泰国·PhuketIsland |   5   |

<table><tr>
    <td> <img src=../assets/smp2020-ewect/length_distribution_virus_train.png></td>
    <td> <img src=../assets/smp2020-ewect/length_distribution_virus_dev.png></td>
    <td> <img src=../assets/smp2020-ewect/length_distribution_virus_test.png></td>
    </tr>
    <tr>
    <td> <img src='../assets/smp2020-ewect/label_distribution_virus_train.png'> </td>
    <td> <img src='../assets/smp2020-ewect/label_distribution_virus_dev.png'> </td>
    <td> <img src='../assets/smp2020-ewect/label_distribution_virus_test.png'> </td>
    </tr>
</table>



## 二、文本生成

### 1. DuReaderQG

官网：https://www.luge.ai/#/luge/dataDetail?id=8

任务描述：给定段落p和答案a，生成自然语言表述的问题q，且该问题符合段落和上下文的限制。

```json
{
  "context": "欠条是永久有效的,未约定还款期限的借款合同纠纷,诉讼时效自债权人主张债权之日起计算,时效为2年。 根据《中华人民共和国民法通则》第一百三十五条:向人民法院请求保护民事权利的诉讼时效期间为二年,法律另有规定的除外。 第一百三十七条:诉讼时效期间从知道或者应当知道权利被侵害时起计算。但是,从权利被侵害之日起超过二十年的,人民法院不予保护。有特殊情况的,人民法院可以延长诉讼时效期间。 第六十二条第(四)项:履行期限不明确的,债务人可以随时履行,债权人也可以随时要求履行,但应当给对方必要的准备时间。",
  "answer": "永久有效",
  "question": "欠条的有效期是多久",
  "id": 17
}
```

数据规模：

|          | train  | dev  | test |
| :------: | :----: | :--: | :--: |
|   官网   | 14,520 | 984  | 约1k |
| 重新划分 | 14,520 | 492  | 492  |

注：官网上只能下载train和dev，不包括标注的test，所以我们将dev划分为新的dev和test，便于之后的模型测试。

<table>
    <tr>
        <td><img src=../assets/DuReaderQG/length_distribution_train_src.png></td>
        <td><img src=../assets/DuReaderQG/length_distribution_val_src.png></td>
        <td><img src=../assets/DuReaderQG/length_distribution_test_src.png></td>
    </tr>
    <tr>
        <td><img src=../assets/DuReaderQG/length_distribution_train_tgt.png></td>
        <td><img src=../assets/DuReaderQG/length_distribution_val_tgt.png></td>
        <td><img src=../assets/DuReaderQG/length_distribution_test_tgt.png></td>
    </tr>
</table>



## 三、句法分析




## 四、阅读理解

### 1. 抽取型阅读理解-cmrc2018

leaderboard：https://ymcui.com/cmrc2018/

| | Train | Dev | Test |
|:---:| :---:|:---:|:---:|
| Question | 10,321 | 3,351 | 4,895 |
| Answer per Q | 1 | 3 | 3 |
| Max P tokens | 962 | 961 | 980 |
| Max Q tokens | 89 | 56 | 50 |
| Max A tokens | 100 | 85 | 92 |
| Avg P tokens | 452 | 469 | 472 |
| Avg Q tokens | 15 | 15 | 15 |
| Avg A tokens | 17 | 9 | 9 |

数据格式：

````json
{
    "version": "v1.0",
    "data":[
        {
            "title": "战国无双3",
            "id": "DEV_0", 
            "paragraphs": [
                {
                    "id": "DEV_0",
                    "context": "《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。xxx",
                    "qas":[
                        {
                            "question": "《战国无双3》是由哪两个公司合作开发的？",
                            "id": "DEV_0_QUERY_0", 
                            "answers": [
                                {
                                    "text": "光荣和ω-force",
                                    "answer_start": 11
                                },
                                ...
                            ]
                        },
                                ...
                    ]
                },
                        ...
            ]
        },
                        ...
    ]
}
````

