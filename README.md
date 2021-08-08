# NLPer-Arsenal

NLP人军火库，主要收录NLP竞赛经验贴、通用工具、学习资料等，如果对你有帮助，请给我们一个star，这是我们更新的动力。

本项目源于2020年7月一次竞赛的经历，当时在找参考资料时遇到了很多困难，包括内容分散、质量不高等。2021年3月开始更新本项目，志在帮助NLPer提升模型性能。2021年6月开放本项目的notion页面，[NLPer-Arsenal-Notion](https://www.notion.so/jjding/NLPer-Arsenal-Notion-9bc5e807983a47e6a2bd37afb6e3442d) ，主要收录我们整理的trick说明与实验验证，内容实时更新，欢迎大家一起参与NLPer-Arsenal开源项目。

下图是我们的项目导航图，以竞赛流程为主干，项目章节和notion对应的内容为分支。当您查看本项目时可以按序查看竞赛流程对应的项目章节，同时您也可以在我们的notion中查看一些竞赛相关的内容。
![NLPer-Arsenal-Guide](./assets/nlper-arsenal-guide.svg?raw=true)
为了帮助您快速地了解本项目的目录结构，我们上传了如下的思维导图，您可以在[链接](https://www.processon.com/view/link/6102914b5653bb3ddc1bee00) 处查看更多的细节。
<!-- 在每一对应的章节您可以查看更加详细的思维导图。 -->
![NLPer-Arsenal-mind-map](./assets/NLPer-Arsenal-mind-map.svg?raw=true)
<!-- ![NLPer-Arsenal-mind-map](./assets/NLPer-Arsenal-mind-map-full.svg?raw=true) -->





项目正在不断完善，如果您有什么建议，欢迎到[issue](https://github.com/TingFree/NLPer-Arsenal/issues) 留言，或通过邮箱（receive@nlper-arsenal.cn）联系我们。

所有内容均由我们从网络公开资料中收集得到，版权归原作者所有，如有侵权请立即与我们联系，我们将及时处理。

整理不易，转载时请务必备注本项目github链接，感谢您为维护良好的创作环境出一份力。

## 重要事件 

* 2021.3：开始更新项目
* 2021.6：开放notion界面，[NLPer-Arsenal-Notion](https://www.notion.so/jjding/NLPer-Arsenal-Notion-9bc5e807983a47e6a2bd37afb6e3442d) 

## 目录  

* [当前赛事](#当前赛事)
* [往期竞赛](#往期竞赛)
* [自媒体推荐](#自媒体推荐)
* [算力推荐](#算力推荐)
* [竞赛平台](#竞赛平台)
* [会议时间](#会议时间)

## 当前赛事

### 重点赛

> 记录当前正在进行的竞赛，奖金丰厚，适合有一定基础的NLPer

| 领域                        | 竞赛                                                         | 开始时间                                        | 结束时间                                           |
| --------------------------- | ------------------------------------------------------------ | ----------------------------------------------- | -------------------------------------------------- |
| CAIL2021                     | [评测主页](http://cail.cipsc.org.cn/) <br> [阅读理解](http://cail.cipsc.org.cn/task1.html?raceID=0) <br>[类案检索](http://cail.cipsc.org.cn/task2.html?raceID=1) <br>[司法考试](http://cail.cipsc.org.cn/task3.html?raceID=2) <br>[司法摘要](http://cail.cipsc.org.cn/task5.html?raceID=3) <br>[论辩理解](http://cail.cipsc.org.cn/task6.html?raceID=4) <br>[案情标签预测](http://cail.cipsc.org.cn/task8.html?raceID=6) <br>[信息抽取](http://cail.cipsc.org.cn/task9.html?raceID=7) | 2021.8                                          | 2021.12                                             |
| SMP2021                     | 评测通知：https://mp.weixin.qq.com/s/9t17lbdNIjpxzh400JODug <br> SMP2021-ECISA中文隐式情感分析评测 <br> SMP2021-EMWRT美团外卖技术评测（[商家推荐](https://www.biendata.xyz/competition/smp2021_1/) 、[菜品推荐](https://www.biendata.xyz/competition/smp2021_2/) ） <br> SMP2021对话式AI算法技术评测（小样本对话式意图识别与槽位提取、[对话式指代消解与省略恢复](https://www.biendata.xyz/competition/xiaobu/) ） | 2021.6                                          | 2021.8                                             |
| NTCIR-16                    | 官网：http://research.nii.ac.jp/ntcir/ntcir-16/tasks.html <br> 核心任务： <br> 1. [Data Search 2](https://ntcir.datasearch.jp/) （[IR](https://ntcir.datasearch.jp/subtasks/ir) 、[QA](https://ntcir.datasearch.jp/subtasks/qa) 、[UI](https://ntcir.datasearch.jp/subtasks/ui) ）<br> 2. [Dialogue Evaluation 2](http://sakailab.com/dialeval2/) <br> 3. [Investor’s and Manager’s Fine-grained Claim Detection](https://sites.google.com/nlg.csie.ntu.edu.tw/finnum3/) <br> 4. [Lifelog Access and Retrieval](http://ntcir-lifelog.computing.dcu.ie/) <br> 5. [Question Answering Lab for Political Information](https://poliinfo3.net/) <br> 6. [We Want Web 4 with CENTRE](http://sakailab.com/www4/) <br> 探索任务：<br> 1. [Reading Comprehension for Information Retrieval](http://ntcir-rcir.computing.dcu.ie/) <br> 2. [Real document-based Medical Natural Language Processing](https://sociocom.naist.jp/real-mednlp/) <br> 3. [Session Search](http://www.thuir.cn/session-search/) <br> 4. [Unbiased Learning to Ranking Evaluation Task ](http://ultre.online/) | ~                                               | 2022.2                                             |
| 中文医疗信息处理挑战榜CBLUE | 目前任务包括医学文本信息抽取（实体识别、关系抽取）、医学术语归一化、医学文本分类、医学句子关系判定和医学QA共5大类任务8个子任务，-> [官网](https://tianchi.aliyun.com/specials/promotion/2021chinesemedicalnlpleaderboardchallenge) | 现在                                            | 暂无                                               |
| 文本分类                    | [科大讯飞-2021试题标签预测挑战赛](http://challenge.xfyun.cn/topic/info?type=test-questions) <br>[科大讯飞-2021连续多语种分类挑战赛](http://challenge.xfyun.cn/topic/info?type=continuous-multilingual) | 2021.6<br>2021.6                                | 2021.10<br>2021.10                                 |
| 机器翻译                    | [CCMT2021机器翻译评测](http://sc.cipsc.org.cn/mt/conference/2021/tech-eval/) （在线评测持续到12月份）<br> [科大讯飞-2021低资源多语种文本翻译挑战赛](http://challenge.xfyun.cn/topic/info?type=multi-language-2021) <br>  [小牛翻译-领域迁移机器翻译挑战赛](http://challenge.xfyun.cn/topic/info?type=domain-migration) | 2021.4<br>2021.6<br> 2021.7                     | 2021.12<br>2021.10<br> 2021.10                     |
| 文本匹配                    | [科大讯飞-中文问题相似度挑战赛](http://challenge.xfyun.cn/topic/info?type=chinese-question-similarity) | 2021.7                                          | 2021.10                                            |
| 信息抽取                    | [科大讯飞-医疗实体及关系识别挑战赛](https://challenge.xfyun.cn/topic/info?type=medical-entity&ch=dc-web-35) | 2021.7                                          | 2021.10                                            |
| 其它                        | [科大讯飞-2021文本纠错及知识点填充挑战赛](http://challenge.xfyun.cn/topic/info?type=error-correction) <br> [中国人工智能学会-中文文本纠错比赛](http://2021aichina.caai.cn/track?id=5) <br> [华为-2021基于多模型迁移预训练文章质量判别](https://developer.huawei.com/consumer/cn/activity/digixActivity/digixdetail/201621215957378831) (赛题二) <br> [2021未来杯-人工智能知识图谱](https://ai.futurelab.tv/contest_detail/22#contest_index) <br> [2021未来杯-探索科技未来](https://ai.futurelab.tv/contest_detail/21#contest_index) (论文推荐) | 2021.6<br>2021.6<br>2021.6<br>2021.6<br> 2021.6 | 2021.10<br>2021.10<br>2021.9<br>2021.10<br> 2021.9 |

### 训练赛

> 记录长期进行的训练赛，有排行榜，方便刚入门的NLPer练手

|   领域   |                             竞赛                             |                           开始时间                           |                           结束时间                           |
| :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 文本分类 | [新闻文本分类](https://tianchi.aliyun.com/competition/entrance/531810/introduction) <br> [文本分类对抗攻击](https://tianchi.aliyun.com/competition/entrance/231762/introduction) <br> [虚假职位招聘预测](https://www.datafountain.cn/competitions/448) <br> [疫情期间互联网虚假新闻检测](https://www.datafountain.cn/competitions/422) <br> [疫情期间网民情绪识别](https://www.datafountain.cn/competitions/423) <br> [O2O商铺食品安全相关评论发现](https://www.datafountain.cn/competitions/370) <br> [互联网新闻情感分析](https://www.datafountain.cn/competitions/350) <br> [汽车行业用户观点主题及情感识别](https://www.datafountain.cn/competitions/310) <br> [影评文本情感分析](https://js.dclab.run/v2/cmptDetail.html?id=359) <br> [垃圾邮件分类](https://js.dclab.run/v2/cmptDetail.html?id=352) <br> [短文本分类大赛-图灵联邦](https://www.turingtopia.com/competitionnew/detail/6f2569bf525c4bc8a6049a52ec919aac/sketch) <br> [情感分类大赛-图灵联邦](https://www.turingtopia.com/competitionnew/detail/319f33ab29c04d9583e7f5c208dea119/sketch) <br> [医疗文本分类 - FlyAI](https://www.flyai.com/d/303) <br> [中文垃圾短信识别 - FlyAI](https://www.flyai.com/d/199) <br> [社交网站消息内容分类 - FlyAI](https://www.flyai.com/d/180) <br> [用户商场评价情感分析 - FlyAI](https://www.flyai.com/d/8) <br>  [Stanford-Sentiment-Treebank 情感分析 - FlyAI](https://www.flyai.com/d/162) <br> [COLA 英文句子可理解性分类 - FlyAI](https://www.flyai.com/d/160) <br> [今日头条新闻分类 - FlyAI](https://www.flyai.com/d/138) <br> [美国点评网站Yelp评价预测赛 - FlyAI](https://www.flyai.com/d/3) <br> [千言数据集：情感分析 - 百度AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/50) <br>  [Kaggle-Contradictory, My Dear Watson](https://www.kaggle.com/c/contradictory-my-dear-watson) <br> [Kaggle-Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/rules) | 现在 <br> 现在 <br> 现在 <br> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 每月1号<br> 每月1号<br/>现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在<br> 现在<br> 现在 | 暂无 <br> 2021.12.31 <br> 暂无 <br> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 每月27号<br> 每月27号<br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 2023.1<br> 暂无<br> 暂无 |
| 文本匹配 | [Quora-检测两个问题是否重复 - FlyAI](https://www.flyai.com/d/73) <br> [千言数据集：文本相似度 - 百度AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/45) |                       现在 <br/> 现在                        |                      暂无 <br/> 2023.1                       |
| 推荐系统 | [零基础入门推荐系统 - 新闻推荐](https://tianchi.aliyun.com/competition/entrance/531842/introduction?spm=5176.12281949.1003.21.493e2448CFpD1w) <br> [天池新人挑战赛之阿里移动推荐算法](https://tianchi.aliyun.com/competition/entrance/231522/introduction) <br> [电商用户购买行为预测](https://www.datafountain.cn/competitions/482) <br> [基于用户画像的商品推荐挑战赛](http://challenge.xfyun.cn/topic/info?type=user-portrait) (大奖赛) |             现在 <br> 现在 <br> 现在<br> 2021.6              |             暂无 <br> 暂无 <br> 暂无<br> 2021.10             |
|   问答   | [疫情政务问答助手](https://www.datafountain.cn/competitions/424) <br> [医疗智能问答 - FlyAI](https://www.flyai.com/d/305) <br> [2021心理对话问答挑战赛](https://www.flyai.com/d/319) <br>  [CommonsenseQA Dataset](https://www.biendata.xyz/competition/commonsense_qa/) <br> [OpenBookQA Dataset](https://www.biendata.xyz/competition/open_book_qa/) |         现在 <br> 现在 <br> 现在 <br> 现在<br> 现在          |    暂无  <br> 暂无<br> 暂无 <br> 2026.4.15 <br> 2026.4.15    |
| 语义解析 | [千言数据集：语义解析 - 百度AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/47) |                             现在                             |                            2023.1                            |
|   语音   |     [生活场景汉语语音识别](https://www.flyai.com/d/203)      |                             现在                             |                             暂无                             |
| 信息抽取 | [CCF BDCI 文本实体识别及关系抽取](https://www.datafountain.cn/competitions/371) |                             现在                             |                             暂无                             |
| 实体识别 |  [中文的命名实体识别 - FlyAI](https://www.flyai.com/d/174)   |                             现在                             |                             暂无                             |
| 立场检测 |  [中文微博的立场检测 - FlyAI](https://www.flyai.com/d/187)   |                             现在                             |                             暂无                             |
|   对话   | [MuTual Dataset](https://www.biendata.xyz/competition/mutual/) |                             现在                             |                          2026.4.15                           |
| Text2SQL |         [耶鲁文本转SQL](https://www.flyai.com/d/302)         |                             现在                             |                             暂无                             |
| 阅读理解 | [中文阅读理解练习赛 - FlyAI](https://www.flyai.com/d/161) <br> [RACE Dataset](https://www.biendata.xyz/competition/race/) <br> [RACE-C Dataset](https://www.biendata.xyz/competition/race_c/) <br>  [Dream Dataset](https://www.biendata.xyz/competition/dream/) <br> [C3 Dataset](https://www.biendata.xyz/competition/c3/) <br> [SciQ Dataset](https://www.biendata.xyz/competition/sciq/) <br> [LogiQA Dataset](https://www.biendata.xyz/competition/logiqa/) <br> [MCTest Dataset](https://www.biendata.xyz/competition/mctest/) | 现在<br> 现在<br> 现在<br>现在<br> 现在<br> 现在<br> 现在<br> 现在 | 暂无 <br> 2026.4.15 <br> 2026.4.15 <br>2026.4.15 <br>2026.4.15 <br>2026.4.15 <br>2026.4.15 <br>2026.4.15 |

## 往期竞赛

> 这里记录整理好的竞赛，包含数据下载以及竞赛方案

| 目录     | 赛事                                                         |
| -------- | ------------------------------------------------------------ |
| 文本分类 | [2019“技术需求”与“技术成果”项目之间关联度计算模型](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB/2019%E2%80%9C%E6%8A%80%E6%9C%AF%E9%9C%80%E6%B1%82%E2%80%9D%E4%B8%8E%E2%80%9C%E6%8A%80%E6%9C%AF%E6%88%90%E6%9E%9C%E2%80%9D%E9%A1%B9%E7%9B%AE%E4%B9%8B%E9%97%B4%E5%85%B3%E8%81%94%E5%BA%A6%E8%AE%A1%E7%AE%97%E6%A8%A1%E5%9E%8B.md) <br> [2020smp微博情绪分析评测](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB/2020smp%E5%BE%AE%E5%8D%9A%E6%83%85%E7%BB%AA%E5%88%86%E6%9E%90%E8%AF%84%E6%B5%8B%EF%BC%88EWECT%EF%BC%89.md) <br> [2020百度人工智能开源大赛-观点阅读理解任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB/2020%E7%99%BE%E5%BA%A6%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%BC%80%E6%BA%90%E5%A4%A7%E8%B5%9B-%E8%A7%82%E7%82%B9%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E4%BB%BB%E5%8A%A1.md) |
| 实体链指 | [2019CCKS中文短文本实体链指](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E5%AE%9E%E4%BD%93%E9%93%BE%E6%8C%87/ccks2019%E4%B8%AD%E6%96%87%E7%9F%AD%E6%96%87%E6%9C%AC%E5%AE%9E%E4%BD%93%E9%93%BE%E6%8C%87.md) <br> [2020CCKS面向中文短文本的实体链指任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E5%AE%9E%E4%BD%93%E9%93%BE%E6%8C%87/2020ccks%E9%9D%A2%E5%90%91%E4%B8%AD%E6%96%87%E7%9F%AD%E6%96%87%E6%9C%AC%E7%9A%84%E5%AE%9E%E4%BD%93%E9%93%BE%E6%8C%87%E4%BB%BB%E5%8A%A1.md) <br> [2020CCKS基于标题的大规模商品实体检索](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E5%AE%9E%E4%BD%93%E9%93%BE%E6%8C%87/2020ccks%E5%9F%BA%E4%BA%8E%E6%A0%87%E9%A2%98%E7%9A%84%E5%A4%A7%E8%A7%84%E6%A8%A1%E5%95%86%E5%93%81%E5%AE%9E%E4%BD%93%E6%A3%80%E7%B4%A2.md) <br> [2020千言数据集：面向中文短文本的实体链指任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E5%AE%9E%E4%BD%93%E9%93%BE%E6%8C%87/2020%E5%8D%83%E8%A8%80%E6%95%B0%E6%8D%AE%E9%9B%86%EF%BC%9A%E9%9D%A2%E5%90%91%E4%B8%AD%E6%96%87%E7%9F%AD%E6%96%87%E6%9C%AC%E7%9A%84%E5%AE%9E%E4%BD%93%E9%93%BE%E6%8C%87%E4%BB%BB%E5%8A%A1.md) |
| 实体识别 | 2019互联网金融新实体发现 <br> [2020中药说明书实体识别挑战](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB/2020%E4%B8%AD%E8%8D%AF%E8%AF%B4%E6%98%8E%E4%B9%A6%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB%E6%8C%91%E6%88%98.md) <br> [2020中文医学文本命名实体识别](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB/2020%E4%B8%AD%E6%96%87%E5%8C%BB%E5%AD%A6%E6%96%87%E6%9C%AC%E5%91%BD%E5%90%8D%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB.md) <br> 2020CCKS面向试验鉴定的名门实体识别 <br> 2021智能医疗决策 <br> 2021互联网舆情企业风险事件的识别和预警 <br> 2021海通&工商-2021互联网舆情企业风险事件的识别和预警 |
| 问题生成 | [2020中医文献问题生成挑战](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E9%97%AE%E9%A2%98%E7%94%9F%E6%88%90/2020%E4%B8%AD%E5%8C%BB%E6%96%87%E7%8C%AE%E9%97%AE%E9%A2%98%E7%94%9F%E6%88%90%E6%8C%91%E6%88%98.md) |
| 阅读理解 | [2018机器阅读理解技术竞赛](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3/2018%E6%9C%BA%E5%99%A8%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E6%8A%80%E6%9C%AF%E7%AB%9E%E8%B5%9B.md) <br/> [2020语言与智能技术竞赛：机器阅读理解任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3/2020%E8%AF%AD%E8%A8%80%E4%B8%8E%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF%E7%AB%9E%E8%B5%9B%EF%BC%9A%E6%9C%BA%E5%99%A8%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E4%BB%BB%E5%8A%A1.md) <br> 2021海华AI挑战赛·中文阅读理解 <br> 2021NLPCC语言与智能技术竞赛：机器阅读理解任务 <br> 法研杯 |
| 文本匹配 | [2019大数据挑战赛](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D/2019%E5%A4%A7%E6%95%B0%E6%8D%AE%E6%8C%91%E6%88%98%E8%B5%9B.md) <br> [2019金融信息负面及主体判定](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D/2019%E9%87%91%E8%9E%8D%E4%BF%A1%E6%81%AF%E8%B4%9F%E9%9D%A2%E5%8F%8A%E4%B8%BB%E4%BD%93%E5%88%A4%E5%AE%9A%20.md) <br> [2020“公益AI之星”挑战赛-新冠疫情相似句对判定大赛](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D/2020%E2%80%9C%E5%85%AC%E7%9B%8AAI%E4%B9%8B%E6%98%9F%E2%80%9D%E6%8C%91%E6%88%98%E8%B5%9B-%E6%96%B0%E5%86%A0%E7%96%AB%E6%83%85%E7%9B%B8%E4%BC%BC%E5%8F%A5%E5%AF%B9%E5%88%A4%E5%AE%9A%E5%A4%A7%E8%B5%9B.md) <br> [2020房产行业聊天匹配问答](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D/2020%E6%88%BF%E4%BA%A7%E8%A1%8C%E4%B8%9A%E8%81%8A%E5%A4%A9%E5%8C%B9%E9%85%8D%E9%97%AE%E7%AD%94.md) <br> 2021搜狐校园文本匹配算法大赛 <br> 2021小布助手对话短文本语义匹配 |
| 对话生成 | [2020千言：多技能对话](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E5%AF%B9%E8%AF%9D%E7%94%9F%E6%88%90/2020%E5%8D%83%E8%A8%80%EF%BC%9A%E5%A4%9A%E6%8A%80%E8%83%BD%E5%AF%B9%E8%AF%9D.md) <br> [2020语言与智能技术竞赛：面向推荐的对话任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E5%AF%B9%E8%AF%9D%E7%94%9F%E6%88%90/2020%E8%AF%AD%E8%A8%80%E4%B8%8E%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF%E7%AB%9E%E8%B5%9B%EF%BC%9A%E9%9D%A2%E5%90%91%E6%8E%A8%E8%8D%90%E7%9A%84%E5%AF%B9%E8%AF%9D%E4%BB%BB%E5%8A%A1.md) <br> 2021心理对话问答挑战赛 |
| Text2SQL | [2019中文NL2SQL挑战赛](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/Text2SQL/2019%E4%B8%AD%E6%96%87NL2SQL%E6%8C%91%E6%88%98%E8%B5%9B.md) <br/> [2020语言与智能技术竞赛：语义解析任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/Text2SQL/2020%E8%AF%AD%E8%A8%80%E4%B8%8E%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF%E7%AB%9E%E8%B5%9B%EF%BC%9A%E8%AF%AD%E4%B9%89%E8%A7%A3%E6%9E%90%E4%BB%BB%E5%8A%A1.md) |
| 问答     | 2020CCKS新冠知识图谱构建与问答评测                           |
| 信息抽取 | [2020科大讯飞事件抽取挑战赛](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E4%BF%A1%E6%81%AF%E6%8A%BD%E5%8F%96/2020%E7%A7%91%E5%A4%A7%E8%AE%AF%E9%A3%9E%E4%BA%8B%E4%BB%B6%E6%8A%BD%E5%8F%96%E6%8C%91%E6%88%98%E8%B5%9B.md) <br> [2020语言与智能技术竞赛：关系抽取任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E4%BF%A1%E6%81%AF%E6%8A%BD%E5%8F%96/2020%E8%AF%AD%E8%A8%80%E4%B8%8E%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF%E7%AB%9E%E8%B5%9B%EF%BC%9A%E5%85%B3%E7%B3%BB%E6%8A%BD%E5%8F%96%E4%BB%BB%E5%8A%A1.md) <br> [2020语言与智能技术竞赛：事件抽取任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E4%BF%A1%E6%81%AF%E6%8A%BD%E5%8F%96/2020%E8%AF%AD%E8%A8%80%E4%B8%8E%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF%E7%AB%9E%E8%B5%9B%EF%BC%9A%E4%BA%8B%E4%BB%B6%E6%8A%BD%E5%8F%96%E4%BB%BB%E5%8A%A1.md) <br> 2020SemEval-自由文本关系抽取 <br> 2020CCKS面向中文电子病历的医疗实体及事件抽取 <br> 2020CCKS面向金融领域的小样本跨类迁移事件抽取 <br> 2020CCKS面向京荣领驭的篇章级事件主体与要素抽取 <br> 2021NLPCC语言与智能技术竞赛：多形态信息抽取任务 |
| 机器翻译 | [2021NAACL同传Workshop：千言 - 机器同传](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E6%9C%BA%E5%99%A8%E7%BF%BB%E8%AF%91/2021NAACL%E5%90%8C%E4%BC%A0Workshop%EF%BC%9A%E5%8D%83%E8%A8%80%20-%20%E6%9C%BA%E5%99%A8%E5%90%8C%E4%BC%A0.md) |
| 其它     | [2020NLP中文预训练模型泛化能力挑战赛](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%BE%80%E6%9C%9F%E7%AB%9E%E8%B5%9B/%E5%85%B6%E5%AE%83/2020NLP%E4%B8%AD%E6%96%87%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B%E6%8C%91%E6%88%98%E8%B5%9B.md) <br> |

## 自媒体推荐  

> 学界、业界、理论、实践以及时事动态，NLPer都应该有所了解  

| 平台       | 主要领域 | 自媒体                                                       |
| ---------- | -------- | ------------------------------------------------------------ |
| 微信公众号 | 技术     | Coggle数据科学、DataFunTalk                                  |
|            | 行业信息 | 机器之心、机器之能、AI报道、AI前线、AI科技评论、机器学习研究组订阅 |
|            | 理论     | 科学空间、PaperWeekly、智源社区、人工智能前沿讲习、专知、AINLP、AI TIME 论道 |
| BiliBili   | 前沿论坛 | 智源社区、AITIME论道                                         |
| 网站       | 竞赛     | [Coggle数据科学](https://coggle.club/)                       |
|            | 学术     | [Paper With Code](https://paperswithcode.com/) 、[AMiner学术头条](https://www.aminer.cn/) |

## 算力推荐

> 结合个人情况使用不同的GPU平台

|                             平台                             |                            算力                            |                     价格                      |                             说明                             |
| :----------------------------------------------------------: | :--------------------------------------------------------: | :-------------------------------------------: | :----------------------------------------------------------: |
|             [BitaHub](https://www.bitahub.com/)              |                   1080Ti、Titan xp、V100                   | 1080Ti(￥0.7/h)、Titan xp(￥1/h)、V100(￥9/h) | 中科大先研院的平台，价格实惠，但一块GPU只搭配2核CPU，通过提交任务，按运行时间收取费用 |
|             [沣云平台](https://www.fenghub.com/)             |                           ML270                            |                    ￥2.8/h                    |    一站式AI计算平台，CPU可以增量配置，按运行时间收取费用     |
|               [恒源云](https://gpushare.com/)                |                   2080Ti、rtx5000、3090                    |                 ￥3/h~￥4.5/h                 | 可以搭配完整的CPU和硬盘，相比bithub有更高的自由度，目前处于推广期，有很多优惠 |
|        [并行云](https://www.paratera.com/index.html)         |                    V100、2080Ti、P100等                    |                     不明                      | 计算节点来自超算，可个性化定制CPU核数、GPU、存储空间，有非常简便的操作界面，并且提供远程linux桌面，灵活度优于以上三个平台。目前处于推广期，有很多优惠 |
|           [1024LAB](https://www.1024gpu.top/home)            | 1080Ti、P102-100、2080Ti、2080、T4、2070、P100、XP、3080等 |             ￥1/h ~ ￥6/h之间不等             | 这个是直接租用服务器的，有独立IP，使用虚拟货币DBC支付(可以用支付宝购买)，DBC汇率波动较大，请谨慎持有 |
|    [AI Studio](https://aistudio.baidu.com/aistudio/index)    |                            V100                            |                   基本免费                    | 由百度开发, 偶尔申请不到V100，主要使用PaddlePaddle框架，其它框架好像也可以用(请自行搜索使用方法) |
|        [天池DSW](https://dsw-dev.data.aliyun.com/#/)         |                            p100                            |         免费，单次限时8小时，不限次数         |              阿里的一个在线平台，运行时不能关闭              |
|     [天池实验室](https://tianchi.aliyun.com/notebook-ai)     |                            V100                            |                 免费，60h/年                  |      相比于AI Studio不限制深度学习框架，就是时间比较短       |
| [Kaggle](https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu) |                            k80                             |             免费，每周限时30小时              |                           外网访问                           |
| [Google Colab](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjlws2zvLzvAhULPnAKHUKCAQAQFjAAegQIBhAD&url=https%3A%2F%2Fcolab.research.google.com%2F&usg=AOvVaw3A5aPK2kLFzKOzb6sOckVw) |                     k80、T4、P4、P100                      |             免费，单次限时12小时              | 外网访问，无法指定具体GPU，未订阅Colab Pro用户多数时间下估计会被分配k80 |

## 竞赛平台

* [阿里天池](https://tianchi.aliyun.com/competition/gameList/activeList) ：阿里，奖金丰厚  
* [AiStudio](https://aistudio.baidu.com/aistudio/competition) ：百度
* [讯飞开发平台](http://challenge.xfyun.cn/) ：科大讯飞
* [DataFountain](https://www.datafountain.cn/competitions) ： CCF指定专业大数据及人工智能竞赛平台，有很多训练赛  
* [图灵联邦](https://www.turingtopia.com/competitionnew) ：NLP竞赛不多
* [biendata](https://www.biendata.com/) ： 国内领先的人工智能竞赛平台，包含大量NLP学术评测  
* [FlyAI-AI竞赛服务平台](https://www.flyai.com/c/nlp) ：难度分为新手、简单、中等、精英、困难，有大量GPU算力可供获取，奖金不多，但适合练手  
* [和鲸社区](https://www.kesci.com/home/competition) ：一个综合的学习平台  
* NLPCC、CCL、CCKS、SMP等会议每年都会举办相关学术评测  
* [Codalab](https://competitions.codalab.org/) ：一个可重复计算平台，很多国外的竞赛都会在上面提交代码用于检验
* [DCLab](https://www.dclab.run/index.html) ：和天池比较像
* [AI研习社](https://god.yanxishe.com/) ：很多很多NLP竞赛

## 会议时间

> [中国计算机学会推荐国际学术会议和期刊目录-2019](https://www.ccf.org.cn/ccf/contentcore/resource/download?ID=99185)  
> [中国计算机学会推荐中文科技期刊目录](https://www.ccf.org.cn/ccftjgjxskwml/)  
> [dblp](https://dblp.org)：计算机科学文献库  
> [AI会议deadline](https://aideadlin.es/?sub=ML,CV,NLP,RO,SP,DM) ：会议倒计时  
> [会议时间记录表](https://jackietseng.github.io/conference_call_for_paper/conferences.html) ：Updated by Jackie Tseng, Tsinghua Computer Vision and Intelligent Learning Lab  

|                             会议                             | 级别  |              摘要截稿              |   原文截稿    |                   审稿通知                   |      开会时间       |              说明              |
| :----------------------------------------------------------: | :---: | :--------------------------------: | :-----------: | :------------------------------------------: | :-----------------: | :----------------------------: |
| ACL([官网](https://www.2022.aclweb.org/)、[dblp](http://dblp.uni-trier.de/db/conf/acl/)) | CCF-A | 2021.11.15(roling review deadline) |   2021.1.7    |                  2022.2.20                   |   2022，5.22~5.27   |        Dublin, Ireland         |
| AAAI([官网](https://aaai.org/Conferences/AAAI-22/)、[dblp](http://dblp.uni-trier.de/db/conf/aaai/)) | CCF-A |             2021.8.30              |   2021.9.8    | 2021.10.15 (phase 1)<br> 2021.11.29 (final)  |   2022，2.22~3.1    |       Vancouver，Canada        |
| NeurIPS([官网](https://nips.cc/)、[dblp](http://dblp.uni-trier.de/db/conf/nips/)) | CCF-A |           ~~2021.5.19~~            | ~~2021.5.26~~ |                  2021.9.28                   |  2021，12.6~12.14   |             online             |
| IJCAI([官网](https://ijcai-21.org/)、[dblp](http://dblp.uni-trier.de/db/conf/ijcai/)) | CCF-A |           ~~2021.1.13~~            | ~~2021.1.20~~ |                ~~2021.4.30~~                 |   2021，8.21~8.26   |        Montreal, Canada        |
| ICML([官网](https://icml.cc/Conferences/2021)、[dblp](https://dblp.uni-trier.de/db/conf/icml/index.html)) | CCF-A |           ~~2021.1.28~~            | ~~2021.2.4~~  |                 ~~2021.5.8~~                 | ~~2021，7.18~7.24~~ |             online             |
| SIGIR([官网](https://sigir.org/sigir2021)、[dblp](https://dblp.uni-trier.de/db/conf/sigir/index.html)) | CCF-A |           ~~2021.1.28~~            | ~~2021.2.4~~  |                ~~2021.4.14~~                 | ~~2021，7.11~7.15~~ |             online             |
| WWW([官网](https://www2022.thewebconf.org/)、[dblp](https://dblp.uni-trier.de/db/conf/www/index.html)) | CCF-A |             2021.10.14             |  2021.10.21   |                  2022.1.13                   |   2022，4.25~4.29   |          Lyon，France          |
| EMNLP([官网](https://2021.emnlp.org/)、[dblp](http://dblp.uni-trier.de/db/conf/emnlp/)) | CCF-B |           ~~2021.5.10~~            | ~~2021.5.17~~ |                  2021.8.25                   |  2021，11.7~11.11   | Punta Cana, Dominican Republic |
| COLING([官网](https://coling2022.org/)、[dblp](http://dblp.uni-trier.de/db/conf/coling/)) | CCF-B |                 ?                  |       ?       |                      ?                       |  2022，10.9~10.15   |        Gyeongju, Korea         |
| CoNLL([官网](https://www.conll.org/2021)、[dblp](http://dblp.uni-trier.de/db/conf/conll)) | CCF-C |                 *                  | ~~2021.6.14~~ |                  2021.8.31                   |  2021，11.10~11.11  |         same as emnlp          |
| NLPCC([官网](http://tcci.ccf.org.cn/conference/2021/)、[dblp](https://dblp.uni-trier.de/db/conf/nlpcc/)) | CCF-C |                 *                  | ~~2021.6.8~~  |                ~~2021.7.30~~                 |  2021，10.13~10.17  |              青岛              |
| NAACL([官网](https://naacl.org/)、[dblp](http://dblp.uni-trier.de/db/conf/naacl/)) | CCF-C |                 *                  |      ？       |                      ？                      |   2022, 7.10~7.15   |      Seattle, Washington       |
|        ICONIP([官网](https://iconip2021.apnns.org/))         | CCF-C |                 *                  | ~~2021.6.30~~ |                  2021.8.31                   |  2021, 12.8~12.12   |        BALI, Indonesia         |
|         ACML([官网](http://www.acml-conf.org/2021/))         | CCF-C |                 *                  | ~~2021.7.2~~  |                  2021.9.10                   |  2021, 11.17~11.19  |             online             |
| ICLR([官网](https://iclr.cc/)、[dblp](https://dblp.uni-trier.de/db/conf/iclr/index.html)) |   *   |             2021.9.28              |   2021.10.5   | 2021.11.8~22 (rebutal)<br> 2022.1.24 (final) |   2022, 4.25~4.29   |             online             |
|              AACL([官网](https://aaclweb.org/))              |   *   |                 *                  |       *       |                      *                       |          *          |    announced at EMNLP 2021     |
| EACL([官网](https://eacl.org/)、[dblp](https://dblp.uni-trier.de/db/conf/eacl/)) |   *   |                 *                  |       *       |                      *                       |          *          |         合并至ACL2022          |
| CCL([官网](http://cips-cl.org/static/CCL2021/index.html)、[dblp](https://dblp.uni-trier.de/db/conf/cncl/)) |   *   |                 *                  | ~~2021.4.15~~ |                ~~2021.5.29~~                 |     2021.10下旬     |            呼和浩特            |
| CCKS([官网](www.sigkg.cn/ccks2021)、[dblp](https://dblp.uni-trier.de/db/conf/ccks/)) |   *   |                 *                  | ~~2021.5.25~~ |                 ~~2021.7.2~~                 |   2021，11.4~11.7   |              广州              |
| SMP([官网](https://conference.cipsc.org.cn/smp2021/)、[dblp](https://dblp.uni-trier.de/db/conf/smp/)) |   *   |                 *                  | ~~2021.5.15~~ |                ~~2021.6.20~~                 |    2021，9.3~9.5    |              北京              |
|    CCIR([官网](https://ccir2021.dlufl.edu.cn/index.html))    |   *   |                 *                  | ~~2021.6.10~~ |                 ~~2021.7.5~~                 |  2021, 10.29~10.31  |              大连              |
|   CCMT([官网](http://sc.cipsc.org.cn/mt/conference/2021/))   |   *   |                 *                  | ~~2021.6.12~~ |                ~~2021.7.10~~                 |      延期待定       |            青海西宁            |
|     WISE([官网](http://www.wise-conferences.org/2021/))      |   *   |           ~~2021.6.10~~            | ~~2021.6.17~~ |                ~~2021.6.30~~                 |  2021, 10.26~10.29  |      Melbourne, Australia      |

