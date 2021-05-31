# NLPer-Arsenal
NLP人军火库，主要收录NLP竞赛经验贴、通用工具、学习资料等，如果对你有帮助，请给我一个star，这是我更新的动力。

2020年7月份我参加一次竞赛，但是找参考资料的时候遇到很多困难，包括内容分散、质量不高等。比赛结束后就开始做这个项目了，但是一个人搜集整理资料实在无趣，坚持一星期后就放弃了。2021年3月份，突然发现多了2个star，没想到仅凭之前微薄的工作量也能帮到大家，能获得认可，实在令我意外与感动，所以我打算继续更新下去。

所有内容均由本人整理自网络公开资料，整理不易，转载时请务必备注本项目github链接，感谢您为维护良好的创作环境出一份力。

项目正在不断完善，如果您有什么建议，欢迎到[issue](https://github.com/TingFree/NLPer-Arsenal/issues) 留言，或者填写[**问卷**](https://www.wjx.cn/vj/rr4Zqph.aspx) 反馈。<------  问卷

2021.5.20：本项目迎来第一位合作者[@SFUMECJF](https://github.com/SFUMECJF) 

## TODO 

- [ ] 添加NLP入门、进修资料
- [ ] 添加NLP竞赛教程、组建竞赛群
- [ ] 发布竞赛baseline

## 目录  

> 1.**当前赛事**  
> 2.**竞赛收录**  
> 3.**会议时间**  
> 4.**竞赛平台**  
> 5.**自媒体推荐**  
> 6.**通用工具**  
> 7.**其它资源**  

## 1.当前赛事

### 重点赛

| 领域                        | 竞赛                                                         | 开始时间          | 结束时间            |
| --------------------------- | ------------------------------------------------------------ | ----------------- | ------------------- |
| 文本分类                    | [2021试题标签预测挑战赛](http://challenge.xfyun.cn/topic/info?type=test-questions) <br>[2021连续多语种分类挑战赛](http://challenge.xfyun.cn/topic/info?type=continuous-multilingual) | 2021.6<br> 2021.6 | 2021.10<br> 2021.10 |
| 其它                        | [2021文本纠错及知识点填充挑战赛](http://challenge.xfyun.cn/topic/info?type=error-correction) | 2021.6            | 2021.10             |
| 实体识别                    | [2021互联网舆情企业风险事件的识别和预警](http://ailab.aiwin.org.cn/competitions/48) （[赛题解读与baseline](https://mp.weixin.qq.com/s/u1VCgEP-Fn6nQ7ObHMMkSg) 识别+分类） | 2021.4            | 2021.7              |
| 机器翻译                    | [CCMT2021机器翻译评测](http://sc.cipsc.org.cn/mt/conference/2021/tech-eval/) （在线评测持续到12月份）<br>[2021低资源多语种文本翻译挑战赛](http://challenge.xfyun.cn/topic/info?type=multi-language-2021) | 2021.4<br> 2021.6 | 2021.5<br> 2021.10  |
| CCL2021                     | [任务一：跨领域句法分析](http://hlt.suda.edu.cn/index.php/CCL2021) <br> [任务二：中文空间语义理解](https://github.com/2030NLP/SpaCE2021) <br> [任务三：智能医疗对话诊疗](http://www.fudan-disc.com/sharedtask/imcs21/index.html) <br> [任务四：图文多模态幽默识别](http://cips-cl.org/static/CCL2021/cclEval/humorcomputation/index.html) <br> [任务五：中译语通-Nihao无监督汉语分词](http://114.116.55.241/sharedTask-unsupervisedCWS) | 2021.4            | 2021.7              |
| CCKS2021                    | 官网：http://sigkg.cn/ccks2021/?page_id=27 <br> **主题一：领域信息抽取** <br> 任务一：地址文本分析（[地址要素解析](https://tianchi.aliyun.com/competition/entrance/531900/introduction?spm=5176.12281976.0.0.29f7343cWptieK) 、[地址相关性](https://tianchi.aliyun.com/competition/entrance/531901/introduction) ） <br> 任务二：面向通信领域的过程类知识抽取（[事件抽取](https://www.biendata.xyz/competition/ccks_2021_cpe_1/) 、[事件共指消解](https://www.biendata.xyz/competition/ccks_2021_cpe_2/) ） <br> 任务三：网页文件中学者画像任务<br> 任务四：面向中文电子病历的医疗实体及事件抽取<br> **主题二：篇章级信息抽取**<br> 任务五：[通用细粒度事件检测](https://www.biendata.xyz/competition/ccks_2021_maven/) <br> 任务六：面向金融领域的篇章级事件抽取和事件因果关系抽取（[篇章](https://www.biendata.xyz/competition/ccks_2021_task6_1/) 、[事件](https://www.biendata.xyz/competition/ccks_2021_task6_2/) ）<br> **主题三：链接预测**<br> 任务七：[表型-药物-分子多层次知识图谱的链接预测](https://www.biendata.xyz/competition/ccks_2021_kg_link_prediction/) <br> **主题四：知识图谱构建与问答**<br> 任务八：保险领域信息抽取和运营商知识图谱推理问答<br> 任务九：[通用百科知识图谱实体类型推断](https://www.biendata.xyz/competition/ccks_2021_eti/) <br> 任务十：[面向军用无人机系统的军事垂直领域知识图谱构建](https://www.biendata.xyz/competition/ccks_2021_UAVs_KG/) <br> 任务十一：蕴含实体的中文医疗对话生成<br> 任务十二：面向中文医疗科普知识的内容理解（[阅读理解](https://www.biendata.xyz/competition/ccks_2021_tencentmedical_1/) 、[答非所问识别](https://www.biendata.xyz/competition/ccks_2021_tencentmedical_2/) ）<br> 任务十三：[生活服务领域知识图谱问答](https://www.biendata.xyz/competition/ccks_2021_ckbqa/) <br> **主题五：多模态问答**<br> 任务十四：知识增强的视频语义理解 | 2021.4            | 2021.7              |
| 中文医疗信息处理挑战榜CBLUE | 目前任务包括医学文本信息抽取（实体识别、关系抽取）、医学术语归一化、医学文本分类、医学句子关系判定和医学QA共5大类任务8个子任务，-> [官网](https://tianchi.aliyun.com/specials/promotion/2021chinesemedicalnlpleaderboardchallenge) | 现在              | 暂无                |

### 训练赛

|     领域     |                             竞赛                             |                           开始时间                           |                           结束时间                           |
| :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   文本分类   | [零基础入门NLP - 新闻文本分类](https://tianchi.aliyun.com/competition/entrance/531810/introduction) <br> [安全AI挑战者计划第三期 - 文本分类对抗攻击](https://tianchi.aliyun.com/competition/entrance/231762/introduction) <br> [虚假职位招聘预测](https://www.datafountain.cn/competitions/448) <br> [疫情期间互联网虚假新闻检测](https://www.datafountain.cn/competitions/422) <br> [疫情期间网民情绪识别](https://www.datafountain.cn/competitions/423) <br> [O2O商铺食品安全相关评论发现](https://www.datafountain.cn/competitions/370) <br> [互联网新闻情感分析](https://www.datafountain.cn/competitions/350) <br> [汽车行业用户观点主题及情感识别](https://www.datafountain.cn/competitions/310) <br> [影评文本情感分析](https://js.dclab.run/v2/cmptDetail.html?id=359) <br> [垃圾邮件分类](https://js.dclab.run/v2/cmptDetail.html?id=352) <br> [医疗文本分类 - FlyAI](https://www.flyai.com/d/303) <br> [中文垃圾短信识别 - FlyAI](https://www.flyai.com/d/199) <br> [社交网站消息内容分类 - FlyAI](https://www.flyai.com/d/180) <br> [用户商场评价情感分析 - FlyAI](https://www.flyai.com/d/8) <br>  [Stanford-Sentiment-Treebank 情感分析 - FlyAI](https://www.flyai.com/d/162) <br> [COLA 英文句子可理解性分类 - FlyAI](https://www.flyai.com/d/160) <br> [今日头条新闻分类 - FlyAI](https://www.flyai.com/d/138) <br> [美国点评网站Yelp评价预测赛 - FlyAI](https://www.flyai.com/d/3) <br> [千言数据集：情感分析 - 百度AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/50) | 现在 <br> 现在 <br> 现在 <br> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 <br/> 现在 | 暂无 <br> 2021.12.31 <br> 暂无 <br> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 暂无 <br/> 2023.1 |
|   文本匹配   | [Quora-检测两个问题是否重复 - FlyAI](https://www.flyai.com/d/73) <br> [千言数据集：文本相似度 - 百度AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/45) |                       现在 <br/> 现在                        |                      暂无 <br/> 2023.1                       |
|   推荐系统   | [零基础入门推荐系统 - 新闻推荐](https://tianchi.aliyun.com/competition/entrance/531842/introduction?spm=5176.12281949.1003.21.493e2448CFpD1w) <br> [天池新人挑战赛之阿里移动推荐算法](https://tianchi.aliyun.com/competition/entrance/231522/introduction) <br> [电商用户购买行为预测](https://www.datafountain.cn/competitions/482) <br> [基于用户画像的商品推荐挑战赛](http://challenge.xfyun.cn/topic/info?type=user-portrait) (大奖赛) |             现在 <br> 现在 <br> 现在<br> 2021.6              |             暂无 <br> 暂无 <br> 暂无<br> 2021.10             |
|     问答     | [疫情政务问答助手](https://www.datafountain.cn/competitions/424) <br> [医疗智能问答 - FlyAI](https://www.flyai.com/d/305) <br> [2021心理对话问答挑战赛](https://www.flyai.com/d/319) <br>  [CommonsenseQA Dataset](https://www.biendata.xyz/competition/commonsense_qa/) <br> [OpenBookQA Dataset](https://www.biendata.xyz/competition/open_book_qa/) |         现在 <br> 现在 <br> 现在 <br> 现在<br> 现在          |    暂无  <br> 暂无<br> 暂无 <br> 2026.4.15 <br> 2026.4.15    |
|   语义解析   | [千言数据集：语义解析 - 百度AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/47) |                             现在                             |                            2023.1                            |
| 实体关系抽取 | [文本实体识别及关系抽取](https://www.datafountain.cn/competitions/371) |                             现在                             |                             暂无                             |
|   实体识别   |  [中文的命名实体识别 - FlyAI](https://www.flyai.com/d/174)   |                             现在                             |                             暂无                             |
|   立场检测   |  [中文微博的立场检测 - FlyAI](https://www.flyai.com/d/187)   |                             现在                             |                             暂无                             |
|     对话     | [MuTual Dataset](https://www.biendata.xyz/competition/mutual/) |                             现在                             |                          2026.4.15                           |
|   阅读理解   | [中文阅读理解练习赛 - FlyAI](https://www.flyai.com/d/161) <br> [RACE Dataset](https://www.biendata.xyz/competition/race/) <br> [RACE-C Dataset](https://www.biendata.xyz/competition/race_c/) <br>  [Dream Dataset](https://www.biendata.xyz/competition/dream/) <br> [C3 Dataset](https://www.biendata.xyz/competition/c3/) <br> [SciQ Dataset](https://www.biendata.xyz/competition/sciq/) <br> [LogiQA Dataset](https://www.biendata.xyz/competition/logiqa/) <br> [MCTest Dataset](https://www.biendata.xyz/competition/mctest/) | 现在<br> 现在<br> 现在<br>现在<br> 现在<br> 现在<br> 现在<br> 现在 | 暂无 <br> 2026.4.15 <br> 2026.4.15 <br>2026.4.15 <br>2026.4.15 <br>2026.4.15 <br>2026.4.15 <br>2026.4.15 |

## 2.竞赛收录

> 这里记录已收录的所有竞赛

| 目录     | 赛事                                                         |
| -------- | ------------------------------------------------------------ |
| 文本分类 | [2020smp微博情绪分析评测](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB/smp2020%E5%BE%AE%E5%8D%9A%E6%83%85%E7%BB%AA%E5%88%86%E6%9E%90%E8%AF%84%E6%B5%8B%EF%BC%88EWECT%EF%BC%89.md) <br> [2020百度人工智能开源大赛-观点阅读理解任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB/2020%E7%99%BE%E5%BA%A6%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%BC%80%E6%BA%90%E5%A4%A7%E8%B5%9B-%E8%A7%82%E7%82%B9%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E4%BB%BB%E5%8A%A1.md) |
| 实体链指 | [2019ccks中文短文本实体链指](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%AE%9E%E4%BD%93%E9%93%BE%E6%8C%87/ccks2019%E4%B8%AD%E6%96%87%E7%9F%AD%E6%96%87%E6%9C%AC%E5%AE%9E%E4%BD%93%E9%93%BE%E6%8C%87.md) <br> 2020ccks面向中文短文本的实体链指任务 <br> [2020千言数据集：面向中文短文本的实体链指任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%AE%9E%E4%BD%93%E9%93%BE%E6%8C%87/2020%E5%8D%83%E8%A8%80%E6%95%B0%E6%8D%AE%E9%9B%86%EF%BC%9A%E9%9D%A2%E5%90%91%E4%B8%AD%E6%96%87%E7%9F%AD%E6%96%87%E6%9C%AC%E7%9A%84%E5%AE%9E%E4%BD%93%E9%93%BE%E6%8C%87%E4%BB%BB%E5%8A%A1.md) |
| 实体识别 | [2020中药说明书实体识别挑战](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB/2020%E4%B8%AD%E8%8D%AF%E8%AF%B4%E6%98%8E%E4%B9%A6%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB%E6%8C%91%E6%88%98.md) <br> [2020中文医学文本命名实体识别](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB/2020%E4%B8%AD%E6%96%87%E5%8C%BB%E5%AD%A6%E6%96%87%E6%9C%AC%E5%91%BD%E5%90%8D%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB.md) <br> 2021智能医疗决策 <br> 2021互联网舆情企业风险事件的识别和预警 |
| 问题生成 | [2020中医文献问题生成挑战](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E9%97%AE%E9%A2%98%E7%94%9F%E6%88%90/2020%E4%B8%AD%E5%8C%BB%E6%96%87%E7%8C%AE%E9%97%AE%E9%A2%98%E7%94%9F%E6%88%90%E6%8C%91%E6%88%98.md) |
| 阅读理解 | [2018机器阅读理解技术竞赛](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3/2018%E6%9C%BA%E5%99%A8%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E6%8A%80%E6%9C%AF%E7%AB%9E%E8%B5%9B.md) <br/> [2020语言与智能技术竞赛：机器阅读理解任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3/2020%E8%AF%AD%E8%A8%80%E4%B8%8E%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF%E7%AB%9E%E8%B5%9B%EF%BC%9A%E6%9C%BA%E5%99%A8%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E4%BB%BB%E5%8A%A1.md) <br> 2021海华AI挑战赛·中文阅读理解 <br> 2021NLPCC语言与智能技术竞赛：机器阅读理解任务 |
| 文本匹配 | [2019大数据挑战赛](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D/2019%E5%A4%A7%E6%95%B0%E6%8D%AE%E6%8C%91%E6%88%98%E8%B5%9B.md) <br> [2020“公益AI之星”挑战赛-新冠疫情相似句对判定大赛](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D/2020%E2%80%9C%E5%85%AC%E7%9B%8AAI%E4%B9%8B%E6%98%9F%E2%80%9D%E6%8C%91%E6%88%98%E8%B5%9B-%E6%96%B0%E5%86%A0%E7%96%AB%E6%83%85%E7%9B%B8%E4%BC%BC%E5%8F%A5%E5%AF%B9%E5%88%A4%E5%AE%9A%E5%A4%A7%E8%B5%9B.md) <br> [2020房产行业聊天匹配问答](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D/2020%E6%88%BF%E4%BA%A7%E8%A1%8C%E4%B8%9A%E8%81%8A%E5%A4%A9%E5%8C%B9%E9%85%8D%E9%97%AE%E7%AD%94.md) <br> 2021搜狐校园文本匹配算法大赛 <br> 2021小布助手对话短文本语义匹配 |
| 对话生成 | [2020千言：多技能对话](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%AF%B9%E8%AF%9D%E7%94%9F%E6%88%90/2020%E5%8D%83%E8%A8%80%EF%BC%9A%E5%A4%9A%E6%8A%80%E8%83%BD%E5%AF%B9%E8%AF%9D.md) <br> [2020语言与智能技术竞赛：面向推荐的对话任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%AF%B9%E8%AF%9D%E7%94%9F%E6%88%90/2020%E8%AF%AD%E8%A8%80%E4%B8%8E%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF%E7%AB%9E%E8%B5%9B%EF%BC%9A%E9%9D%A2%E5%90%91%E6%8E%A8%E8%8D%90%E7%9A%84%E5%AF%B9%E8%AF%9D%E4%BB%BB%E5%8A%A1.md) <br> 2021心理对话问答挑战赛 |
| Text2SQL | [2019中文NL2SQL挑战赛](https://github.com/TingFree/NLPer-Arsenal/blob/master/Text2SQL/2019%E4%B8%AD%E6%96%87NL2SQL%E6%8C%91%E6%88%98%E8%B5%9B.md) <br/> [2020语言与智能技术竞赛：语义解析任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/Text2SQL/2020%E8%AF%AD%E8%A8%80%E4%B8%8E%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF%E7%AB%9E%E8%B5%9B%EF%BC%9A%E8%AF%AD%E4%B9%89%E8%A7%A3%E6%9E%90%E4%BB%BB%E5%8A%A1.md) |
| 信息抽取 | [2020科大讯飞事件抽取挑战赛](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E4%BF%A1%E6%81%AF%E6%8A%BD%E5%8F%96/2020%E7%A7%91%E5%A4%A7%E8%AE%AF%E9%A3%9E%E4%BA%8B%E4%BB%B6%E6%8A%BD%E5%8F%96%E6%8C%91%E6%88%98%E8%B5%9B.md) <br> [2020语言与智能技术竞赛：关系抽取任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E4%BF%A1%E6%81%AF%E6%8A%BD%E5%8F%96/2020%E8%AF%AD%E8%A8%80%E4%B8%8E%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF%E7%AB%9E%E8%B5%9B%EF%BC%9A%E5%85%B3%E7%B3%BB%E6%8A%BD%E5%8F%96%E4%BB%BB%E5%8A%A1.md) <br> [2020语言与智能技术竞赛：事件抽取任务](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E4%BF%A1%E6%81%AF%E6%8A%BD%E5%8F%96/2020%E8%AF%AD%E8%A8%80%E4%B8%8E%E6%99%BA%E8%83%BD%E6%8A%80%E6%9C%AF%E7%AB%9E%E8%B5%9B%EF%BC%9A%E4%BA%8B%E4%BB%B6%E6%8A%BD%E5%8F%96%E4%BB%BB%E5%8A%A1.md) （2021.5.12前可下载数据）<br> 2020SemEval-自由文本关系抽取 <br> 2021NLPCC语言与智能技术竞赛：多形态信息抽取任务 |
| 机器翻译 | [2021NAACL同传Workshop：千言 - 机器同传](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E6%9C%BA%E5%99%A8%E7%BF%BB%E8%AF%91/2021NAACL%E5%90%8C%E4%BC%A0Workshop%EF%BC%9A%E5%8D%83%E8%A8%80%20-%20%E6%9C%BA%E5%99%A8%E5%90%8C%E4%BC%A0.md) |
| 其它     | [2020NLP中文预训练模型泛化能力挑战赛](https://github.com/TingFree/NLPer-Arsenal/blob/master/%E5%85%B6%E5%AE%83/2020NLP%E4%B8%AD%E6%96%87%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B%E6%8C%91%E6%88%98%E8%B5%9B.md) <br> |



## 3.会议时间

> [中国计算机学会推荐国际学术会议和期刊目录-2019](https://www.ccf.org.cn/ccf/contentcore/resource/download?ID=99185)  
> [中国计算机学会推荐中文科技期刊目录](https://www.ccf.org.cn/ccftjgjxskwml/)  
> [dblp](https://dblp.org)：计算机科学文献库  
> [AI会议deadline](https://aideadlin.es/?sub=ML,CV,NLP,RO,SP,DM) ：会议倒计时  
> [会议时间记录表](https://jackietseng.github.io/conference_call_for_paper/conferences.html) ：Updated by Jackie Tseng, Tsinghua Computer Vision and Intelligent Learning Lab  

|会议| 级别| 摘要截稿 | 原文截稿 | 审稿通知 | 开会时间|说明|
|:---:| :---:|  :---:|  :---:|  :---: |  :---: |  :---: |
|ACL([官网](https://2021.aclweb.org/)、[dblp](http://dblp.uni-trier.de/db/conf/acl/))|CCF-A|~~2021.1.25~~|~~2021.2.2~~|~~2021.5.5~~|2021，8.1~8.6|Bangkok, Thailand|
|AAAI([官网](https://aaai.org)、[dblp](http://dblp.uni-trier.de/db/conf/aaai/))|CCF-A|?|预计2021.9|?|2022，2.22~3.1|Vancouver，Canada|
|NeurIPS([官网](https://nips.cc/)、[dblp](http://dblp.uni-trier.de/db/conf/nips/))|CCF-A|~~2021.5.19~~|2021.5.26|2021.9.28|2021，12.6~12.14|online|
|IJCAI([官网](https://ijcai-21.org/)、[dblp](http://dblp.uni-trier.de/db/conf/ijcai/))|CCF-A|~~2021.1.13~~|~~2021.1.20~~|~~2021.4.30~~|2021，8.21~8.26|Montreal, Canada|
|ICML([官网](https://icml.cc/Conferences/2021)、[dblp](https://dblp.uni-trier.de/db/conf/icml/index.html))|CCF-A|~~2021.1.28~~|~~2021.2.4~~|~~2021.5.8~~|2021，7.18~7.24|online|
|SIGIR([官网](https://sigir.org/sigir2021)、[dblp](https://dblp.uni-trier.de/db/conf/sigir/index.html))|CCF-A|~~2021.1.28~~|~~2021.2.4~~|~~2021.4.14~~|2021，7.11~7.15|online|
|EMNLP([官网](https://2021.emnlp.org/)、[dblp](http://dblp.uni-trier.de/db/conf/emnlp/))|CCF-B|~~2021.5.10~~|~~2021.5.17~~|2021.8.25|2021，11.7~11.11|Punta Cana, Dominican Republic|
|COLING([官网](https://coling2022.org/)、[dblp](http://dblp.uni-trier.de/db/conf/coling/))|CCF-B|*|2021.7.1(?)|*|2022，10.9~10.15|Gyeongju, Korea|
|CoNLL([官网](https://www.conll.org/2021)、[dblp](http://dblp.uni-trier.de/db/conf/conll))|CCF-C|*|2021.6.14|2021.8.31|2021，11.10~11.11|same as emnlp|
|NLPCC([官网](http://tcci.ccf.org.cn/conference/2021/)、[dblp](https://dblp.uni-trier.de/db/conf/nlpcc/))|CCF-C|*|2021.6.8|*|2021，10.13~10.17|青岛|
|NAACL([官网](https://2021.naacl.org/)、[dblp](http://dblp.uni-trier.de/db/conf/naacl/))|CCF-C|*|~~2020.11.23~~|~~2021.3.10~~|2021，6.6~6.11|Mexico City, Mexico|
|ICLR([官网](https://iclr.cc/)、[dblp](https://dblp.uni-trier.de/db/conf/iclr/index.html))|*|?|预计2021.10|?|?|?|
|AACL([官网](http://aacl2020.org/))|*|*|*|*|*|今年合并至ACL|
|EACL([官网](https://2021.eacl.org/)、[dblp](https://dblp.uni-trier.de/db/conf/eacl/))|*|*|~~2020.10.7~~|~~2021.1.11~~|2021，4.19~4.23|online|
|CCL([官网](http://cips-cl.org/static/CCL2021/index.html)、[dblp](https://dblp.uni-trier.de/db/conf/cncl/))|*|*|~~2021.4.15~~|2021.5.29|2021，8.13~8.15|呼和浩特|
|CCKS([官网](www.sigkg.cn/ccks2021)、[dblp](https://dblp.uni-trier.de/db/conf/ccks/))|*|*|2021.5.25|2021.7.2|2021，8.18~8.21|广州|
|SMP([官网](https://conference.cipsc.org.cn/smp2021/)、[dblp](https://dblp.uni-trier.de/db/conf/smp/))|*|*|~~2021.5.15~~|2021.6.20|2021，9.3~9.5|北京|
|CCIR([官网](https://ccir2021.dlufl.edu.cn/index.html))|*|*|2021.6.10|2021.7.5|2021.10.29 - 2021.10.31|大连|
|CCMT([官网](http://sc.cipsc.org.cn/mt/conference/2021/))|*|*|2021.6.12|2021.7.10|2021，8.6~8.8|青海西宁|

## 4.竞赛平台
* [阿里天池](https://tianchi.aliyun.com/competition/gameList/activeList) ：阿里，奖金丰厚  
* [AiStudio](https://aistudio.baidu.com/aistudio/competition) ：百度
* [讯飞开发平台](http://challenge.xfyun.cn/) ：科大讯飞
* [DataFountain](https://www.datafountain.cn/competitions) ： CCF指定专业大数据及人工智能竞赛平台，有很多训练赛  
* [biendata](https://www.biendata.com/) ： 国内领先的人工智能竞赛平台，包含大量NLP学术评测  
* [FlyAI-AI竞赛服务平台](https://www.flyai.com/c/nlp) ：难度分为新手、简单、中等、精英、困难，有大量GPU算力可供获取，奖金不多，但适合练手  
* [和鲸社区](https://www.kesci.com/home/competition) ：一个综合的学习平台  
* NLPCC、CCL、CCKS、SMP等会议每年都会举办相关学术评测  
* [Codalab](https://competitions.codalab.org/) ：一个可重复计算平台，很多国外的竞赛都会在上面提交代码用于检验

## 5.自媒体推荐  
> 学界、业界、理论、实践以及时事动态，NLPer都应该有所了解  

| 平台       | 主要领域 | 自媒体                                                       |
| ---------- | -------- | ------------------------------------------------------------ |
| 微信公众号 | 技术     | Coggle数据科学、DataFunTalk                                  |
|            | 行业信息 | 机器之心、机器之能、AI报道、AI前线、AI科技评论、机器学习研究组订阅 |
|            | 理论     | 科学空间、PaperWeekly、智源社区、人工智能前沿讲习、专知、AINLP、AI TIME 论道 |
| BiliBili   | 前沿论坛 | 智源社区、AITIME论道                                         |

## 6.通用工具
* NLP中的数据增强，包含常见的数据扩充方法。([网页1](https://zhuanlan.zhihu.com/p/122445216)、[网页2](https://zhuanlan.zhihu.com/p/142168215))
* NLP中的对抗训练，提供插件式的pytorch实现，随拿随用。([网页](https://fyubang.com/2019/10/15/adversarial-train/))
* 混合精度训练，加速模型学习，适用于pytorch、TensorFlow、PaddlePaddle。([网页](https://mp.weixin.qq.com/s/zBtpwrQ5HtI6uzYOx5VsCQ))
## 7.其它资源
* GPU资源
> 结合个人情况使用不同的GPU平台

| 平台 | 算力 | 价格 | 说明|
| :---:|:---:|:---:|:---:|
| [BitaHub](https://www.bitahub.com/)|1080Ti、Titan xp、V100|1080Ti(￥0.7/h)、Titan xp(￥1/h)、V100(￥9/h)| 中科大先研院的平台，价格实惠，但一块GPU只搭配2核CPU，通过提交任务，按运行时间收取费用 |
| [沣云平台](https://www.fenghub.com/) |ML270|￥2.8/h| 一站式AI计算平台，CPU可以增量配置，按运行时间收取费用 |
|               [恒源云](https://gpushare.com/)                |               2080Ti、rtx5000、3090                |                 ￥3/h~￥4.5/h                 | 可以搭配完整的CPU和硬盘，相比bithub有更高的自由度，目前处于推广期，有很多优惠 |
| [并行云](https://www.paratera.com/index.html) | V100、2080Ti、P100等 | 不明 | 计算节点来自超算，可个性化定制CPU核数、GPU、存储空间，有非常简便的操作界面，并且提供远程linux桌面，灵活度优于以上三个平台。目前处于推广期，有很多优惠 |
| [1024LAB](https://www.1024gpu.top/home) |1080Ti、P102-100、2080Ti、2080、T4、2070、P100、XP、3080等|￥1/h ~ ￥6/h之间不等|这个是直接租用服务器的，有独立IP，使用虚拟货币DBC支付(可以用支付宝购买)，DBC汇率波动较大，请谨慎持有|
| [AI Studio](https://aistudio.baidu.com/aistudio/index) |V100|基本免费|由百度开发, 偶尔申请不到V100，主要使用PaddlePaddle框架，其它框架好像也可以用(请自行搜索使用方法)|
| [天池DSW](https://dsw-dev.data.aliyun.com/#/) |p100|免费，单次限时8小时，不限次数|阿里的一个在线平台，运行时不能关闭|
| [天池实验室](https://tianchi.aliyun.com/notebook-ai) |V100|免费，60h/年|相比于AI Studio不限制深度学习框架，就是时间比较短|
| [Kaggle](https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu) |k80|免费，每周限时30小时|外网访问|
| [Google Colab](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjlws2zvLzvAhULPnAKHUKCAQAQFjAAegQIBhAD&url=https%3A%2F%2Fcolab.research.google.com%2F&usg=AOvVaw3A5aPK2kLFzKOzb6sOckVw) |k80、T4、P4、P100|免费，单次限时12小时|外网访问，无法指定具体GPU，未订阅Colab Pro用户多数时间下估计会被分配k80|

* [AMiner](https://www.aminer.cn/) ：学术头条，更全面地了解学术领域
* [Paper With Code](https://paperswithcode.com/) ：除了搜索论文代码，还可以很方便地找到任务SOTA模型
* [GLUE](https://gluebenchmark.com/) ：一个多任务自然语言理解基准和分析平台,目前已经成为衡量模型在语言理解方面最为重要的评价体系之一
* [CLUE](https://www.cluebenchmarks.com/index.html) ：中文版GLUE，一个统一的测试平台，以准确评价模型的**中文**理解能力
* [TextFlint](https://github.com/textflint/textflint) ：复旦大学模型鲁棒性评测平台，快速便捷（[介绍](https://mp.weixin.qq.com/s/_-iMcA73OGcQSdMOxTRDZA) ）
* [Coggle](https://coggle.club/) ：竞赛资讯



