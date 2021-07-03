# 2021NAACL同传Workshop：千言 - 机器同传

* 任务简介
  * 同声传译结合了机器翻译（MT）、自动语音识别（ASR）和文本语音合成（TTS）等人工智能技术，输入streaming transcription或audio文件，输出翻译结果。本次赛题包括中英、英西两个方向共3项任务，参赛者可以选择其中一项或多项任务参赛，1.中文->英文翻译，输入：streaming transcription；2.中文->英文翻译，输入audio文件；3.英文->西班牙文翻译，输入streaming transcription
  * 官网：https://aistudio.baidu.com/aistudio/competition/detail/62
  
* 参赛时间：2020.12~2021.4

* 数据格式

  |   流式转录 Streaming transcription   | 翻译 Translation  |
  | :----------------------------------: | :---------------: |
  |                  大                  |                   |
  |                 大家                 |                   |
  |                大家好                |  Hello everyone   |
  |               大家好！               |        ！         |
  |                  欢                  |                   |
  |                 欢迎                 |                   |
  |                欢迎大                |                   |
  |               欢迎大家               |      Welcome      |
  |              欢迎大家关              |                   |
  |             欢迎大家关注             |        to         |
  |           欢迎大家关注UNIT           |                   |
  |          欢迎大家关注UNIT对          |                   |
  |         欢迎大家关注UNIT对话         |       UNIT        |
  |        欢迎大家关注UNIT对话系        |                   |
  |       欢迎大家关注UNIT对话系统       |   dialog system   |
  |      欢迎大家关注UNIT对话系统的      |                   |
  |     欢迎大家关注UNIT对话系统的高     |                   |
  |    欢迎大家关注UNIT对话系统的高级    |                   |
  |   欢迎大家关注UNIT对话系统的高级课   |                   |
  |  欢迎大家关注UNIT对话系统的高级课程  |                   |
  | 欢迎大家关注UNIT对话系统的高级课程。 | advanced courses. |

  ps1：以流式转录文本形式输入标准转写数据，即把句子切分为多行，每行依次递增一个字，直到整句结束;

  ps2：以语音文件形式输入数据，可以使用自己的语音识别系统。

* 数据说明

  |        任务        |                             下载                             |
  | :----------------: | :----------------------------------------------------------: |
  |   中文->英文翻译   | 报名参赛https://aistudio.baidu.com/aistudio/competition/detail/44/?isFromLuge=1即可获取数据 |
  | 英文->西班牙文翻译 |                             同上                             |

  

* 竞赛方案

  |                             方案                             | 代码 |
  | :----------------------------------------------------------: | :--: |
  | [baseline](https://aistudio.baidu.com/aistudio/projectDetail/315680) |  √   |

  