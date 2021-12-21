# 开发者指南

> 如有疑问，可以发起[issue](https://github.com/TingFree/NLPer-Arsenal/issues) 或 发邮件至 [receive@nlper-arsenal.cn](mailto:receive@nlper-arsenal.cn) 和我们沟通。

## 1. 数据集格式不同怎么办？

### 1.1 各任务标准数据格式

|                             Task                             |                      Data Format                       |                Description                 |
| :----------------------------------------------------------: | :----------------------------------------------------: | :----------------------------------------: |
| text_clf <br />（文本分类，支持二分类与多分类，暂不支持多标签分类） | [ [text1, label1], [text2, label2], ... ]，label为数值 | 文件格式如`data/examples/text_clf.txt`所示 |

### 1.2 在`tricks`下

> 改数据格式 or 改模型

* 修改数据格式

  * 修改自己的数据格式为标准数据格式，参考`data/examples`下的数据集示例，然后将yaml配置文本中的`use_convert`设置为`False`

  * 在`nlper/utils/format_convert.py`中添加自定义数据转换函数，要求输出数据为标准数据格式（见上表），具体可以参考该文件下`xxx_convert`的实现。然后在相应`xxx_handler.py`下的`convert_dataset`里加上`{dataset_name: convert_function}`

* 修改模型

  1. 在`nlper/models`下相应任务里实现自己的模型，然后在`nlper/models/__init__.py`中导入该模型，要求模型输入能够兼容dataloader的迭代值（默认直接将`**iter(dataloader)`直接输入到模型中），模型输出可以参考已有的模型（若输出不同，则需要修改对应`LightningXXX`的处理）；

  2. 修改yaml配置里的`whole_model`为自定义模型名称；

  3. 在`center_controller.py`下手动加载自己的数据，然后将dataloader输入`taskHandler.fit`以及`taskHandler.test`。

### 1.3 在`example`下

理解代码，自行修改数据格式或调整模型

## 2. 想在自己的模型上实验各个策略

### 2.1迁移策略

> 将各个策略移植到自己的模型上

各个trick的readme中提供了详细的策略解耦实现以及迁移说明，自行迁移即可

### 2.2 模型封装

> 将自己的模型封装到项目中，快速搭配已有的策略实现

将自己的模型封装到`nlper/models`下的相应任务里，参考1.2下的`“修改模型”`

## 3. 为NLPer-Arsenal添砖加瓦

> 提交新的模型或新的策略实现，降低复现难度。  
>
> 请详细了解`mini_pytorch_lightning`的实现，这是我们对pytorch_lightning的简化实现，也是项目的核心框架。

### 3.1 提交模型

参考`2.2 模型封装`

### 3.2 提交策略

> 假设您新实现的策略为“trick_A”

1. 在`tricks`下新建`trick_A`目录
2. 配置任务参数。建议优先使用`tricks/default_configs`中已有的配置，便于在同样的设置下比较不同策略。如果使用了新的公开数据集（train & dev & test），或者您觉得有必要使用新的配置，请在`trick_A`目录下保存您的配置文件
3. 新建`tricks/specialModels.py`，实现“trick_A”，并将其应用在各个任务中，格式可以参照已有的策略
4. 新建`tricks/README.md`，提供“trick_A”的解耦实现、迁移说明，性能验证可以交由我们来做。
5. 发起pull request



