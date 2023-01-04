# 数据格式

1. json文件，每行一个样本，每个样本都是一个字典，“sequence2”可以不输入，“label”可以是字符串、整数、浮点数（回归任务）。

```json
{"sequence1": xxx11, "sequence2": xxx21, "label": yyy1}
{"sequence1": xxx21, "sequence2": xxx22, "label": yyy2}
```

2. csv文件，格式如下，要求同上。

| sequence 1 | sequence 2 | label |
| :--------: | :--------: | :---: |
|   xxx11    |   xxx21    | yyy1  |
|   xxx21    |   xxx22    | yyy2  |

# 评估指标

默认Accuracy、Precision、Recall、F1

# 训练

```shell
# 单机单卡/多卡，num_processes要和gpu_ids数量相等，可以指定mixed_precision=fp16开启半精度训练
# model_checkpoint在output_dir目录下，最终模型保存在"output_dir/model_checkpoint"中

# 运行过程中所产生的缓存文件都存放在cache_dir中，运行结束后可以直接删除
# 更多详细参数设置可以参考run_base.py源代码
accelerate launch --multi_gpu --num_processes=2 --gpu_ids="0,1" --num_machines=1 --mixed_precision=no run_base.py \
	--seed 42 \
	--do_train \
	--train_file path/to/your/train_file \
	--validation_file path/to/your/validation_file \
	--do_predict \
	--test_file path/to/your/test_file \
	--cache_dir ../../data/cache_dir \
	--model_name_or_path path/to/pretrained_language_model \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 3e-5 \
	--num_train_epochs 2 \
	--checkpointing_steps epoch \
	--output_dir ../../data/outputs/clf \
	--model_checkpoint final_checkpoint \
	--with_tracking \
	--report_to tensorboard
```

```shell
# 恢复中断的训练，指定--resume_from_checkpoint
accelerate launch --multi_gpu --num_processes=2 --gpu_ids="0,1" --num_machines=1 --mixed_precision=no run_base.py \
	--seed 42 \
	--do_train \
	--train_file path/to/your/train_file \
	--validation_file path/to/your/validation_file \
	--do_predict \
	--test_file path/to/your/test_file \
	--cache_dir ../../data/cache_dir \
	--model_name_or_path path/to/pretrained_language_model \
	--resume_from_checkpoint path/to/last_checkpoint \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--learning_rate 3e-5 \
	--num_train_epochs 2 \
	--checkpointing_steps epoch \
	--output_dir ../../data/outputs/clf \
	--model_checkpoint final_checkpoint \
	--with_tracking \
	--report_to tensorboard
```

# 预测

```shell
# 当test_file中无标签信息时，必须手动指定`num_labels`，回归任务中该值应为1
accelerate launch --multi_gpu --num_processes=2 --gpu_ids="0,1" --num_machines=1 run_base.py \
	--seed 42 \
	--cache_dir ../../data/cache_dir \
	--do_predict \
	--num_labels 5(optional) \
	--per_device_eval_batch_size 8 \
	--test_file path/to/your/test_file \
	--output_dir ../../data/outputs/clf \
	--model_checkpoint final_checkpoint
```

# 更多环境

```shell
# accelerate config，配置多机多卡、DeepSpeed、Megatron-LM、Apple M1等运行环境
accelerate config
accelerate launch run_base.py {--arg1} {--arg2} ...
```

