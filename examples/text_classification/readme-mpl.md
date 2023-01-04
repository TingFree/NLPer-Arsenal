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

# 训练

```shell
# 单机单卡/多卡，num_processes要和gpu_ids数量相等，可以指定mixed_precision=fp16开启半精度训练
# model_checkpoint在output_dir目录下，最终模型保存在"output_dir/model_checkpoint"中
# 运行过程中所产生的缓存文件都存放在cache_dir中，运行结束后可以直接删除
# 需要指定sentence1_key，sentence2_key，label_key关键字来读取数据
# 详细参数配置在arsenal.nlper.utils.options.py
# 如果需要恢复中断的训练，指定--resume_from_checkpoint（最近的一个运行z）即可
accelerate launch --multi_gpu --num_processes=1 --gpu_ids="0" --num_machines=1 --mixed_precision=no run_mpl.py \
	--seed 42 \
	--do_train \
	--train_file path/to//train_file \
	--eval_file path/to/your/eval_file \
	--do_predict \
	--test_file path/to/your/test_file \
	--sentence1_key text \
	--label_key label \
	--cache_dir ../../data/cache_dir \
	--model_name_or_path path/to/pretrained_language_model \
	--metrics accuracy precision recall f1 \
	--target_metric f1 \
	--larger_is_better \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--gradient_accumulation_steps 1 \
	--learning_rate 3e-5 \
	--num_train_epochs 3 \
	--checkpointing_steps epoch \
	--output_dir ../../data/outputs/clf \
	--model_checkpoint final_checkpoint \
	--with_tracking \
	--report_to tensorboard
```

# 预测

```shell
# 当test_file中无标签信息时，必须手动指定`num_labels`，回归任务中该值应为1
accelerate launch --multi_gpu --num_processes=2 --gpu_ids="0,1" --num_machines=1 run_mpl.py \
	--seed 42 \
	--cache_dir ../../data/cache_dir \
	--do_predict \
	--num_labels 5(optional) \
	--per_device_eval_batch_size 8 \
	--test_file path/to/your/test_file \
	--sentence1_key text \
	--label_key label \
	--metrics accuracy precision recall f1 \
	--model_name_or_path path/to/pretrained_language_model \
	--output_dir ../../data/outputs/clf \
	--model_checkpoint final_checkpoint
```

# 更多环境

```shell
# accelerate config，配置多机多卡、DeepSpeed、Megatron-LM、Apple M1等运行环境
accelerate config
accelerate launch run_mpl.py {--arg1} {--arg2} ...
```

