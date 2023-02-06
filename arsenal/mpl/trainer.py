r"""
根据本项目需求对pytorch-lightning.Trainer的简易重构
"""
import os
import math
import random
from tqdm.auto import tqdm
import torch
import datasets
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedType
from arsenal.mpl import MplModule, MplLogger
from arsenal.utils import Reader, Writer, Timer


reader, writer, timer = Reader(), Writer(), Timer()


class Trainer():
    def __init__(self, args, mpl_model: MplModule):
        self.args = args
        self.mpl_model = mpl_model

        logging_dir = os.path.join(args.output_dir, 'logs')
        self.accelerator = (
            Accelerator(log_with=args.report_to, logging_dir=logging_dir)
            if args.with_tracking else Accelerator()
        )
        self.mpl_logger = MplLogger(
            name=__name__,
            accelerator=self.accelerator,
            log_path=os.path.join(args.output_dir, 'log.txt'),
            show_terminal=True,
            save2file=True
        )
        # if args.seed is not None:
        #     set_seed(args.seed)
        self.mpl_logger.log(f"trainer.init random: {random.random()}")
        self.mpl_logger.log(self.accelerator.state)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
        if self.accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(logging_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.mpl_model.prepare_data(use_fp16=self.accelerator.use_fp16)

    def train(self, **kwargs):
        """训练模型，可以指定自定义参数覆盖默认设置，参数见arsenal.nlper.utils.options"""
        self._set_mode('train')

        args = self.args
        accelerator = self.accelerator
        mpl_logger = self.mpl_logger
        mpl_model = self.mpl_model

        # 更新args参数
        for key, value in kwargs.items():
            if key in args:
                args.key = value

        # 使用accelerator内置的梯度累计，减少梯度同步次数，运行更高效
        accelerator.gradient_accumulation_steps = args.gradient_accumulation_steps
        if args.gradient_accumulation_steps > 1:
            if accelerator.state.distributed_type == DistributedType.TPU:
                raise NotImplementedError(
                    "Gradient accumulation on TPU is not supported by accelerate. "
                    "Pass in `gradient_accumulation_steps=1`"
                )

        with accelerator.main_process_first():
            train_data = mpl_model.get_train_data(
                train_file=args.train_file,
                label_key=args.label_key,
                batch_size=args.per_device_train_batch_size,
                cache_dir=args.cache_dir
            )
            eval_data = mpl_model.get_eval_data(
                eval_file=args.eval_file,
                batch_size=args.per_device_eval_batch_size,
                cache_dir=args.cache_dir
            )

        train_dataset, train_dataloader = train_data.dataset, train_data.dataloader
        eval_dataset, eval_dataloader = eval_data.dataset, eval_data.dataloader

        # 随机采样查看数据
        for index in random.sample(range(len(train_dataset)), 3):
            mpl_logger.log(f"Sample {index}(idx) of the training set: {train_dataset[index]}.")

        # 配置优化器-------
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
        optimizer, lr_scheduler = self.mpl_model.configure_optimizers(
            learning_rate=args.learning_rate,
            max_train_steps=args.max_train_steps,
            warmup_ratio=args.warmup_ratio,
            num_warmup_steps=args.num_warmup_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon
        )
        # 配置优化器-------

        mpl_model.model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            mpl_model.model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # 更新训练步数
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # 更新训练轮数
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        checkpointing_steps = args.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)

        # 打印保存超参
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["metrics"] = str(experiment_config['metrics'])
        mpl_logger.log(f"experiment config: {experiment_config}")
        if args.with_tracking:
            accelerator.init_trackers("base-trainer", experiment_config)

        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        mpl_logger.log("***** Running training *****")
        mpl_logger.log(f"  Num examples = {len(train_dataset)}")
        mpl_logger.log(f"  Num Epochs = {args.num_train_epochs}")
        mpl_logger.log(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        mpl_logger.log(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        mpl_logger.log(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        mpl_logger.log(f"  Total optimization steps = {args.max_train_steps}")

        # 记录全局训练开始时间
        running_start = timer.get_cur_time()

        progress_bar = tqdm(
            range(args.max_train_steps),
            desc="train",
            disable=not accelerator.is_local_main_process
        )

        completed_steps = 0
        starting_epoch = 0
        # 继续之前的训练
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None:
                mpl_logger.log(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(args.output_dir)
                        if f.is_dir() and (f.name.startswith('epoch_') or f.name.startswith('step_'))]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                args.resume_from_checkpoint = os.path.join(args.output_dir, path)
                mpl_logger.log(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                progress_bar.update(starting_epoch * len(train_dataloader))
                resume_step = None
            else:
                resume_step = int(training_difference.replace("step_", ""))
                progress_bar.update(resume_step)
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)

        larger_is_better = args.larger_is_better
        best_score = float('-inf') if larger_is_better else float('inf')
        # 记录最佳模型，在指定early stop的情况下，最终保存的模型就是最佳模型
        best_model_path = None
        total_loss = 0
        # 标识是否提前结束训练
        training_over = False
        max_patience, cur_patience = args.patience, args.patience
        for epoch in range(starting_epoch, args.num_train_epochs):
            # 调用evaluate时会改变状态，所以需要重新指定
            self._set_mode('train')
            timer.reset()
            epoch_outputs = []
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    # skip completed steps in current epoch
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue
                with accelerator.accumulate(mpl_model.model):
                    batch_outputs = mpl_model.training_step(batch)
                    loss = batch_outputs.loss
                    optimizer.zero_grad()
                    accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                # We keep track of the loss at each epoch
                total_loss += loss.detach().float().item()
                completed_steps += 1

                batch_outputs = mpl_model.training_step_end(batch_outputs)
                epoch_outputs.append(batch_outputs)

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        model_path = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            model_path = os.path.join(args.output_dir, model_path)
                        accelerator.save_state(model_path)
                        mpl_logger.log(
                            f"training step {completed_steps-checkpointing_steps}-{completed_steps}, "
                            f"train loss: {total_loss/(step+1)}, {timer.get_elapsed_time()}"
                        )
                        eval_outputs = self.evaluate(eval_dataloader)
                        mpl_logger.log(
                            f"eval: {eval_outputs}, {timer.get_elapsed_time()}"
                        )
                        if args.with_tracking:
                            accelerator.log(
                                {
                                    "metrics": eval_outputs.metric_results,
                                    "train_loss": total_loss / (step+1),
                                    "eval_loss": eval_outputs.loss,
                                    "epoch": epoch,
                                    "step": completed_steps
                                },
                                step=completed_steps
                            )
                        tgt_metric_name = self.args.target_metric
                        tgt_metric_value = eval_outputs.target_metric_result[tgt_metric_name]
                        if (larger_is_better and tgt_metric_value >= best_score) \
                                or (not larger_is_better and tgt_metric_value <= best_score):
                            cur_patience = max_patience
                            best_score = tgt_metric_value
                            best_model_path = model_path
                        else:
                            if args.use_early_stop and cur_patience <= 0:
                                training_over = True
                                mpl_logger.log(f"patience over, best tgt metric `{tgt_metric_name}`: {tgt_metric_value}")
                                break
                            cur_patience -= 1
                if completed_steps >= args.max_train_steps:
                    training_over = True
                    break
            mpl_model.training_epoch_end(epoch_outputs)

            if checkpointing_steps == "epoch":
                model_path = f"epoch_{epoch}"
                if args.output_dir is not None:
                    model_path = os.path.join(args.output_dir, model_path)
                accelerator.save_state(model_path)
                mpl_logger.log(
                    f"training epoch {epoch}, train loss: {total_loss/len(train_dataloader)}, "
                    f"{timer.get_elapsed_time()}"
                )
                eval_outputs = self.evaluate(eval_dataloader)
                mpl_logger.log(
                    f"eval: {eval_outputs}, {timer.get_elapsed_time()}"
                )
                if args.with_tracking:
                    accelerator.log(
                        {
                            "metrics": eval_outputs.metric_results,
                            "train_loss": total_loss / len(train_dataloader),
                            "eval_loss": eval_outputs.loss,
                            "epoch": epoch,
                            "step": completed_steps
                        },
                        step=completed_steps
                    )
                tgt_metric_name = self.args.target_metric
                tgt_metric_value = eval_outputs.target_metric_result[tgt_metric_name]
                if (larger_is_better and tgt_metric_value >= best_score) \
                        or (not larger_is_better and tgt_metric_value <= best_score):
                    cur_patience = max_patience
                    best_score = tgt_metric_value
                    best_model_path = model_path
                else:
                    if args.use_early_stop and cur_patience <= 0:
                        training_over = True
                        mpl_logger.log(f"patience over, best tgt metric `{tgt_metric_name}`: {tgt_metric_value}")
                    cur_patience -= 1
            if training_over:
                break

        if args.with_tracking:
            accelerator.end_training()
        accelerator.wait_for_everyone()

        if args.use_early_stop:
            # 保存最佳模型
            accelerator.load_state(best_model_path)
            mpl_logger.log("in early stop mode, trainer load best model in last to save")

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(mpl_model.model)
            unwrapped_model.save_pretrained(
                args.model_checkpoint,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dic=accelerator.get_state_dict(mpl_model.model)
            )
            mpl_model.tokenizer.save_pretrained(args.model_checkpoint)

        mpl_logger.log(f"training over, {timer.get_elapsed_time(start=running_start)}")
        accelerator.wait_for_everyone()

    def train_loop(self, mpl_model, batch):
        batch_outputs = mpl_model.training_step(batch)
        loss = batch_outputs.loss
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return batch_outputs

    def evaluate(self, eval_loader, checkpoint_dir=None):
        self._set_mode('eval')
        accelerator = self.accelerator
        mpl_model = self.mpl_model
        mpl_logger = self.mpl_logger
        if checkpoint_dir:
            model_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
            self._load_state_dict(model_path)
            mpl_logger.log(f"load from model checkpoint: {model_path}")

        epoch_outputs = []
        for step, batch in enumerate(tqdm(eval_loader, desc='eval', disable=not accelerator.is_local_main_process)):
            batch_outputs = mpl_model.validation_step(batch, accelerator)
            batch_outputs = mpl_model.validation_step_end(batch_outputs)
            epoch_outputs.append(batch_outputs)
        return mpl_model.validation_epoch_end(epoch_outputs, tgt_metric=self.args.target_metric)

    def predict(self, test_file=None, label_key=None, batch_size=None, cache_dir=None, checkpoint_dir=None):
        """ 加载训练好的模型，在测试集上预测，预测结果保存在checkpoint_dir中

        :param test_file: 测试数据路径，csv、json格式
        :param label_key: 数据中标签的关键字，若无标签，可指定为None
        :param batch_size: 单GPU处理的批次大小
        :param cache_dir: 数据处理后的缓存目录
        :param checkpoint_dir: 模型参数目录，必须包含pytorch_model.bin
        :return:
        """
        self._set_mode('eval')
        args = self.args
        mpl_logger = self.mpl_logger
        accelerator = self.accelerator
        mpl_model = self.mpl_model

        test_file = args.test_file if test_file is None else test_file
        label_key = args.label_key if label_key is None else label_key
        batch_size = args.per_device_eval_batch_size if batch_size is None else batch_size
        cache_dir = args.cache_dir if cache_dir is None else cache_dir
        checkpoint_dir = args.model_checkpoint if checkpoint_dir is None else checkpoint_dir

        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes

        model_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
        self._load_state_dict(model_path)
        mpl_logger.log(f"load from model checkpoint: {model_path}")

        with accelerator.main_process_first():
            test_data = mpl_model.get_test_data(
                test_file=test_file,
                label_key=label_key,
                batch_size=batch_size,
                cache_dir=cache_dir
            )
            test_dataset, test_dataloader = test_data.dataset, test_data.dataloader
        for index in random.sample(range(len(test_dataset)), 3):
            mpl_logger.log(f"Sample {index}(idx) of the test set: {test_dataset[index]}.")
        mpl_model.model, test_dataloader = accelerator.prepare(mpl_model.model, test_dataloader)

        mpl_logger.log("***** Running predict *****")
        mpl_logger.log(f"  Num examples = {len(test_dataset)}")
        mpl_logger.log(f"  Instantaneous batch size per device = {batch_size}")
        mpl_logger.log(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")

        timer.reset()

        epoch_outputs = []
        for step, batch in enumerate(tqdm(test_dataloader, desc='predict', disable=not accelerator.is_local_main_process)):
            batch_outputs = mpl_model.test_step(batch, accelerator)
            batch_outputs = mpl_model.test_step_end(batch_outputs)
            epoch_outputs.append(batch_outputs)
        predict_results = mpl_model.test_epoch_end(epoch_outputs)

        mpl_logger.log(f"predict over, {timer.get_elapsed_time()}")
        if predict_results.loss is not None:
            mpl_logger.log(f"predict loss: {predict_results.loss}, {predict_results.metric_results}")

        prediction_path = os.path.join(checkpoint_dir, "predictions.json")
        if accelerator.is_main_process:
            writer.write_json(predict_results.predictions, prediction_path)
        mpl_logger.log(f"predict results saved in {prediction_path}")

    def _set_mode(self, mode:str):
        """快速修改模型状态，同时根据状态自动决定是否计算梯度

        :param mode: 'train' or 'eval'
        """
        if mode == 'train':
            self.mpl_model.train()
            torch.set_grad_enabled(True)
        elif mode == 'eval':
            self.mpl_model.eval()
            torch.set_grad_enabled(False)
        else:
            raise ValueError(f"mode must be train or eval, but you set {mode}, please fix it")

    def _load_state_dict(self, checkpoint_path):
        """加载checkpoint_path中的模型参数

        :param checkpoint_path: 如果是目录，则必须包含pytorch_model.bin
        """
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if hasattr(self.mpl_model.model, 'module'):
            self.mpl_model.model.module.load_state_dict(state_dict)
        else:
            self.mpl_model.model.load_state_dict(state_dict)