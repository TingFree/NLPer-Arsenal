"""
固定训练流程，只修改模型、数据以及训练参数，要求能够加载预训练模型即可
"""
import os
import math
import json
import random
import logging
import argparse
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

import evaluate
import datasets
import transformers
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.18.0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--do_train", action='store_true', help='whether to train model with train and validation data.'
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--do_predict", action='store_true', help='whether to use model to inference with test data.'
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--sentence1_key", type=str, default=None, help="sentence1 key name in dataset"
    )
    parser.add_argument(
        "--sentence2_key", type=str, default=None, help="sentence2 key name in dataset"
    )
    parser.add_argument(
        "--label_key", type=str, default="label", help="label key name in dataset"
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=None,
        help=(
            "if None, automatically set by training/test set labels (when them exists). "
            "when predict with no label test data, and train_file is empty, raise error."
        )
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='../../data/cache_dir',
        help="the directory to save cache files, if None, '~/.cache/huggingface/datasets' will be set"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation/test dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default='../../data/outputs/clf', help="Where to store model/log and all about the train/predict")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help=(
            "relative path to `--output_dir`, for train, save final checkpoint; for predict, load for inference."
            "if None, same with `--output_dir`"
        )
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        choices=[None, 'epoch', str],
        help=(
            "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
            "if None, model will be saved after all training steps/epoch"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    args = parser.parse_args()

    # convert relative path to absolute path
    args.cache_dir = os.path.realpath(args.cache_dir)
    args.output_dir = os.path.realpath(args.output_dir)
    if args.train_file is not None:
        args.train_file = os.path.realpath(args.train_file)
    if args.validation_file is not None:
        args.validation_file = os.path.realpath(args.validation_file)
    if args.test_file is not None:
        args.test_file = os.path.realpath(args.test_file)
    if args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = os.path.realpath(args.resume_from_checkpoint)
    if args.model_checkpoint is not None:
        args.model_checkpoint = os.path.join(args.output_dir, args.model_checkpoint)
    else:
        args.model_checkpoint = args.output_dir

    # Sanity checks
    if args.do_train and args.train_file is None and args.validation_file is None:
        raise ValueError("Need training and validation file when train model")
    if args.do_predict and args.test_file is None:
        raise ValueError("Need test file when model inference")
    if not (args.do_train or args.do_predict):
        raise ValueError("you must set '`--do_train`' or '`--do_predict`'")
    if args.do_train and args.model_name_or_path is None:
        raise ValueError("Need pretrained language model when train model")

    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    if args.test_file is not None:
        extension = args.test_file.split('.')[-1]
        assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    logging_dir = os.path.join(args.output_dir, 'logs')
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=logging_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        # filename=os.path.join(logging_dir, 'log.txt'),
        # filemode='w'
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the metric function
    metrics = evaluate.combine(["accuracy", "precision", "recall", "f1"])

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # Loading the dataset from local csv or json file.
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file
    # csv, json
    extension = (args.train_file if args.train_file is not None else args.test_file).split(".")[-1]
    with accelerator.main_process_first():
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    # default to 'label'
    label_key = args.label_key
    if "train" in raw_datasets:
        tgt_domain = "train"
    elif "test" in raw_datasets and label_key in raw_datasets["test"].features:
        tgt_domain = "test"
    else:
        tgt_domain = None

    label_list = None
    is_regression = False
    if tgt_domain:
        is_regression = raw_datasets[tgt_domain].features[label_key].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets[tgt_domain].unique(label_key)
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    else:
        num_labels = args.num_labels
        assert args.num_labels is not None, "`--num_labels` is None, but labels not found in test file."

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.do_predict and not args.do_train:
        # only inference, just load checkpoint
        args.model_name_or_path = args.model_checkpoint

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    accelerator.print(f"initialize model: {args.model_name_or_path}")

    # Preprocessing the datasets
    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    if args.sentence1_key is None and args.sentence2_key is None:
        if args.train_file is not None:
            no_label_column_names = [name for name in raw_datasets["train"].column_names if name != label_key]
        else:
            no_label_column_names = [name for name in raw_datasets["test"].column_names if name != label_key]
        if "sentence1" in no_label_column_names and "sentence2" in no_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(no_label_column_names) >= 2:
                sentence1_key, sentence2_key = no_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = no_label_column_names[0], None
    else:
        sentence1_key = args.sentence1_key
        sentence2_key = args.sentence2_key

    label_to_id = None
    if  label_list is not None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names if args.train_file is not None else raw_datasets["test"].column_names,
            desc="Running tokenizer on dataset",
        )

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done to max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    if args.do_train:
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Figure out how many steps we should save the Accelerator states
        checkpointing_steps = args.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if args.with_tracking:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("base-trainer", experiment_config)

        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(args.max_train_steps),
            desc="train",
            disable=not accelerator.is_local_main_process
        )
        completed_steps = 0
        starting_epoch = 0
        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(args.output_dir)
                        if f.is_dir() and (f.name.startswith('epoch_') or f.name.startswith('step_'))]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                args.resume_from_checkpoint = os.path.join(args.output_dir, path)
                accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
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

        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            if args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    # skip completed steps in current epoch
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

            # evaluate each epoch
            model.eval()
            for step, batch in enumerate(tqdm(eval_dataloader, desc='eval', disable=not accelerator.is_local_main_process)):
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                metrics.add_batch(
                    predictions=predictions,
                    references=references,
                )

            eval_metrics = metrics.compute()
            logger.info(f"epoch {epoch}: {eval_metrics}")

            if args.with_tracking:
                accelerator.log(
                    {
                        "metrics": eval_metrics,
                        "train_loss": total_loss.item() / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

        if args.with_tracking:
            accelerator.end_training()

        accelerator.wait_for_everyone()

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.model_checkpoint,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dic=accelerator.get_state_dict(model)
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.model_checkpoint)
        accelerator.wait_for_everyone()

    if args.do_predict:
        test_dataset = processed_datasets["test"]
        # Log a few random samples from the test set:
        for index in random.sample(range(len(test_dataset)), 3):
            logger.info(f"Sample {index} of the test set: {test_dataset[index]}.")

        logger.info("***** Running Predict *****")
        logger.info(f"  Num examples = {len(test_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
        logger.info(f"  Total predict batch size (w. parallel, distributed) = {args.per_device_eval_batch_size * accelerator.num_processes}")

        do_metrics = False
        if 'labels' in test_dataset.features:
            do_metrics = True
        test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        # load checkpoint
        accelerator.print(f"load from model checkpoint: {args.model_checkpoint}")
        state_dicts = torch.load(os.path.join(args.model_checkpoint, 'pytorch_model.bin'), map_location='cpu')
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dicts)
        else:
            model.load_state_dict(state_dicts)

        model, test_dataloader = accelerator.prepare(model, test_dataloader)

        model.eval()
        predict_loss, samples_seen = 0, 0
        total_predictions = []
        for step, batch in enumerate(tqdm(test_dataloader, desc="predict", disable=not accelerator.is_local_main_process)):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if num_labels>1 else outputs.logits.squeeze()
            predictions = accelerator.gather_for_metrics(predictions)
            if do_metrics:
                references = accelerator.gather_for_metrics(batch["labels"])
                predict_loss += outputs.loss.detach().float()

                metrics.add_batch(
                    predictions=predictions,
                    references=references,
                )
            if label_list is not None:
                predictions = [label_list[pred] for pred in predictions.tolist()]
                total_predictions.extend(predictions)
            else:
                total_predictions.extend(predictions.tolist())

        if do_metrics:
            pred_metrics = metrics.compute()
            predict_loss = predict_loss.item() / len(test_dataloader)
            logger.info(f"predict metrics: {pred_metrics}")
            logger.info(f"predict loss: {predict_loss}")

            with open(os.path.join(args.output_dir, "prediction_metrics.json"), "w") as f:
                json.dump({
                    "model": args.model_checkpoint,
                    "dataset": args.test_file,
                    "metrics": pred_metrics,
                    "predict_loss": predict_loss
                }, f, indent=1)
                accelerator.print(f"prediction metrics -> {os.path.join(args.output_dir, 'prediction_metrics.json')}")

        with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
            json.dump(total_predictions, f, indent=1)
            accelerator.print(f"predictions -> {os.path.join(args.output_dir, 'predictions.json')}")
        accelerator.print("predict over")

if __name__ == "__main__":
    main()