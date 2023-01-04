"""
重点参考https://github.com/facebookresearch/fairseq/blob/b5a039c292facba9c73f59ff34621ec131d82341/fairseq/options.py
"""
import os
import argparse
import warnings


def parse_args(parser=None, task='text-classification') -> argparse.Namespace:
    """解析参数并进行合法性检查, parser和task只需要传入一个即可

    :param parser: 用于解析
    :param task: 用于加载任务特定参数
    :return: args
    """
    if parser is not None and task:
        warnings.warn("both `parser` and `task` is provided, ignore `task`")

    parser = get_parser(task) if parser is None else parser
    args = parser.parse_args()

    # 合法性检查
    args.output_dir = os.path.realpath(args.output_dir)
    if args.model_checkpoint is not None:
        args.model_checkpoint = os.path.join(args.output_dir, args.model_checkpoint)
    else:
        args.model_checkpoint = args.output_dir

    if args.do_train and args.train_file is None and args.eval_file is None:
        raise ValueError("Need training and validation file when train model")
    if args.do_predict and args.test_file is None:
        raise ValueError("Need test file when model inference")
    if not (args.do_train or args.do_predict):
        raise ValueError("you must set '`--do_train`' or '`--do_predict`' at least one")
    if args.do_train and args.model_name_or_path is None:
        raise ValueError("Need pretrained language model to initialize model before train")

    if args.train_file is not None:
        args.train_file = os.path.realpath(args.train_file)
        extension = args.train_file.split(".")[-1]
        assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    if args.eval_file is not None:
        args.eval_file = os.path.realpath(args.eval_file)
        extension = args.eval_file.split(".")[-1]
        assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    if args.test_file is not None:
        args.test_file = os.path.realpath(args.test_file)
        extension = args.test_file.split('.')[-1]
        assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
    if args.cache_dir is not None:
        args.cache_dir = os.path.realpath(args.cache_dir)

    if args.warmup_ratio < 0 or args.warmup_ratio > 1:
        raise ValueError("warmup_ratio must lie in range [0,1]")
    elif args.warmup_ratio > 0 and args.num_warmup_steps > 0:
        warnings.warn(
            "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio"
            " during training"
        )
    if args.target_metric != 'eval_loss' and args.target_metric not in args.metrics:
        raise ValueError(f"you tgt_metric is {args.target_metric}, not in the metrics you set: {args.metrics}")
    if args.max_train_steps is not None and args.checkpointing_steps is not None:
        if not args.checkpointing_steps.isdigit():
            raise ValueError(
                "you set `max_train_steps`, in this case, `checkpointing_steps` should be None or n steps, not epoch"
            )

    return args


def get_parser(task='text-classification'):
    parser = argparse.ArgumentParser(task)
    add_logging_args(parser)
    add_dataset_args(parser, task=task)
    add_optimizer_args(parser)
    add_scheduler_args(parser)
    add_training_args(parser)
    add_metric_args(parser)
    add_plm_args(parser)
    add_tokenizer_args(parser)
    return parser


def add_training_args(parser:argparse.ArgumentParser):
    group = parser.add_argument_group('training')
    group.add_argument(
        "--do_train", action='store_true', help='whether to train model with train and validation data.'
    )
    group.add_argument(
        "--do_predict", action='store_true', help='whether to use model to inference with test data.'
    )
    group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    group.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation/test dataloader.",
    )
    group.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    group.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=(
            "Total number of training steps to perform. "
            "If provided, overrides num_train_epochs. Once provided, must manually set `--checkpointing_steps`"
        ),
    )
    group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    group.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help=(
            "relative path to `--output_dir`, for predict, load for inference; "
            "for train, save final checkpoint, when use early stop, it will be the best model; "
            "if None, same with `--output_dir`"
        )
    )
    group.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    group.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        choices=[None, 'epoch', str],
        help=(
            "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
            "if None, model will be saved after all training steps/epoch. Saved with validate!"
        ),
    )
    group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    group.add_argument(
        "--cache_dir",
        type=str,
        help="the directory to save cache files, if None, '~/.cache/huggingface/datasets' will be set"
    )
    group.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to store model/log and all about the train/predict"
    )
    group.add_argument(
        "--use_early_stop",
        action="store_true",
        help="whether use early stop to avoid overfitting"
    )
    group.add_argument(
        "--patience",
        type=int,
        default=3,
        help=(
            "early stop training if valid performance doesn’t improve for N consecutive validation runs; "
            "note that this is influenced by `–-validate_interval`"
        )
    )
    return group


def add_optimizer_args(parser:argparse.ArgumentParser):
    group = parser.add_argument_group('optimizer')
    group.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    group.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    group.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 hyperparameter for the AdamW optimizer.")
    group.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 hyperparameter for the AdamW optimizer.")
    group.add_argument("--adam_epsilon", type=float, default=1e-8, help="The epsilon hyperparameter for the AdamW optimizer.")
    return group


def add_scheduler_args(parser:argparse.ArgumentParser):
    group = parser.add_argument_group('lr_scheduler')
    group.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    group.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="The number of warmup steps to do for a linear warmup from 0 to learning_rate. Overrides any effect of `--warmup_ratio`."
    )
    group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Ratio of total training steps used for a linear warmup from 0 to learning_rate."
    )
    return group


def add_metric_args(parser:argparse.ArgumentParser):
    group = parser.add_argument_group('metric')
    group.add_argument('--metrics', nargs="+", required=True, type=str, help="use to evaluate model")
    group.add_argument(
        '--target_metric',
        type=str,
        default='eval_loss',
        help="use to early stop, must be contained in `--metrics`, except eval_loss"
    )
    group.add_argument(
        "--larger_is_better",
        action='store_true',
        help="use to `--target_metric`, larger = better"
    )
    return group


def add_dataset_args(parser:argparse.ArgumentParser, task='text-classification'):
    group = parser.add_argument_group('dataset')
    group.add_argument('--train_file', type=str)
    group.add_argument('--eval_file', type=str)
    group.add_argument('--test_file', type=str)
    group.add_argument('--label_key', type=str, default='label', help='label key name in dataset')
    # group.add_argument(
    #     '--num_proc',
    #     type=int,
    #     default=1,
    #     help='num worker to process data, use for datasets.map and dataloader'
    # )
    if task == 'text-classification':
        group.add_argument('--sentence1_key', type=str, required=True, help='sentence1 key name in dataset')
        group.add_argument('--sentence2_key', type=str, default=None, help='sentence2 key name in dataset')
        group.add_argument(
            '--num_labels',
            type=int,
            default=None,
            help=(
                "if None, automatically set by training/test set labels (when them exists). "
                "when predict with no label test data, and train_file is empty, raise error."
            )
        )
    return group


def add_tokenizer_args(parser:argparse.ArgumentParser):
    group = parser.add_argument_group('tokenizer')
    group.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        )
    )
    group.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    group.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    return group


def add_plm_args(parser:argparse.ArgumentParser):
    group = parser.add_argument_group('plm-model')
    group.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    group.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    return group


def add_logging_args(parser:argparse.ArgumentParser):
    group = parser.add_argument_group('logging')
    group.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    group.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    return group