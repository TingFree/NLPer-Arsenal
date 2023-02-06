"""
调用mpl完成文本分类、回归任务，可加载huggingface预训练模型
"""
import sys
sys.path.append('../..')
import evaluate
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from accelerate.utils import set_seed
from arsenal.mpl import Trainer
from arsenal.models import MplCLF
from arsenal.utils.options import parse_args
from arsenal.utils import Timer


if __name__ == '__main__':
    args = parse_args(task='text-classification')
    if args.seed:
        set_seed(args.seed)
    timer = Timer()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes
    )
    metrics = evaluate.combine(args.metrics)

    mpl_model = MplCLF(
        config=args,
        model=model,
        tokenizer=tokenizer,
        metrics=metrics
    )
    trainer = Trainer(args, mpl_model)

    if args.do_train:
        # 若不指定自定义参数，则默认使用arsenal.nlper.utils.options.py中的参数值以及终端指定的参数值
        trainer.train()

    if args.do_predict:
        trainer.predict(
            test_file=args.test_file,
            label_key=args.label_key,
            batch_size=args.per_device_eval_batch_size,
            cache_dir=args.cache_dir,
            checkpoint_dir=args.model_checkpoint
        )

    print(f"run_mpl over, {timer.get_elapsed_time()}")