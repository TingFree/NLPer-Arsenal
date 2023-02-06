"""
调用mpl完成文本分类、回归任务，可加载huggingface预训练模型
"""
import sys
sys.path.append('../../..')
import torch
import evaluate
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from arsenal.models import MplCLF
from arsenal.mpl import Trainer, MplOutput
from arsenal.utils.options import get_parser, parse_args


def get_simcse_loss(once_emb, twice_emb, t=0.05):
    """用于无监督SimCSE训练的loss

    :param once_emb: [batch_size, emb_dim], 第一次dropout后的句子编码
    :param twice_emb: [batch_size, emb_dim], 第二次dropout后的句子编码
    :param t: 温度系数
    """
    # 构造标签，[1,0,3,2,5,4,...]
    batch_size = once_emb.size(0)
    y_true = torch.cat([torch.arange(1, batch_size*2, step=2, dtype=torch.long).unsqueeze(1),
                        torch.arange(0, batch_size*2, step=2, dtype=torch.long).unsqueeze(1)],
                       dim=1).reshape([batch_size*2,]).to(once_emb.device)

    batch_emb = torch.cat([once_emb, twice_emb], dim=1).reshape(batch_size*2, -1)  # [a1,a2,b1,b2,...]
    # 计算score和loss
    # L2标准化
    norm_emb = F.normalize(batch_emb, dim=1, p=2)
    # 计算一个batch内样本之间的相似度
    sim_score = torch.matmul(norm_emb, norm_emb.transpose(0,1))
    # mask掉和自身的相似度
    sim_score = sim_score - torch.eye(batch_size*2, device=once_emb.device) * 1e12
    sim_score = sim_score / t
    loss = F.cross_entropy(sim_score, y_true)
    return loss


class SimcseCLF(MplCLF):
    def training_step(self, batch):
        outputs1 = self.model(**batch)
        outputs2 = self.model(**batch)
        simcse_loss = get_simcse_loss(outputs1.logits, outputs2.logits, t=self.config.temperature)
        final_loss = simcse_loss + outputs1.loss
        return MplOutput(loss=final_loss)


if __name__ == '__main__':
    parser = get_parser(task='text-classification')
    parser.add_argument('--temperature', type=float, default=0.05, help="the temperature of simcse")
    args = parse_args(parser)

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

    mpl_model = SimcseCLF(
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
