try:
    import bitsandbytes as bnb
except:
    raise ImportError(
        f'please install bitsandbytes before you use this trick, see README in detail'
    )
from transformers import get_linear_schedule_with_warmup
from codes.nlper.models import LightningCLF


class CLFModel(LightningCLF):
    """
    通过继承标准分类模型，将该策略应用于分类任务
    """
    def __init__(self, *args, **kwargs):
        print('use trick in CLF Task: 8-bit')
        super(CLFModel, self).__init__(*args, **kwargs)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.configs.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            lr=self.configs.lr
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            self.configs.warmup_steps,
            self.configs.trainer_args.max_epochs * self.aux_configs.num_train_batch
        )
        return optimizer, scheduler
