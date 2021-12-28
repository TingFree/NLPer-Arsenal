import torch
import torch.nn.functional as F
from codes.nlper.models import LightningCLF


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


class CLFModel(LightningCLF):
    """
    通过继承标准分类模型，将该策略应用于分类任务
    """
    def __init__(self, *args, **kwargs):
        print('use trick in CLF Task: fgm')
        super(CLFModel, self).__init__(*args, **kwargs)
        self.fgm = FGM(self.model)
        self.auto_optimization = False  # we need manual optimization

    def training_step(self, batch, batch_idx):
        labels = batch['labels']
        logits = self.model(**batch)
        loss = F.cross_entropy(logits.view(-1, self.configs.num_class),
                               labels.view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.fgm.attack()  # attack before run model
        loss_adv = F.cross_entropy(self.model(**batch).view(-1, self.configs.num_class),
                                   labels.view(-1))
        loss_adv.backward()
        self.fgm.restore()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        return loss,
