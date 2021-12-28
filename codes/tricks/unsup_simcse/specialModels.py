import torch
import torch.nn.functional as F
from codes.nlper.models import LightningCLF

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

    batch_emb = torch.cat([once_emb, twice_emb], dim=1).reshape(batch_size*2, -1)  # [a,a1,b,b1,...]
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


class CLFModel(LightningCLF):
    """
    通过继承标准分类模型，将该策略应用于分类任务
    """
    def __init__(self, *args, **kwargs):
        print('use trick in CLF Task: unsup simcse')
        super(CLFModel, self).__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        labels = batch['labels']

        logits, once_emb = self.model(**batch, return_pooler_output=True)
        _, twice_emb = self.model(**batch, return_pooler_output=True)

        loss = F.cross_entropy(logits.view(-1, self.configs.num_class),
                               labels.view(-1))
        simcse_loss = get_simcse_loss(once_emb, twice_emb)
        final_loss = loss + simcse_loss

        return final_loss,