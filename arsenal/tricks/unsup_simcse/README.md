# unsup SimCSE

## 说明

对比学习，无监督SimCSE

## how to use

```python
def get_simcse_loss(once_emb, twice_emb, t=0.05):
    """用于无监督SimCSE训练的loss

    :param once_emb: [batch_size, emb_dim], 第一次dropout后的句子编码
    :param twice_emb: [batch_size, emb_dim], 第二次dropout后的句子编码
    :param t: 温度系数
    """
    # 构造标签，[1,0,3,2,5,4,...]
    batch_size = once_emb.size(0)
    y_true = torch.cat([torch.arange(1, batch_size*2, step=2, dtype=torch.long).unsqueeze(1), torch.arange(0, batch_size*2, step=2, dtype=torch.long).unsqueeze(1)], dim=1).reshape([batch_size*2,]).to(once_emb.device)

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

for batch_input, batch_label in data:
    logits, seq_emb1 = model(batch_input)  # 第一次dropout
    _, seq_emb2 = model(batch_input)  # 第二次dropout
    loss = loss_fn(logits, batch_label)
    simcse_loss = get_simcse_loss(seq_emb1, seq_emb2)
    final_loss = loss + simcse_loss
    final_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```



## ablation study

```shell
cd tricks
python center_controller.py ---trick_name unsup_s --task_config default_configs/text_clf_smp2020_ewect_usual.yaml
```



|          task           |       dataset       |    method    | GPU max mem (MiB) | running time |        score         |
| :---------------------: | :-----------------: | :----------: | :---------------: | :----------: | :------------------: |
| text_clf (P/R/Macro F1) | smp2020-ewect-usual |   default    |      3623.06      |   00:38:21   | 0.7346/0.7293/0.7293 |
|                         |                     | unsup simcse |      4767.06      |   00:53:09   | 0.7327/0.7484/0.7394 |
|                         | smp2020-ewect-virus |   default    |      5897.06      |   00:25:39   | 0.6662/0.6195/0.6380 |
|                         |                     | unsup simcse |      9287.06      |   00:38:12   | 0.6434/0.6431/0.6399 |



## references 

1. [《超细节的对比学习和SimCSE知识点》](https://zhuanlan.zhihu.com/p/378340148) 
