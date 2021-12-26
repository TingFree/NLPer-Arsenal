# FGM

## how to use

```python
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

# 初始化
fgm = FGM(model)
for batch_input, batch_label in data:
    # 正常训练
    loss = model(batch_input, batch_label)
    loss.backward() # 反向传播，得到正常的grad
    # 对抗训练
    fgm.attack() # 在embedding上添加对抗扰动
    loss_adv = model(batch_input, batch_label)
    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    fgm.restore() # 恢复embedding参数
    # 梯度下降，更新参数
    optimizer.step()
    model.zero_grad()
```



## ablation study

```shell
cd tricks
python center_controller.py ---trick_name fgm --task_config default_configs/text_clf_smp2020_ewect_usual.yaml
```



|          task           |       dataset       | method  | GPU max mem (MiB) | running time |        score         |
| :---------------------: | :-----------------: | :-----: | :---------------: | :----------: | :------------------: |
| text_clf (P/R/Macro F1) | smp2020-ewect-usual | default |      9357.06      |   00:31:36   | 0.7325/0.7513/0.7402 |
|                         |                     |   fgm   |      9325.06      |   01:17:33   | 0.7198/0.7429/0.7293 |
|                         | smp2020-ewect-virus | default |     10853.06      |   00:17:34   | 0.6409/0.6442/0.6309 |
|                         |                     |   fgm   |     10817.06      |   00:44:48   | 0.6473/0.6598/0.6496 |

# references
https://fyubang.com/2019/10/15/adversarial-train/