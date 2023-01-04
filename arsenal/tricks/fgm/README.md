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
| text_clf (P/R/Macro F1) | smp2020-ewect-usual | default |      3623.06      |   00:38:21   | 0.7346/0.7293/0.7293 |
|                         |                     |   fgm   |      3649.06      |   00:53:50   | 0.7361/0.7553/0.7432 |
|                         | smp2020-ewect-virus | default |      5897.06      |   00:25:39   | 0.6662/0.6195/0.6380 |
|                         |                     |   fgm   |      5875.06      |   01:11:32   | 0.6605/0.6404/0.6480 |

# references
https://fyubang.com/2019/10/15/adversarial-train/