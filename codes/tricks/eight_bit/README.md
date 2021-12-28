# 8bit
## environment setup
```shell
# choices: {cuda92, cuda 100, cuda101, cuda102, cuda110, cuda111, cuda113}
# replace XXX with the respective number
pip install bitsandbytes-cudaXXX
```
## how to use
```python
import bitsandbytes as bnb

# adam = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # comment out old optimizer
adam = bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995)) # add bnb optimizer
adam = bnb.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995), optim_bits=8) # equivalent


torch.nn.Embedding(...) ->  bnb.nn.StableEmbedding(...) # recommended for NLP models
```

## ablation study

```shell
cd tricks
python center_controller.py --whole_model BertCLF --trick_name eight_bit --task_config default_configs/text_clf_sm.yaml
```



|          task           |       dataset       | method  | GPU max mem (MiB) | running time |        score         |
| :---------------------: | :-----------------: | :-----: | :---------------: | :----------: | :------------------: |
| text_clf (P/R/Macro F1) | smp2020-ewect-usual | default |      3623.06      |   00:38:21   | 0.7346/0.7293/0.7293 |
|                         |                     |  8bit   |      2989.06      |   00:33:13   | 0.7282/0.7144/0.7172 |
|                         | smp2020-ewect-virus | default |      5897.06      |   00:25:39   | 0.6662/0.6195/0.6380 |
|                         |                     |  8bit   |      5333.06      |   00:12:45   | 0.6713/0.6186/0.6380 |



# references

https://github.com/facebookresearch/bitsandbytes