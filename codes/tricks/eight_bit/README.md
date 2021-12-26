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
python center_controller.py --whole_model BertCLF --trick_name eight_bit --task_config default_configs/text_clf_s.yaml
```



|          task           |       dataset       | method  | GPU max mem (MiB) | running time |        score         |
| :---------------------: | :-----------------: | :-----: | :---------------: | :----------: | :------------------: |
| text_clf (P/R/Macro F1) | smp2020-ewect-usual | default |      9357.06      |   00:31:36   | 0.7325/0.7513/0.7402 |
|                         |                     |  8bit   |      8741.06      |   00:27:36   | 0.7284/0.7308/0.7279 |
|                         | smp2020-ewect-virus | default |     10853.06      |   00:17:34   | 0.6409/0.6442/0.6309 |
|                         |                     |  8bit   |     10289.06      |   00:23:23   | 0.6617/0.6438/0.6493 |



# references

https://github.com/facebookresearch/bitsandbytes