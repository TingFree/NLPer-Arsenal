import math
import torch
from torch import nn


class PosEmbedding(nn.Module):
    """绝对位置编码
    """
    def __init__(self, emb_size: int, max_len: int = 512):
        super(PosEmbedding).__init__()
        assert emb_size % 2 == 0, "emb_size must be even"
        # [max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)
        # [d_model // 2]
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        pe = torch.zeros(max_len, emb_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, seq_len:int):
        """

        :param seq_len: 每个batch中的序列长度
        :return: [1, seq_len, emb_size]
        """
        emb = self.pe[:seq_len]
        return emb.unsqueeze(0)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size=300, dropout=0.1, max_len=512):
        super(TransformerEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.tok_embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_embedding = PosEmbedding(emb_size, max_len)

    def forward(self, x):
        """

        :param x: Tensor, [batch, seq_len]
        :return: [batch, seq_len, emb_size]
        """
        tok_emb = self.tok_embedding(x)
        pos_emb = self.pos_embedding(x.size(1))
        final_emb = tok_emb + pos_emb
        return self.dropout(final_emb)