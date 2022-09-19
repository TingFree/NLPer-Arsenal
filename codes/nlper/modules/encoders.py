from torch import nn
from codes.nlper.modules.modeling_outputs import EncoderOutput

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, src, src_mask=None):
        """

        :param src: tensor, [batch_size, src_len, emb_size]
        :param src_mask: tensor/None, [batch_size, src_len];
        :return: EncoderOutput(seqEmb), seqEmb: [batch_size, src_len, d_model]
        """
        outputs = self.encoder(src, src_key_padding_mask=src_mask)
        return EncoderOutput(
            seqEmb = outputs
        )
