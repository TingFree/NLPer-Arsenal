import torch
from torch import nn
from codes.nlper.modules.modeling_outputs import DecoderOutput

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

    def forward(self, tgt, memory, memory_padding_mask=None, tgt_padding_mask=None):
        """

        :param tgt: tensor, [batch_size, tgt_len, d_model]
        :param memory: tensor, [batch_size, src_len, d_model]
        :param memory_padding_mask: tensor, [batch_size, src_len], nn.transformerDecoderLayer中的memory_key_padding_mask
        :param tgt_padding_mask: tensor/None, [batch_size, tgt_len], nn.transformerDecoderLayer中的tgt_key_padding_mask
        :return: DecoderOutput(last_hidden_state), last_hidden_state: [batch_size, tgt_len, d_model]
        """
        tgt_len = tgt.size(1)
        outputs = self.decoder(
            tgt.transpose(0,1),
            memory.transpose(0,1),
            tgt_mask=self.generate_square_subsequent_mask(tgt_len).to(tgt.device),
            memory_key_padding_mask=memory_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        ).transpose(0,1)
        return DecoderOutput(
            last_hidden_state = outputs
        )

    def generate_square_subsequent_mask(self, sz): 
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask