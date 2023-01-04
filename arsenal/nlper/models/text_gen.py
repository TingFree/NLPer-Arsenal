r"""
各种文本生成模型的实现
"""
import torch
from torch import nn
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
import codes.nlper.mini_pytorch_lightning as mpl
from codes.nlper.modules.modeling_outputs import LightningOutput, ModelOutput, TextGenOutput
from codes.nlper.modules.decoders import TransformerDecoder


class LightningGen(mpl.StandardModel):
    def __init__(self, config, metrics, **kwargs):
        super(LightningGen, self).__init__(config, metrics, **kwargs)
        pass


# class EncoderDecoderBase(nn.Module):
#     def __init__(self, hidden_size, vocab_size):
#         super(EncoderDecoderBase, self).__init__()
#         self.embedding = None
#         self.encoder = None
#         self.decoder = None
#         self.generator = nn.Linear(hidden_size, vocab_size)
#
#     def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
#         """
#
#         :param src: tensor, [batch_size, src_len]
#         :param tgt: tensor/None, [batch_size, tgt_len]
#         :param src_mask: tensor/None, [batch_size, src_len]; if None, no mask
#         :param tgt_mask: tensor/None, [batch_size, tgt_len]; if None, no mask
#         :return:
#         """
#         # [batch_size, src_len, emb_size]
#         src_emb = self.embedding(src)
#         # [batch_size, tgt_len, emb_size]
#         tgt_emb = self.embedding(tgt)
#         # EncoderOutput(seqEmb), more attr defined by users
#         memory = self.encoder(src=src_emb, src_mask=src_mask)
#         # DecoderOutput(seq_inf), more attr defined by users
#         outputs = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
#         return TextGenOutput(
#             pred = self.generator(outputs.seq_inf)
#         )


class Roberta2Transformer(nn.Module):
    def __init__(self, args):
        super(Roberta2Transformer, self).__init__()
        roberta_config = RobertaConfig.from_pretrained(args.pretrained_model)
        # self.tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model)
        self.encoder = RobertaModel.from_pretrained(args.pretrained_model)
        self.decoder = TransformerDecoder(d_model=roberta_config.hidden_size)
        self.generator = nn.Linear(roberta_config.hidden_size, roberta_config.vocab_size)

    def forward(self, encoded_src, encoded_tgt=None, encoder_hidden_states=False, decoder_hidden_states=True):
        """
        :param encoded_src: {'input_ids':[batch_size, src_len], 'token_type_ids':[batch_size, src_len],'attention_mask':[batch_size, src_len]}
        :param encoded_tgt: 和encoded_src类似
        :param encoder_hidden_states: 是否返回encoder的输出信息
        :param decoder_hidden_states: 是否返回decoder的输出信息
        :return: TextGenOutput
        """
        if not encoded_tgt:
            encoded_tgt = encoded_src
        src_input_ids, src_token_type_ids, src_attention_mask = encoded_src['input_ids'], \
                                                                encoded_src['token_type_ids'], \
                                                                encoded_src['attention_mask']  # 0:mask
        tgt_input_ids, tgt_token_type_ids, tgt_attention_mask = encoded_tgt['input_ids'], \
                                                                encoded_tgt['token_type_ids'], \
                                                                encoded_tgt['attention_mask']  # 0:mask
        # [batch_size, tgt_len, dim]
        embed_tgt = self.encoder.embeddings(input_ids=tgt_input_ids, token_type_ids=tgt_token_type_ids)
        encode_outputs = self.encoder(src_input_ids, src_attention_mask, src_token_type_ids)
        memory = encode_outputs.last_hidden_state
        decode_output = self.decoder(embed_tgt, memory, tgt_attention_mask==0)
        final = decode_output.last_hidden_state

        output =  TextGenOutput(
            pred = self.generator(final)  # [batch_size, tgt_len, voc_size]
        )
        if encoder_hidden_states:
            output.update(**{'encoder_'+k: v for k, v in encode_outputs.items()})
        if decoder_hidden_states:
            del decode_output.last_hidden_state
            output.update(**{'decoder_'+k: v for k, v in decode_output.items()})
        return output
