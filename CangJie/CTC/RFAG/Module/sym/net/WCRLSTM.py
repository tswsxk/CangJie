# coding: utf-8
# 2020/1/2 @ tongshiwei
from longling.ML.MxnetHelper.gallery.layer.attention import \
    DotProductAttentionCell
import mxnet as mx
from mxnet import gluon

from .net import EmbeddingLSTM


class WCRLSTM(EmbeddingLSTM):
    def __init__(self,
                 net_type,
                 class_num, embedding_dim,
                 lstm_hidden=None, embedding_size=None,
                 embed_dropout=0.5, fc_dropout=0.5,
                 **kwargs):
        r"""Our method: 包含词和字，以及字、词部首的网络模型"""
        super(WCRLSTM, self).__init__(**kwargs)
        self.word_length = None
        self.character_length = None
        self.lstm_hidden = lstm_hidden if lstm_hidden is not None else embedding_dim
        self.net_type = net_type

        with self.name_scope():
            self.embedding = WCREmbedding(
                word_embedding_size=embedding_size["w"],
                word_radical_embedding_size=embedding_size["rw"],
                char_embedding_size=embedding_size["c"],
                char_radical_embedding_size=embedding_size["rc"],
                embedding_dim=embedding_dim,
                dropout=embed_dropout,
            )
            for i in range(4):
                if self.net_type in ("bilstm", "bilstm_att"):
                    setattr(self, "rnn%s" % i,
                            gluon.rnn.BidirectionalCell(
                                gluon.rnn.LSTMCell(self.lstm_hidden),
                                gluon.rnn.LSTMCell(self.lstm_hidden))
                            )
                elif self.net_type == "lstm":
                    setattr(
                        self, "rnn%s" % i, gluon.rnn.LSTMCell(self.lstm_hidden),
                    )
                else:
                    raise TypeError(
                        "net_type should be lstm, bilstm or bilstm_att,"
                        " now is %s" % self.net_type
                    )

            if self.net_type == "bilstm_att":
                self.word_attention = DotProductAttentionCell(
                    units=self.lstm_hidden, scaled=False
                )
                self.char_attention = DotProductAttentionCell(
                    units=self.lstm_hidden, scaled=False
                )

            self.dropout = gluon.nn.Dropout(fc_dropout)
            self.fc = gluon.nn.Dense(class_num)

    def hybrid_forward(self, F, w, rw, c, rc, word_mask, charater_mask, *args, **kwargs):
        word_length = self.word_length if self.word_length else len(w[0])
        character_length = self.character_length if self.character_length else len(
            c[0])

        word_embedding, word_radical_embedding, character_embedding, character_radical_embedding = self.embedding(w, rw,
                                                                                                                  c, rc)

        merge_outputs = True
        if F is mx.symbol:
            # 似乎 gluon 有问题， 对symbol
            word_mask, charater_mask = None, None

        if self.net_type == "bilstm_att":
            w_e, (w_lo, w_ls, w_ro, w_rs) = getattr(self, "rnn0").unroll(
                word_length, word_embedding,
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )
            word_radical_embedding = \
                self.word_attention(word_radical_embedding, w_e,
                                    mask=word_mask
                                    )[0]
            wr_e, (wr_lo, wr_ls, wr_ro, wr_rs) = getattr(self, "rnn1").unroll(
                word_length, word_radical_embedding,
                begin_state=(w_ls, w_ls, w_rs, w_rs),
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )
            c_e, (c_lo, c_ls, c_ro, c_rs) = getattr(self, "rnn2").unroll(
                character_length, character_embedding,
                merge_outputs=merge_outputs,
                valid_length=charater_mask
            )
            character_radical_embedding = \
                self.char_attention(character_radical_embedding, c_e,
                                    mask=charater_mask
                                    )[0]
            cr_e, (cr_lo, cr_ls, cr_ro, cr_rs) = getattr(self, "rnn3").unroll(
                character_length, character_radical_embedding,
                begin_state=(c_ls, c_ls, c_rs, c_rs),
                merge_outputs=merge_outputs,
                valid_length=charater_mask
            )

            attention = F.concat(
                w_lo, w_ro, wr_lo, wr_ro, c_lo, c_ro, cr_lo, cr_ro
            )
        elif self.net_type == "bilstm":
            w_e, (w_lo, w_ls, w_ro, w_rs) = getattr(self, "rnn0").unroll(
                word_length, word_embedding,
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )

            wr_e, (wr_lo, wr_ls, wr_ro, wr_rs) = getattr(self, "rnn1").unroll(
                word_length, word_radical_embedding,
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )
            c_e, (c_lo, c_ls, c_ro, c_rs) = getattr(self, "rnn2").unroll(
                character_length, character_embedding,
                merge_outputs=merge_outputs,
                valid_length=charater_mask
            )
            cr_e, (cr_lo, cr_ls, cr_ro, cr_rs) = getattr(self, "rnn3").unroll(
                character_length, character_radical_embedding,
                merge_outputs=merge_outputs,
                valid_length=charater_mask
            )

            attention = F.concat(
                w_lo, w_ro, wr_lo, wr_ro, c_lo, c_ro, cr_lo, cr_ro
            )
        elif self.net_type == "lstm":
            w_e, (w_o, w_s) = getattr(self, "rnn0").unroll(
                word_length,
                word_embedding,
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )
            wr_e, (wr_o, wr_s) = getattr(self, "rnn1").unroll(
                word_length,
                word_radical_embedding,
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )
            c_e, (c_o, c_s) = getattr(self, "rnn2").unroll(
                character_length,
                character_embedding,
                merge_outputs=merge_outputs,
                valid_length=charater_mask
            )
            cr_e, (cr_o, cr_s) = getattr(self, "rnn3").unroll(
                character_length, character_radical_embedding,
                merge_outputs=merge_outputs,
                valid_length=charater_mask)

            attention = F.concat(w_o, wr_o, c_o, cr_o)
        else:
            raise TypeError(
                "net_type should be lstm, bilstm or bilstm_att,"
                " now is %s" % self.net_type
            )
        attention = self.dropout(attention)
        # attention = self.dropout(
        #     F.Activation(self.nn(attention), act_type='softrelu')
        # )
        # fc_in = self.layers_attention(attention)
        fc_in = attention
        return self.fc(fc_in)

    def set_network_unroll(self, word_length, character_length):
        self.word_length = word_length
        self.character_length = character_length


class WCREmbedding(gluon.HybridBlock):
    def __init__(self, word_embedding_size, word_radical_embedding_size,
                 char_embedding_size, char_radical_embedding_size,
                 embedding_dim, dropout=0.5, prefix=None,
                 params=None):
        super(WCREmbedding, self).__init__(prefix, params)
        with self.name_scope():
            self.word_embedding = gluon.nn.Embedding(
                word_embedding_size, embedding_dim
            )
            self.word_radical_embedding = gluon.nn.Embedding(
                word_radical_embedding_size, embedding_dim
            )
            self.char_embedding = gluon.nn.Embedding(
                char_embedding_size, embedding_dim
            )
            self.char_radical_embedding = gluon.nn.Embedding(
                char_radical_embedding_size, embedding_dim
            )
            self.word_dropout = gluon.nn.Dropout(dropout)
            self.char_dropout = gluon.nn.Dropout(dropout)

    def hybrid_forward(self,
                       F, word_seq, word_radical_seq,
                       character_seq, character_radical_seq,
                       *args, **kwargs
                       ):
        word_embedding = self.word_embedding(word_seq)
        word_radical_embedding = self.word_radical_embedding(word_radical_seq)
        character_embedding = self.char_embedding(character_seq)
        character_radical_embedding = self.char_radical_embedding(
            character_radical_seq
        )

        word_embedding = self.word_dropout(word_embedding)
        word_radical_embedding = self.word_dropout(word_radical_embedding)
        character_embedding = self.char_dropout(character_embedding)
        character_radical_embedding = self.char_dropout(
            character_radical_embedding
        )

        return word_embedding, word_radical_embedding, character_embedding, character_radical_embedding

    def set_weight(self, we, rwe, ce, rce):
        self.word_embedding.weight.set_data(we)
        self.word_radical_embedding.weight.set_data(rwe)
        self.char_embedding.weight.set_data(ce)
        self.char_radical_embedding.weight.set_data(rce)
