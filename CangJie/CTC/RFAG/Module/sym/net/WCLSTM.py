# coding: utf-8
# 2020/1/2 @ tongshiwei
from longling.ML.MxnetHelper.gallery.layer.attention import \
    DotProductAttentionCell
from mxnet import gluon
from .net import EmbeddingLSTM


class WCLSTM(EmbeddingLSTM):
    def __init__(self, net_type,
                 class_num, embedding_dim,
                 lstm_hidden=None,
                 embed_dropout=0.5, fc_dropout=0.5, embedding_size=None,
                 **kwargs):
        r"""Baseline: 仅包含词和字，不包括部首的网络模型"""
        super(WCLSTM, self).__init__(**kwargs)
        self.word_length = None
        self.character_length = None
        self.lstm_hidden = lstm_hidden if lstm_hidden is not None else embedding_dim
        self.net_type = net_type

        with self.name_scope():
            self.embedding = WCEmbedding(
                word_embedding_size=embedding_size["w"],
                char_embedding_size=embedding_size["c"],
                embedding_dim=embedding_dim,
                dropout=embed_dropout,
            )
            for i in range(2):
                if self.net_type == "lstm":
                    setattr(
                        self, "rnn%s" % i,
                        gluon.rnn.LSTMCell(self.lstm_hidden)
                    )
                elif self.net_type == "bilstm":
                    setattr(
                        self, "rnn%s" % i,
                        gluon.rnn.BidirectionalCell(
                            gluon.rnn.LSTMCell(self.lstm_hidden),
                            gluon.rnn.LSTMCell(self.lstm_hidden)
                        )
                    )
                else:
                    raise TypeError(
                        "net_type should be either lstm or bilstm, now is %s"
                        % self.net_type
                    )

            self.dropout = gluon.nn.Dropout(fc_dropout)
            self.fc = gluon.nn.Dense(class_num)

    def hybrid_forward(self, F, w, rw, c, rc, word_mask, charater_mask, *args, **kwargs):
        word_length = self.word_length if self.word_length else len(
            w[0]
        )
        character_length = self.character_length if self.character_length \
            else len(c[0])

        word_embedding, character_embedding = self.embedding(
            w, c,
        )

        merge_outputs = True
        if self.net_type == "lstm":
            w_e, (w_o, w_s) = getattr(self, "rnn0").unroll(
                word_length,
                word_embedding,
                merge_outputs=merge_outputs,
                valid_length=word_mask
            )

            c_e, (c_o, c_s) = getattr(self, "rnn1").unroll(
                character_length,
                character_embedding,
                merge_outputs=merge_outputs,
                valid_length=charater_mask
            )

            attention = F.concat(w_o, c_o)
        elif self.net_type == "bilstm":
            w_e, (w_lo, w_ls, w_ro, w_rs) = getattr(self, "rnn0").unroll(
                word_length, word_embedding, merge_outputs=merge_outputs,
                valid_length=word_mask)

            c_e, (c_lo, c_ls, c_ro, c_rs) = getattr(self, "rnn1").unroll(
                character_length, character_embedding,
                merge_outputs=merge_outputs,
                valid_length=charater_mask)

            attention = F.concat(w_lo, c_lo, w_ro, c_ro)
        else:
            raise TypeError(
                "net_type should be either lstm or bilstm, now is %s"
                % self.net_type
            )
        # attention = F.concat(w_o, c_o)

        # attention = self.dropout(
        #     F.Activation(self.nn(attention), act_type='softrelu')
        # )
        # fc_in = self.layers_attention(attention)
        fc_in = self.dropout(attention)
        return self.fc(fc_in)

    def set_network_unroll(self, word_length, character_length):
        self.word_length = word_length
        self.character_length = character_length


class WCEmbedding(gluon.HybridBlock):
    def __init__(self, word_embedding_size,
                 char_embedding_size, embedding_dim, dropout=0.5, prefix=None,
                 params=None):
        super(WCEmbedding, self).__init__(prefix, params)
        with self.name_scope():
            self.word_embedding = gluon.nn.Embedding(
                word_embedding_size, embedding_dim
            )
            self.char_embedding = gluon.nn.Embedding(
                char_embedding_size, embedding_dim
            )
            self.word_dropout = gluon.nn.Dropout(dropout)
            self.char_dropout = gluon.nn.Dropout(dropout)

    def hybrid_forward(self, F, word_seq, character_seq, *args, **kwargs):
        word_embedding = self.word_embedding(word_seq)
        character_embedding = self.char_embedding(character_seq)

        word_embedding = self.word_dropout(word_embedding)
        character_embedding = self.char_dropout(character_embedding)

        return word_embedding, character_embedding

    def set_weight(self, embeddings):
        self.word_embedding.weight.set_data(embeddings["w"])
        self.char_embedding.weight.set_data(embeddings["c"])
