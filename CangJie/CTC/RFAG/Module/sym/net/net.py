# coding: utf-8
# 2020/1/2 @ tongshiwei

from mxnet import gluon


class EmbeddingLSTM(gluon.HybridBlock):
    def hybrid_forward(self, F, w, rw, c, rc, word_mask, charater_mask, *args, **kwargs):
        raise NotImplementedError
