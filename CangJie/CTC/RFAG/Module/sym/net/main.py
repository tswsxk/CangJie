# coding: utf-8
# 2020/1/3 @ tongshiwei


__all__ = ["get_net", "get_bp_loss"]

from mxnet import gluon
from .WCLSTM import WCLSTM
from .WCRLSTM import WCRLSTM
from .WRCLSTM import WRCLSTM


def get_net(model_type, class_num, embedding_dim, net_type="lstm", **kwargs):
    if model_type == "wclstm":
        return WCLSTM(net_type=net_type, class_num=class_num, embedding_dim=embedding_dim, **kwargs)
    elif model_type == "wcrlstm":
        return WCRLSTM(net_type=net_type, class_num=class_num, embedding_dim=embedding_dim, **kwargs)
    elif model_type == "wrclstm":
        return WRCLSTM(net_type=net_type, class_num=class_num, embedding_dim=embedding_dim, **kwargs)
    else:
        raise TypeError("unknown model_type: %s" % model_type)


def get_bp_loss(**kwargs):
    return {"cross-entropy": gluon.loss.SoftmaxCrossEntropyLoss()}
