# coding: utf-8
# 2020/1/3 @ tongshiwei


__all__ = ["get_net", "get_bp_loss"]

# todo: define your network symbol and back propagation loss function

from mxnet import gluon


def get_net(**kwargs):
    # return NetName(**kwargs)
    pass


def get_bp_loss(**kwargs):
    return {"L2Loss": gluon.loss.L2Loss(**kwargs)}
