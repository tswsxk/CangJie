# coding: utf-8
# 2020/1/2 @ tongshiwei

import jieba

__all__ = ["tokenize", "characterize"]

tokenize = jieba.cut


def characterize(sentence: (str, list)) -> list:
    if isinstance(sentence, str):
        return [c for c in sentence]
    elif isinstance(sentence, list):
        ret = []
        for s in sentence:
            ret.extend([c for c in s])
        return ret
    else:
        raise TypeError("cannot handle %s" % type(sentence))
