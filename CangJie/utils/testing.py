# coding: utf-8
# 2020/1/2 @ tongshiwei

import random
from CangJie import CHI_CHAR

__all__ = ["pseudo_sentence"]


def _pseudo_sentence(length):
    return "".join([random.choice(CHI_CHAR) for _ in range(length)])


def pseudo_sentence(num, max_length):
    return [_pseudo_sentence(random.randint(1, max_length)) for _ in range(num)]
