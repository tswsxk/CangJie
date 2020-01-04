# coding: utf-8
# 2019/12/19 @ tongshiwei

import fire

from CangJie.utils import seq2idx
from CangJie.utils.format import csv2json, json2csv, gensim2json, gensim2csv
from CangJie.Features import char_features as _char_features
from pprint import pprint


def char_features(token: (str, list), **kwargs):  # pragma: no cover
    pprint(_char_features(token, **kwargs))


def cli():  # pragma: no cover
    fire.Fire({
        "csv2json": csv2json,
        "json2csv": json2csv,
        "gensim2json": gensim2json,
        "gensim2csv": gensim2csv,
        "utils": {
            "seq2idx": seq2idx,
        },
        "char_features": char_features,
    })


if __name__ == '__main__':
    cli()
