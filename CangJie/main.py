# coding: utf-8
# 2019/12/19 @ tongshiwei

import fire

from CangJie.utils import seq2idx
from CangJie.utils.format import csv2json, json2csv, gensim2json, gensim2csv


def cli():  # pragma: no cover
    fire.Fire({
        "format": {
            "csv2json": csv2json,
            "json2csv": json2csv,
            "gensim2json": gensim2json,
            "gensim2csv": gensim2csv
        },
        "utils": {
            "seq2idx": seq2idx,
        }
    })


if __name__ == '__main__':
    cli()
