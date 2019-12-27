# coding: utf-8
# 2019/12/19 @ tongshiwei

import fire

from CangJie.utils import token2idx
from CangJie.utils.format import csv2json, json2csv


def cli():  # pragma: no cover
    fire.Fire({
        "format": {
            "csv2json": csv2json,
            "json2csv": json2csv,
        },
        "utils": {
            "token2idx": token2idx,
        }
    })


if __name__ == '__main__':
    cli()
