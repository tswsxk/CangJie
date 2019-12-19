# coding: utf-8
# 2019/12/19 @ tongshiwei

from tqdm import tqdm
import json
from longling import rf_open, wf_open
import csv


def csv2json(src, tar, delimiter=',', skip_first_line=False):
    with rf_open(src) as f, wf_open(tar) as wf:
        if skip_first_line:  # pragma: no cover
            f.readline()
        for line in tqdm(csv.reader(f, delimiter=delimiter), "csv2json: %s --> %s" % (src, tar)):
            token = line[0]
            vec = list(map(float, line[1:]))
            print(json.dumps([token, vec]), file=wf)

    return tar


def json2csv(src, tar, delimiter=','):
    with rf_open(src) as f, wf_open(tar) as wf:
        for line in f:
            token, vec = json.loads(line)
            print(("%s" % delimiter).join([token] + list(map(str, vec))), file=wf)

    return tar
