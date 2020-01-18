# coding: utf-8
# 2019/12/28 @ tongshiwei

import json
from tqdm import tqdm
from longling import rf_open, wf_open

from gluonnlp.embedding import TokenEmbedding


def raw2processed(src, tar):
    with rf_open(src) as f, wf_open(tar) as wf:
        for line in tqdm(src, "raw to processed: %s --> %s" % (src, tar)):
            _data = json.loads(line)
            raw = _data["raw"]
            label = _data["label"]
            data = {
                "w": [],
                "c": [c for c in raw],
                "rw": [],
                "rc": [],
                "label": 0,
            }
            print(json.dumps(data), file=wf)


def processed2mature(src, tar):
    pass


def raw2mature(src, tar):
    pass
