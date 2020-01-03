# coding: utf-8
# 2019/12/19 @ tongshiwei

from tqdm import tqdm
import json
from longling import rf_open, wf_open
import csv


def csv2json(src, tar, delimiter=' ', skip_first_line=False):
    with rf_open(src) as f, wf_open(tar) as wf:
        if skip_first_line:  # pragma: no cover
            f.readline()
        for line in tqdm(csv.reader(f, delimiter=delimiter), "csv2json: %s --> %s" % (src, tar)):
            token = line[0]
            vec = list(map(float, line[1:]))
            print(json.dumps([token, vec]), file=wf)
    return tar


def json2csv(src, tar, delimiter=' '):
    with rf_open(src) as f, wf_open(tar) as wf:
        writer = csv.writer(wf, delimiter=delimiter)
        for line in f:
            token, vec = json.loads(line)
            writer.writerow([token] + list(map(str, vec)))
    return tar


def _load_gensim(src):
    import gensim
    import logging

    logging.getLogger("CangJie").info("loading gensim model from %s" % src)
    model = gensim.models.Word2Vec.load(src)

    return model


def gensim2json(src, tar):
    model = _load_gensim(src)

    with wf_open(tar) as wf:
        for word in tqdm(model.wv.vocab, "gensim2json: %s --> %s" % (src, tar)):
            print(json.dumps([word, model.wv[word].tolist()]), file=wf)


def gensim2csv(src, tar, delimiter=" "):
    model = _load_gensim(src)

    with wf_open(tar) as wf:
        writer = csv.writer(wf, delimiter=delimiter)
        for word in tqdm(model.wv.vocab, "gensim2json: %s --> %s" % (src, tar)):
            writer.writerow([word] + model.wv[word].tolist())
