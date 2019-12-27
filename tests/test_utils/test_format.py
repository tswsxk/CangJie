# coding: utf-8
# 2019/12/19 @ tongshiwei

from longling import path_append
from CangJie.utils.format import json2csv, csv2json, gensim2json
from CangJie.utils.VecDict import VecDict


def test_json_csv(vec_json, tmpdir):
    vec_csv = json2csv(vec_json, path_append(tmpdir, "vec.csv", to_str=True))
    vec_dict = VecDict.load_from_vec_json(csv2json(vec_csv, path_append(tmpdir, "vec.json", to_str=True)))

    assert vec_dict["hello"] == [1., 1., 1.]


def test_gensim(tmpdir):
    from gensim.test.utils import datapath
    from gensim import utils

    class MyCorpus(object):
        """An interator that yields sentences (lists of str)."""

        def __iter__(self):
            corpus_path = datapath('lee_background.cor')
            for line in open(corpus_path):
                # assume there's one document per line, tokens separated by whitespace
                yield utils.simple_preprocess(line)

    import gensim.models

    sentences = MyCorpus()
    model = gensim.models.Word2Vec(sentences=sentences, iter=1)

    gensim_model_bin = path_append(tmpdir, "model.bin", to_str=True)
    vec_json = path_append(tmpdir, "vec.json", to_str=True)
    model.save(gensim_model_bin)

    gensim2json(gensim_model_bin, vec_json)

    vec_dict = VecDict.load_from_vec_json(vec_json)

    for word in model.wv.vocab:
        assert model.wv[word].tolist() == vec_dict[word]
        break
