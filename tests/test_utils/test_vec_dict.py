# coding: utf-8
# 2019/12/19 @ tongshiwei

import json
import pytest
from CangJie.utils import WVDict, seq2idx
from longling import rf_open


def test_vec_dict(vec_json):
    vec_dict = WVDict.from_file(vec_json)

    assert vec_dict.size == 4
    assert vec_dict.dim == 3

    assert vec_dict["hello"] == [1., 1., 1.]
    assert vec_dict.idx2token(vec_dict.token2idx("龙")) == "龙"

    assert vec_dict[["hello", "仓颉"]] == [[1., 1., 1.], [0., 0., 0.]]
    assert vec_dict.idx2token(vec_dict.token2idx(["hello", "仓颉"])) == ["hello", "仓颉"]

    with pytest.raises(TypeError):
        vec_dict.idx2token("hello")

    with pytest.raises(TypeError):
        vec_dict.token2idx(1)

    with pytest.raises(TypeError):
        vec_dict.idx2vec("hello")

    with pytest.raises(TypeError):
        vec_dict.token2vec(1)


def test_token2idx(token_seq, vec_json):
    seq2idx(token_seq, token_seq + ".idx", vec_json)

    with rf_open(token_seq + ".idx") as f:
        assert json.loads(f.readline()) == [0, 1]
        assert json.loads(f.readline()) == [2]
