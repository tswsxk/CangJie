# coding: utf-8
# 2019/12/19 @ tongshiwei

import pytest
from CangJie.utils import VecDict


def test_vec_dict(vec_json):
    vec_dict = VecDict(vec_json)

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
