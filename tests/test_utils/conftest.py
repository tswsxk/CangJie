# coding: utf-8
# 2019/12/19 @ tongshiwei

from longling import path_append, wf_open
import pytest
import json
from CangJie.utils.format import json2csv

_tested_vec_json = [
    ["hello", [1., 1., 1.]],
    ["仓颉", [0., 0., 0.]],
    ["龙", [1., 1., 0.]],
    ["枫叶", [0., 0., 0.]]
]

_test_token_sequence = [
    ["hello", "仓颉"],
    ["龙"]
]


@pytest.fixture(scope="module")
def utils_test_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module")
def vec_json(utils_test_dir):
    _vec_json = path_append(utils_test_dir, "vec.json", to_str=True)
    with wf_open(_vec_json) as wf:
        for _vec in _tested_vec_json:
            print(json.dumps(_vec), file=wf)
    return _vec_json


@pytest.fixture(scope="module")
def vec_csv(utils_test_dir, vec_json):
    _vec_csv = path_append(utils_test_dir, "vec.csv", to_str=True)
    json2csv(vec_json, _vec_csv)
    return _vec_csv


@pytest.fixture(scope="module")
def token_seq(utils_test_dir):
    _token_seq = path_append(utils_test_dir, "token.seq", to_str=True)
    with wf_open(_token_seq) as wf:
        for _seq in _test_token_sequence:
            print(json.dumps(_seq), file=wf)
    return _token_seq
