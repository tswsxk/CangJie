# coding: utf-8
# 2019/12/19 @ tongshiwei

from longling import path_append, wf_open
import pytest
import json

_tested_vec_json = [
    ["hello", [1., 1., 1.]],
    ["仓颉", [0., 0., 0.]],
    ["龙", [1., 1., 0.]],
    [" ", [0., 0., 0.]]
]


@pytest.fixture(scope="module")
def utils_test_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module")
def vec_json(utils_test_dir):
    _vec_json = path_append(utils_test_dir, "vec.json", to_str=True)
    with wf_open(_vec_json) as wf:
        for _vec in _tested_vec_json:
            print(_vec)
            print(json.dumps(_vec), file=wf)
    return _vec_json
