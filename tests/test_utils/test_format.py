# coding: utf-8
# 2019/12/19 @ tongshiwei

from longling import path_append
from CangJie.utils.format import json2csv, csv2json
from CangJie.utils.VecDict import VecDict


def test_json_csv(vec_json, tmpdir):
    vec_csv = json2csv(vec_json, path_append(tmpdir, "vec.csv", to_str=True))
    vec_dict = VecDict(csv2json(vec_csv, path_append(tmpdir, "vec.json", to_str=True)))

    assert vec_dict["hello"] == [1., 1., 1.]
