# coding: utf-8
# 2019/12/30 @ tongshiwei

import json
from tqdm import tqdm
import pathlib
from longling import path_append, loading

__all__ = ["Stroke", "token2stroke"]

META = path_append(pathlib.PurePath(__file__).parents[2], "meta_data")
DEFAULT_STROKE = [None]


def load_stroke_dict(dict_path) -> dict:
    stroke_dict = {}
    for line in tqdm(loading(dict_path), "loading from %s" % dict_path):
        stroke_dict[line["c"]] = line["s"]
    return stroke_dict


class Stroke(object):
    def __init__(self, stroke_dict, allow_missing=True, stype=None):
        self._dict = stroke_dict
        self._allow_missing = allow_missing
        self.stype = stype

    @staticmethod
    def from_file(dict_path=None, allow_missing=True):
        dict_path = path_append(META, "stroke.json") if dict_path is None else dict_path
        return Stroke(load_stroke_dict(dict_path), allow_missing=allow_missing)

    def token2stroke(self, token: (str, list), allow_missing=None):
        _allow_missing = allow_missing if allow_missing is not None else self._allow_missing
        if isinstance(token, str):
            if len(token) == 1:
                try:
                    return self._dict[token] if self.stype is None else self._dict[token][self.stype]
                except KeyError as e:
                    if _allow_missing:
                        return ""
                    raise e

            else:
                return "".join([self.token2stroke(_token) for _token in token])
        elif isinstance(token, list):
            return [self.token2stroke(_token) for _token in token]
        else:
            raise TypeError("cannot handle %s" % type(token))

    def __getitem__(self, item):
        return self.token2stroke(item)


def token2stroke(token: (str, list), allow_missing=None):
    if DEFAULT_STROKE[0] is None:
        DEFAULT_STROKE[0] = Stroke.from_file()
    return DEFAULT_STROKE[0].token2stroke(token, allow_missing)
