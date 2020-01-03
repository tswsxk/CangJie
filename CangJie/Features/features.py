# coding: utf-8
# 2019/12/30 @ tongshiwei

from longling import path_append, loading
import pathlib
from .constants import CHAR, STROKE, RADICAL, GLYPH, PRON
from .glyph import character_glyph
from .stroke import token2stroke

__all__ = ["load_dict", "CDict", "token2radical", "char_features"]

DEFAULT_DICT = path_append(pathlib.PurePath(__file__).parents[2], "meta_data", "cdict.csv", to_str=True)
DEFAULT_CDICT = [None]


def load_dict(dict_path):
    cdict = {}
    for line in loading(dict_path):
        cdict[line[CHAR]] = {}
        cdict[line[CHAR]][RADICAL] = line[RADICAL]
    return cdict


class CDict(object):
    def __init__(self, dict_obj=None, allow_missing=True):
        self._dict = dict_obj
        self._allow_missing = allow_missing

    @staticmethod
    def from_file(dict_path=DEFAULT_DICT, allow_missing=True):
        return CDict(load_dict(dict_path), allow_missing=allow_missing)

    def _get_char_features(self, char, stroke=False, radical=False, pron=False, glyph=False):
        return self.get_char_features(char, stroke, radical, pron, glyph)

    def get_char_features(self, char, stroke=True, radical=True, pron=True, glyph=True, **kwargs):
        assert isinstance(char, str) and len(char) == 1
        ret = {}
        if stroke:
            ret[STROKE] = token2stroke(char, self._allow_missing)
        if radical:
            ret[RADICAL] = self._dict[char][RADICAL]
        if glyph:  # pragma: no cover
            ret[GLYPH] = character_glyph(char)
        return ret

    def get_word_features(self, token: (str, list)):
        raise NotImplementedError

    def _get(self, token: (str, list), feature: str, **kwargs):
        if isinstance(token, str):
            if len(token) == 1:
                try:
                    kwargs.update({feature: True})
                    return self._get_char_features(token, **kwargs)[eval(feature.upper())]
                except KeyError as e:
                    if self._allow_missing:  # pragma: no cover
                        return ""
                    else:
                        raise e
            else:
                return "".join([self._get(_token, feature, **kwargs) for _token in token])
        elif isinstance(token, list):
            return [self._get(_token, feature, **kwargs) for _token in token]
        else:
            raise TypeError("cannot handle %s" % type(token))

    def get_radical(self, token):
        return self._get(token, "radical")

    def get_stroke(self, token):
        return self._get(token, "stroke")

    def get_glyph(self, token, size=28):  # pragma: no cover
        return self._get(token, "glyph", size=size)

    def get_pron(self, token):
        raise NotImplementedError


def _get_cdict() -> CDict:
    if DEFAULT_CDICT[0] is None:
        DEFAULT_CDICT[0] = CDict.from_file(DEFAULT_DICT)
    return DEFAULT_CDICT[0]


def token2radical(token: (str, list)):
    return _get_cdict().get_radical(token)


def char_features(token: (str, list), stroke=True, radical=True, pron=True, glyph=False):
    return _get_cdict().get_char_features(token, stroke=stroke, radical=radical, pron=pron, glyph=glyph)
