# coding: utf-8
# 2019/12/19 @ tongshiwei

from longling import rf_open
from tqdm import tqdm
import json

__all__ = ["load_vec_json", "VecDict"]


def load_vec_json(vec_json) -> tuple:
    _token2idx = {}
    _idx2token = []
    _idx2vec = []
    _dim = None
    with rf_open(vec_json) as f:
        for i, line in tqdm(enumerate(f), "loading %s" % vec_json):
            print(line)
            _word, _vec = json.loads(line)
            assert _word not in _token2idx, "duplicate: %s" % _word
            if _dim is None:
                _dim = len(_vec)
                assert _dim > 0, "empty vec %s" % _word
            else:
                assert len(_vec) == _dim, "dimension inconsistent %s vs %s: %s" % (_dim, _word, len(_vec))
            _token2idx[_word] = i
            _idx2vec.append(_vec)
            _idx2token.append(_word)

    return _token2idx, _idx2vec, _idx2token


class VecDict(object):
    def __init__(self, vec_json):
        self._token2idx, self._idx2vec, self._idx2token = load_vec_json(vec_json)

        assert self._token2idx

    @property
    def size(self):
        return len(self._token2idx)

    @property
    def dim(self):
        return len(self._idx2vec[0])

    def idx2vec(self, idx):
        return self._idx2vec[idx]

    def token2idx(self, token):
        return self._token2idx[token]

    def idx2token(self, idx):
        return self._idx2token[idx]

    def token2vec(self, token):
        return self.idx2vec(self.token2idx(token))

    def __getitem__(self, item):
        return self.token2vec(item)
