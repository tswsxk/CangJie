# coding: utf-8
# 2020/1/4 @ tongshiwei

from gluonnlp.embedding import TokenEmbedding
import logging
from longling import print_time


def load_embedding(embeddings: dict, logger=None) -> dict:
    logger = logging.getLogger("RFAG") if logger is None else logger
    assert isinstance(embeddings, dict)

    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(len(embeddings))
    _ret = dict()

    with print_time(logger=logger, tips='loading embedding'):
        for k, v in embeddings.items():
            _ret[k] = pool.apply_async(TokenEmbedding.from_file, args=(v,))

    pool.close()
    pool.join()

    for k, v in _ret.items():
        _ret[k] = v.get()

    return _ret
