# coding: utf-8
# 2020/1/4 @ tongshiwei

import os
import csv
import warnings
import mxnet.ndarray as nd
from gluonnlp.embedding import TokenEmbedding as GluonNLPTE
import logging
from longling import print_time, rf_open
from CangJie.constants import UNK_TOKEN


class TokenEmbedding(GluonNLPTE):  # pragma: no cover
    """Hack GLuonNlP TokenEmbedding"""

    @staticmethod
    def from_file(file_path, elem_delim=' ', encoding='utf8', **kwargs):
        """Creates a user-defined token embedding from a pre-trained embedding file.


        This is to load embedding vectors from a user-defined pre-trained token embedding file.
        For example, if `elem_delim` = ' ', the expected format of a custom pre-trained token
        embedding file may look like:

        'hello 0.1 0.2 0.3 0.4 0.5\\\\nworld 1.1 1.2 1.3 1.4 1.5\\\\n'

        where embedding vectors of words `hello` and `world` are [0.1, 0.2, 0.3, 0.4, 0.5] and
        [1.1, 1.2, 1.3, 1.4, 1.5] respectively.


        Parameters
        ----------
        file_path : str
            The path to the user-defined pre-trained token embedding file.
        elem_delim : str, default ' '
            The delimiter for splitting a token and every embedding vector element value on the same
            line of the custom pre-trained token embedding file.
        encoding : str, default 'utf8'
            The encoding scheme for reading the custom pre-trained token embedding file.
        kwargs : dict
            All other keyword arguments are passed to the TokenEmbedding initializer.


        Returns
        -------
        instance of :class:`gluonnlp.embedding.TokenEmbedding`
            The user-defined token embedding instance.
        """
        unknown_token = kwargs.pop('unknown_token', UNK_TOKEN)
        init_unknown_vec = kwargs.pop('init_unknown_vec', nd.zeros)
        idx_to_token, idx_to_vec, unknown_token = TokenEmbedding._load_embedding(
            file_path,
            elem_delim=elem_delim,
            unknown_token=unknown_token,
            init_unknown_vec=init_unknown_vec,
            encoding=encoding)

        assert 'idx_to_vec' not in kwargs
        assert 'idx_to_token' not in kwargs
        return TokenEmbedding(unknown_token=unknown_token,
                              init_unknown_vec=None,
                              idx_to_token=idx_to_token,
                              idx_to_vec=idx_to_vec,
                              **kwargs)

    @staticmethod
    def _load_embedding(pretrained_file_path, elem_delim, unknown_token,
                        init_unknown_vec, encoding='utf8'):
        """Load embedding vectors from a pre-trained token embedding file.

        Both text files and TokenEmbedding serialization files are supported.
        elem_delim and encoding are ignored for non-text files.

        For every unknown token, if its representation `self.unknown_token` is encountered in the
        pre-trained token embedding file, index 0 of `self.idx_to_vec` maps to the pre-trained token
        embedding vector loaded from the file; otherwise, index 0 of `self.idx_to_vec` maps to the
        text embedding vector initialized by `self._init_unknown_vec`.

        If a token is encountered multiple times in the pre-trained text embedding file, only the
        first-encountered token embedding vector will be loaded and the rest will be skipped.

        """

        pretrained_file_path = os.path.expanduser(pretrained_file_path)

        if not os.path.isfile(pretrained_file_path):
            raise ValueError('`pretrained_file_path` must be a valid path '
                             'to the pre-trained token embedding file.')

        logging.info('Loading pre-trained token embedding vectors from %s',
                     pretrained_file_path)

        if pretrained_file_path.endswith('.npz'):
            return TokenEmbedding._load_embedding_serialized(
                pretrained_file_path=pretrained_file_path,
                unknown_token=unknown_token,
                init_unknown_vec=init_unknown_vec)
        else:
            return TokenEmbedding._load_embedding_txt(
                pretrained_file_path=pretrained_file_path,
                elem_delim=elem_delim,
                unknown_token=unknown_token,
                init_unknown_vec=init_unknown_vec,
                encoding=encoding)

    @staticmethod
    def _load_embedding_txt(pretrained_file_path, elem_delim, unknown_token,
                            init_unknown_vec, encoding='utf8'):
        """Load embedding vectors from a pre-trained token embedding file.

        Returns idx_to_token, idx_to_vec and unknown_token suitable for the
        TokenEmbedding constructor.

        For every unknown token, if its representation `unknown_token` is encountered in the
        pre-trained token embedding file, index 0 of `idx_to_vec` maps to the pre-trained token
        embedding vector loaded from the file; otherwise, index 0 of `idx_to_vec` maps to the
        text embedding vector initialized by `init_unknown_vec`.

        If a token is encountered multiple times in the pre-trained text embedding file, only the
        first-encountered token embedding vector will be loaded and the rest will be skipped.

        """
        idx_to_token = [unknown_token] if unknown_token else []
        unk_idx = None
        if unknown_token:
            unk_idx = 0

        vec_len = None
        all_elems = []
        tokens = set()
        loaded_unknown_vec = None
        with rf_open(pretrained_file_path, encoding=encoding) as f:
            for line_num, elems in enumerate(csv.reader(f, delimiter=elem_delim)):

                assert len(elems) > 1, 'line {} in {}: unexpected data format.'.format(
                    line_num, pretrained_file_path)

                token, elems = elems[0], [float(i) for i in elems[1:]]

                if loaded_unknown_vec is None and token == unknown_token:
                    loaded_unknown_vec = elems
                    tokens.add(unknown_token)
                elif token in tokens:
                    warnings.warn('line {} in {}: duplicate embedding found for '
                                  'token "{}". Skipped.'.format(line_num, pretrained_file_path,
                                                                token))
                elif len(elems) == 1 and line_num == 0:
                    warnings.warn('line {} in {}: skipped likely header line.'
                                  .format(line_num, pretrained_file_path))
                else:
                    if not vec_len:
                        vec_len = len(elems)
                        if unknown_token:
                            # Reserve a vector slot for the unknown token at the very beggining
                            # because the unknown token index is 0.
                            assert len(all_elems) == 0
                            all_elems.extend([0] * vec_len)
                    else:
                        assert len(elems) == vec_len, \
                            'line {} in {}: found vector of inconsistent dimension for token ' \
                            '"{}". expected dim: {}, found: {}'.format(line_num,
                                                                       pretrained_file_path,
                                                                       token, vec_len, len(elems))
                    all_elems.extend(elems)
                    idx_to_token.append(token)
                    tokens.add(token)

        idx_to_vec = nd.array(all_elems).reshape((-1, vec_len))

        if unknown_token:
            if loaded_unknown_vec is None:
                idx_to_vec[unk_idx] = init_unknown_vec(shape=vec_len)
            else:
                idx_to_vec[unk_idx] = nd.array(loaded_unknown_vec)

        return idx_to_token, idx_to_vec, unknown_token


def load_embedding(embeddings: dict, logger=None) -> dict:
    assert isinstance(embeddings, dict)

    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(len(embeddings))
    _ret = dict()

    for k, v in embeddings.items():
        _ret[k] = pool.apply_async(TokenEmbedding.from_file, args=(v,))

    pool.close()
    pool.join()

    for k, v in _ret.items():
        _ret[k] = v.get()

    return _ret


def get_embedding_size(embeddings: dict):
    _ret = dict()
    for k, v in embeddings.items():
        _ret[k] = len(v.idx_to_vec)

    return _ret


def get_embedding_array(embeddings: dict):
    _ret = dict()
    for k, v in embeddings.items():
        _ret[k] = v.idx_to_vec

    return _ret


def token_to_idx(embedding: GluonNLPTE, token: (str, list)):
    if isinstance(token, str):
        return embedding.token_to_idx[token]
    elif isinstance(token, list):
        return [token_to_idx(embedding, e) for e in token]
    else:
        raise TypeError("cannot handle %s" % type(token))
