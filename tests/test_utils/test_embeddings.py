# coding: utf-8
# 2020/1/4 @ tongshiwei

import pytest
from gluonnlp.embedding import TokenEmbedding
from CangJie.utils.embeddings import load_embedding, token_to_idx, get_embedding_size, get_embedding_array


def test_load_embeddings(vec_csv, tmpdir):
    embeddings = load_embedding({"a": vec_csv, "b": vec_csv})
    assert "a" in embeddings
    assert "b" in embeddings

    assert isinstance(embeddings["a"], TokenEmbedding)
    assert isinstance(embeddings["b"], TokenEmbedding)

    with pytest.raises(TypeError):
        token_to_idx(embeddings["a"], [123, 456])

    token_to_idx(embeddings["a"], ["仓颉", "龙"])

    get_embedding_size(embeddings)
    get_embedding_array(embeddings)
