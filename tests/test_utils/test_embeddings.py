# coding: utf-8
# 2020/1/4 @ tongshiwei

from gluonnlp.embedding import TokenEmbedding
from CangJie.utils.embeddings import load_embedding


def test_load_embeddings(vec_csv, tmpdir):
    embeddings = load_embedding({"a": vec_csv, "b": vec_csv})
    assert "a" in embeddings
    assert "b" in embeddings

    assert isinstance(embeddings["a"], TokenEmbedding)
    assert isinstance(embeddings["b"], TokenEmbedding)
