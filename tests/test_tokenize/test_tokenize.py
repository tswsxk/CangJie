# coding: utf-8
# 2020/1/3 @ tongshiwei

import pytest
from CangJie import tokenize, characterize


def test_tokenize():
    assert "".join(tokenize("再给我两分钟")) == "再给我两分钟"
    assert characterize("后会无期") == ["后", "会", "无", "期"]
    assert characterize(["后会", "无期"]) == ["后", "会", "无", "期"]

    with pytest.raises(TypeError):
        characterize(123)
