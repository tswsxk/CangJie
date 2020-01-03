# coding: utf-8
# 2019/12/30 @ tongshiwei

import pytest
from CangJie.Features import Stroke, character_glyph, CDict
from CangJie import token2stroke, token2radical, char_features


def test_features():
    cdict = CDict.from_file()
    char_features("一")

    assert len(cdict.get_stroke("一s")) == 1
    assert len(cdict.get_radical(["一二", "三"])) == 2

    cdict = CDict.from_file(allow_missing=False)

    with pytest.raises(KeyError):
        assert len(cdict.get_stroke("一s")) == 1

    with pytest.raises(TypeError):
        print(cdict.get_stroke(123))


def test_stroke():
    stroke = Stroke.from_file()

    assert len(stroke["一"]) == 1
    assert len(stroke["一二"]) == 3
    assert len(stroke[["一", "二"]]) == 2

    with pytest.raises(TypeError):
        print(stroke[123])

    assert stroke["s"] == ""

    stroke = Stroke.from_file(allow_missing=False)
    with pytest.raises(KeyError):
        assert stroke["s"] == ""

    assert len(token2stroke("一s")) == 1


def test_radical():
    token2radical("一")


@pytest.mark.skip(reason="require simsun, which are usually unavailable in most testing platform")
def test_glyph():
    character_glyph("一")
