# coding: utf-8
# 2019/12/30 @ tongshiwei

from PIL import Image, ImageDraw, ImageFont

__all__ = ["character_glyph"]


def character_glyph(character, size=28):
    im = Image.new("1", (size, size), 0)
    text = character
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype("simsun", size)

    dr.text((0, 0), text, font=font, fill="white")

    return im
