# coding: utf-8
# 2020/1/7 @ tongshiwei


def parse_vec_files(vec_files_str: str):
    """
    Examples
    --------
    >>> parse_vec_files("w:123,c:456")
    {'w': '123', 'c': '456'}
    """
    _ret = dict()
    vec_key_value = vec_files_str.split(",")
    for key_value in vec_key_value:
        key, value = key_value.split(":")
        _ret[key] = value
    return _ret
