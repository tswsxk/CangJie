# coding: utf-8
# 2019/12/19 @ tongshiwei


if __name__ == '__main__':
    try:
        raise TypeError("???")
    except ModuleNotFoundError:
        print("m")
    except Exception as e:
        print(e)
