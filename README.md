# CangJie
[![Build Status](https://www.travis-ci.org/tswsxk/CangJie.svg?branch=master)](https://www.travis-ci.org/tswsxk/CangJie)
[![codecov](https://codecov.io/gh/tswsxk/CangJie/branch/master/graph/badge.svg)](https://codecov.io/gh/tswsxk/CangJie)

针对现有的多种自然语言工具包资源进行整合


## Chinese Text Classification

Example raw data

* `raw`

```text
{"raw": "夏末秋初红眼病高发 白领群体患病居高不下", "label": "health"}
{"raw": "损人真是件爆笑又过瘾滴事", "label": "joke"}
```

By the tools, we can tokenize the sentence, generate the radical sequences 
and finally covert this kind of data into the format which our model can handle.

Also, you can do those procedures like tokenize by yourself, but the generated data should followed the following format 

* `processed`

```text
{"w": ["$w1", "$w2"],"c":, "rw":, "rc":, "label": 0}
```

* `mature`

The data which can be finally inputted into our model is like:

```text
{"w": [$w1_idx, $w2_idx, $w3_idx, ...], "c":, "rw": , "rc": "", "label": 0}
```

