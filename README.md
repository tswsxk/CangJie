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
{"w": ["夏末秋", "初", "红眼病", "高发", "白领", "群体", "患病", "居高不下"], "c": [], "rw":, "rc":, "label": 0}
```

* `mature`

The data which can be finally inputted into our model is like:

```text
{"w": [$w1_idx, $w2_idx, $w3_idx, ...], "c":, "rw": , "rc": "", "label": 0}
```

### 实验

#### 获取实验数据

* 获取词向量 (gensim model)
```shell
# get word embedding vector
wget https://data.bdaa.pro/datasets/NLP/vec/word.vec.dat.gz
# get word radical embedding vector
wget https://data.bdaa.pro/datasets/NLP/vec/word_radical.vec.dat.gz
# get char embedding vector
wget https://data.bdaa.pro/datasets/NLP/vec/char.vec.dat.gz
# get char radical embedding vector
wget https://data.bdaa.pro/datasets/NLP/vec/char_radical.vec.data.gz
```

获得词向量压缩文件后，使用如下命令进行解压
```shell
gzip -d $dat.gz
```

或者可以使用
```shell
longling download $url
```
来直接获取解压缩后的文件

The dim of all mentioned above embedding vectors are `256`

And the vocabulary size is
```text
word_embedding_size = 352183
word_radical_embedding_size = 31759
char_embedding_size = 4746
char_radical_embedding_size = 242
```

* 数据集2
```text
word_radical_embedding_size = 28771
char_radical_embedding_size = 235
```

* 分类任务
```shell
wget https://data.bdaa.pro/datasets/NLP/ctc/ctc32/train.json
wget https://data.bdaa.pro/datasets/NLP/ctc/ctc32/test.json
```

#### 预处理

词向量文件均为`空格`分隔的`csv`格式，第一列为`token`, 其它列为 `vec value`
