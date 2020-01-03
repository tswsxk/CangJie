# format

## Token2Vec

May also be known as word2vec, character2vec.

We use the following format to stand a token-vec pair

```json
["$token", [1.0, 1.0, 1.0]]
```

In the following, we use `vec_json` format to stand the mentioned above format

For there are several other formats, we provide several commands to convert them into `json` format
and also convert them to other formats.


### vec_csv

```text
$token 1.0 1.0 1.0
```

This kind of format can be directly used by [gluonnlp TokenEmbedding](https://gluon-nlp.mxnet.io/api/modules/embedding.html?highlight=tokenembedding#gluonnlp.embedding.TokenEmbedding.from_file)
 