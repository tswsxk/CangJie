# coding: utf-8
# create by tongshiwei on 2019/4/12

import numpy as np
import logging
import warnings
from longling import loading, path_append, print_time
import mxnet as mx
from tqdm import tqdm
from gluonnlp.embedding import TokenEmbedding
from gluonnlp.data import FixedBucketSampler, PadSequence

__all__ = ["extract", "transform", "etl", "pseudo_data_iter"]


def load_embedding(vec_root="./", logger=None,
                   word_embedding_file=None, word_radical_embedding_file=None,
                   char_embedding_file=None, char_radical_embedding_file=None
                   ):
    logger = logging.getLogger("RFAG") if logger is None else logger

    word_embedding_file = path_append(
        vec_root, "word.vec.dat", to_str=True
    ) if word_embedding_file is None else word_embedding_file
    word_radical_embedding_file = path_append(
        vec_root, "word_radical.vec.dat", to_str=True
    ) if word_radical_embedding_file is None else word_radical_embedding_file
    char_embedding_file = path_append(
        vec_root, "char.vec.dat", to_str=True
    ) if char_embedding_file is None else char_embedding_file
    char_radical_embedding_file = path_append(
        vec_root, "char_radical.vec.dat", to_str=True
    ) if char_radical_embedding_file is None else char_radical_embedding_file

    with print_time(logger=logger, tips='loading embedding'):
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(4)

        p1 = pool.apply_async(TokenEmbedding.from_file,
                              args=(word_embedding_file,))
        p2 = pool.apply_async(TokenEmbedding.from_file,
                              args=(word_radical_embedding_file,))
        p3 = pool.apply_async(TokenEmbedding.from_file,
                              args=(char_embedding_file,))
        p4 = pool.apply_async(TokenEmbedding.from_file,
                              args=(char_radical_embedding_file,))

        pool.close()
        pool.join()

        word_embedding = p1.get()
        word_radical_embedding = p2.get()
        char_embedding = p3.get()
        char_radical_embedding = p4.get()

        return word_embedding, word_radical_embedding, char_embedding, char_radical_embedding


def pseudo_data_iter(_cfg):
    def pseudo_data_generation():
        # 在这里定义测试用伪数据流
        import random
        random.seed(10)

        from CangJie.utils.testing import pseudo_sentence
        from CangJie import tokenize, characterize, token2radical

        sentences = pseudo_sentence(1000, 20)

        def feature2num(token):
            return random.randint(0, 10)

        w = [feature2num(tokenize(s)) for s in sentences]
        rw = [feature2num(token2radical(_w)) for _w in w]
        c = [feature2num(characterize(s)) for s in sentences]
        rc = [feature2num(token2radical(_c)) for _c in c]

        labels = [random.randint(0, 32) for _ in sentences]
        features = [w, rw, c, rc]

        return features, labels

    return load(transform(pseudo_data_generation(), _cfg), _cfg)


def extract(data_src, embedding_list):
    word_feature = []
    word_radical_feature = []
    char_feature = []
    char_radical_feature = []
    features = [word_feature, word_radical_feature, char_feature,
                char_radical_feature]
    labels = []
    word_embedding, word_radical_embedding, char_embedding, char_radical_embedding = embedding_list
    for ds in tqdm(loading(data_src), "loading data from %s" % data_src):
        w, rw, c, rc, label = ds['w'], ds['rw'], ds['c'], ds['rc'], ds['label']
        w = word_embedding.token_to_idx(w)
        rw = word_embedding.token_to_idx(rw)
        c = word_embedding.token_to_idx(c)
        rc = word_embedding.token_to_idx(rc)
        try:
            assert len(w) == len(rw), "some word miss radical"
            assert len(c) == len(rc), "some char miss radical"
        except AssertionError as e:
            warnings.warn("%s" % e)
            continue
        word_feature.append(w)
        word_radical_feature.append(rw)
        char_feature.append(c)
        char_radical_feature.append(rc)
        labels.append(label)

    return features, labels


def transform(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    batch_size = params.batch_size
    padding = params.padding
    num_buckets = params.num_buckets
    fixed_length = params.fixed_length

    features, labels = raw_data
    word_feature, word_radical_feature, char_feature, char_radical_feature = features
    batch_idxes = FixedBucketSampler(
        [len(word_f) for word_f in word_feature],
        batch_size, num_buckets=num_buckets
    )
    batch = []
    for batch_idx in batch_idxes:
        batch_features = [[] for _ in range(len(features))]
        batch_labels = []
        for idx in batch_idx:
            for i, feature in enumerate(batch_features):
                batch_features[i].append(features[i][idx])
            batch_labels.append(labels[idx])
        batch_data = []
        word_mask = []
        char_mask = []
        for i, feature in enumerate(batch_features):
            max_len = max(
                [len(fea) for fea in feature]
            ) if not fixed_length else fixed_length
            padder = PadSequence(max_len, pad_val=padding)
            feature, mask = zip(*[(padder(fea), len(fea)) for fea in feature])
            if i == 0:
                word_mask = mask
            elif i == 2:
                char_mask = mask
            batch_data.append(mx.nd.array(feature))
        batch_data.append(mx.nd.array(word_mask))
        batch_data.append(mx.nd.array(char_mask))
        batch_data.append(mx.nd.array(batch_labels, dtype=np.int))
        batch.append(batch_data)
    return batch[::-1]


def load(transformed_data, params):
    return transformed_data


def etl(filename, embedding_list, params):
    raw_data = extract(filename, embedding_list)
    transformed_data = transform(raw_data, params)
    return load(transformed_data, params)


if __name__ == '__main__':
    from longling.lib.structure import AttrDict
    from CangJie import PAD_TOKEN
    import os

    filename = "../../../../data/train.json"
    print(os.path.abspath(filename))

    embedding_list = load_embedding("../../../../data/vec/")

    for data in tqdm(extract(filename, embedding_list)):
        pass

    parameters = AttrDict({"batch_size": 128, "padding": PAD_TOKEN}, num_buckets=100, fixed_length=None)
    for data in tqdm(etl(filename, embedding_list, params=parameters)):
        pass
