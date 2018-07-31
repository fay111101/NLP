#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-7-19
@Author  : fay
@Email   : fay625@sina.cn
@File    : word2vecdemo.py
@Software: PyCharm
"""
'''


'''
import gensim
import logging
import sys
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split

# code from the tutorial of the python model logging.
# create a logger, the same name corresponding to the same logger.
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# create console handler and set level to info
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
# create formatter and add formatter to ch
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)  # add ch to the logger

corpus_path = './corpus/aclImdb/alldata/'
train_pos_path = corpus_path + 'train-pos.txt'
train_neg_path = corpus_path + 'train-neg.txt'
test_pos_path = corpus_path + 'test-pos.txt'
test_neg_path = corpus_path + 'test-neg.txt'
model_path = './model/'


# # 词向量是基于谷歌新闻数据（大约一千亿个单词）训练所得。需要注意的是，这个文件解压后的大小是 3.5 GB
# model = gensim.models.KeyedVectors.load_word2vec_format('%sGoogleNews-vectors-negative300.bin' % corpus_path,
#                                                         binary=True)
# result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)
# print(result)
# # 可以发现语法关系,比如识别出最高级或单词形态的单词 “biggest”-“big”+“small”=“smallest”
# model.most_similar(positive=['biggest', 'small'], negative=['big'], topn=5)

def clean_text(corpus):
    corpus = [z.lower().replace('\n', '') for z in corpus]
    return corpus


def get_data():
    with open(train_pos_path, 'r') as file:
        pos = file.readlines()
    with open(train_neg_path, 'r') as file:
        neg = file.readlines()
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos, neg)), y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def train_vector(n_dims):
    x_train, x_test, y_train, y_test = get_data()
    imdb_w2v = Word2Vec(size=n_dims, workers=7, window=10, min_count=10)
    imdb_w2v.build_vocab(x_train)
    imdb_w2v.train(sentences=x_train, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)
    imdb_w2v.save(model_path + 'imdb_w2vtrain.model')
    imdb_w2v1 = Word2Vec(size=n_dims, workers=7, window=10, min_count=10)
    imdb_w2v1.build_vocab(x_test)
    imdb_w2v1.train(x_test, total_examples=imdb_w2v1.corpus_count, epochs=imdb_w2v1.iter)
    imdb_w2v1.save(model_path + 'imdb_w2vtest.model')


def buildWordVector(text, size, model):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def classifier(size):
    x_train, x_test, y_train, y_test = get_data()
    from sklearn.preprocessing import scale
    imdb_w2v_train = Word2Vec.load(model_path + 'imdb_w2vtrain.model')
    train_vecs = np.concatenate([buildWordVector(z, size=size, model=imdb_w2v_train) for z in x_train])
    train_vecs = scale(train_vecs)
    imdb_w2v_test = Word2Vec.load(model_path + 'imdb_w2vtest.model')
    test_vecs = np.concatenate([buildWordVector(z, size=size, model=imdb_w2v_train) for z in x_test])
    # test_vecs = np.concatenate([buildWordVector(z, size=size, model=imdb_w2v_test) for z in x_test])
    test_vecs = scale(test_vecs)
    from sklearn.linear_model import SGDClassifier
    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    pred = lr.predict(test_vecs)
    fpr, tpr ,_= roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='area=%.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')

    plt.show()


if __name__ == '__main__':
    # imdb_w2v = train_vector(100)
    classifier(100)
