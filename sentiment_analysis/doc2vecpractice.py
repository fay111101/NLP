#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-7-19
@Author  : fay
@Email   : fay625@sina.cn
@File    : imdb.py
@Software: PyCharm
"""
'''
https://github.com/keras-team/keras/blob/master/examples/imdb.py
https://blog.csdn.net/walker_hao/article/details/78995591
'''
import logging
import os
import sys

import gensim
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
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
logger.info('starting !!')
pos_corpus_dir = './corpus/aclImdb/train/pos/'
neg_corpus_dir = './corpus/aclImdb/train/neg/'
unsup_corpus_dir = './corpus/aclImdb/train/unsup/'


def get_dataset():
    pos_corpus_files = os.listdir(pos_corpus_dir)
    pos_reviews = []
    for file in pos_corpus_files:
        with open(pos_corpus_dir + file, 'r') as f:
            pos_reviews.extend(f.readlines())

    neg_corpus_files = os.listdir(neg_corpus_dir)
    neg_reviews = []
    for file in neg_corpus_files:
        with open(neg_corpus_dir + file, 'r') as f:
            neg_reviews.extend(f.readlines())

    unsup_corpus_files = os.listdir(unsup_corpus_dir)
    unsup_reviews = []
    for file in unsup_corpus_files:
        with open(unsup_corpus_dir + file, 'r') as f:
            unsup_reviews.extend(f.readlines())
    # 给语料定义标签
    # use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)

    x_train = clean_text(x_train)
    x_test = clean_text(x_test)
    unsup_reviews = clean_text(unsup_reviews)

    x_train = labelize_reviews(x_train, 'TRAIN')
    x_test = labelize_reviews(x_test, 'TEST')
    unsup_reviews = labelize_reviews(unsup_reviews, 'UNSUP')

    return x_train, x_test, y_train, y_test, unsup_reviews


def clean_text(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', '') for z in corpus]
    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


# Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
# We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
# a dummy index of the review.
# Gensim的Doc2Vec应用于训练要求每一篇文章/句子有一个唯一标识的label.
# 我们使用Gensim自带的LabeledSentence方法. 标识的格式为"TRAIN_i"和"TEST_i"，其中i为序号+
# LabeledSentence=gensim.models.doc2vec.LabeledSentence
# Bug AttributeError: 'numpy.ndarray' object has no attribute 'words'

TaggedDocument = gensim.models.doc2vec.TaggedDocument


def labelize_reviews(reviews, label_type):
    labelized = []

    for i, doc in enumerate(reviews):
        # print(doc)
        label = '%s_%s' % (label_type, i)
        labelized.append(TaggedDocument(doc, [label]))
    # print(labelized)
    return labelized


def train(x_train, x_test, unsup_reviews, size=300, epoch_num=10):
    # instantiate our DM and DBOW models
    model_dm = gensim.models.Doc2Vec(min_count=1, window=5, size=size, sample=1e-3, negative=5, workers=5)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    # build vocab over all reviews
    # AttributeError: 'list' object has no attribute 'words'
    # model_dm.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
    # AttributeError: 'list' object has no attribute 'words'
    # model_dbow.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)).tolist())
    all_reviews = x_train + x_test + unsup_reviews
    model_dm.build_vocab(all_reviews)
    model_dbow.build_vocab(all_reviews)
    # We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
    # 进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    # AttributeError: 'numpy.ndarray' object has no attribute 'words'
    # all_train_reviews = np.concatenate((x_train, unsup_reviews))
    all_train_reviews = x_train + unsup_reviews
    logger.info('Begin trainning!')
    for epoch in range(epoch_num):
        logger.info('epoch %d' % epoch)
        # perm = np.random.permutation(all_train_reviews.shape[0])
        # perm = np.random.permutation(len(all_train_reviews))
        import random
        random.shuffle(all_train_reviews)
        model_dm.train(all_train_reviews, total_examples=model_dm.corpus_count,
                       epochs=model_dm.iter)
        model_dbow.train(all_train_reviews, total_examples=model_dbow.corpus_count,
                         epochs=model_dbow.iter)
    # train over test set
    # AttributeError: 'numpy.ndarray' object has no attribute 'words'
    # x_test = np.array(x_test)
    for epoch in range(epoch_num):
        logger.info('epoch %d' % epoch)
        # perm = np.random.permutation(x_test.shape[0])
        random.shuffle(x_test)
        model_dm.train(x_test, total_examples=model_dm.corpus_count, epochs=model_dm.iter)
        model_dbow.train(x_test, total_examples=model_dbow.corpus_count, epochs=model_dbow.iter)

    logger.info('model saved')
    model_dm.save('./model/imdbdm.d2v')
    model_dbow.save('./model/imdbdbow.d2v')

    return model_dm, model_dbow


# Get training set vectors from our models
def getVecs(model, corpus, size):
    # print(model.docvecs)
    # print(corpus)
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)


def get_vectors(model_dm, model_dbow, size):
    '''
    将训练完成的数据转换为vectors
    :param model_dm:
    :param model_dbow:
    :param size:
    :return:
    '''
    # 获取训练数据集的文档向量
    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    # 获取测试数据集的文档向量
    # Construct vectors for test reviews
    test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))
    return train_vecs, test_vecs


def classifier(train_vecs, y_train, test_vecs, y_test):
    logger.info('beginning class')
    from sklearn.linear_model import SGDClassifier
    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)
    # 返回预测属于某标签的概率
    print(lr.predict_proba(test_vecs).shape)
    pred_probas = lr.predict_proba(test_vecs)[:,1]
    print('Test Accuracy: %.2f' % lr.score(test_vecs, y_test))
    return pred_probas


def ROC_curve(pred_probas, y_test):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    # _thresholds
    fpr, tpr, _ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()


if __name__ == '__main__':
    # 设置向量维度和训练次数
    size, epoch_num = 400, 10
    # 获取训练与测试数据及其类别标注
    x_train, x_test, y_train, y_test, unsup_reviews = get_dataset()
    # 对数据进行训练，获得模型
    model_dm, model_dbow = train(x_train, x_test, unsup_reviews, size, epoch_num)
    # load and test the model
    logger.info('model loaded')
    model_dm = Doc2Vec.load('./model/imdbdm.d2v')
    model_dbow = Doc2Vec.load('./model/imdbdbow.d2v')
    # 从模型中抽取文档相应的向量
    train_vecs, test_vecs = get_vectors(model_dm, model_dbow, size)
    # 使用文章所转换的向量进行情感正负分类训练
    pred_probas = classifier(train_vecs, y_train, test_vecs, y_test)
    # 画出ROC曲线
    ROC_curve(pred_probas, y_test)

# build vocab over all reviews
# np.concatenate返回ndarray 正常是没有namedtuple的
# [
#     [
#     ['picking', 'up', 'the', 'jacket', 'of', 'this', 'dvd', 'in', 'the', 'video', 'store', 'i', 'was', 'intrigued',
#
#       'bob', "thornton's", 'wig', 'all', 'about', '?', '?', '?', 'i', 'could', 'go', 'on', 'for', 'another', 'ten',
#       'lines', ',', 'but', 'this', 'film', 'just', "isn't", 'worth', 'the', 'bother', '.', 'anyone', 'who', 'hates',
#       'wasting', 'money', 'should', 'stay', 'well', 'away', 'from', 'this', 'stinker', '.'], ['UNSUP_49500']
#       ],
#       [['.', '.', '.', 'which', 'makes', 'me', 'wonder', 'about', 'myself', '!', 'this', 'film', 'is', 'horrible', '.', '.', ], ['UNSUP_49999']]
#  ]

# all_reviews=np.concatenate((x_train, x_test, unsup_reviews))
# all_reviews=np.concatenate((x_train, x_test, unsup_reviews)).tolist()

# [
#
#        TaggedDocument(words=['in', 'the', 'past', 'few', 'years', 'i', 'have', 'rarely', 'done', 'reviews', 'of', 'films', 'or',
#        'tv', 'on', 'here', 'for', 'one', 'simple', 'reason', ',', 'abuse', ',', 'abuse', 'from', 'people', 'who', 'think', 'you',
#        'are', 'wrong', 'or', 'are', 'thinking', 'you', 'have', 'been', 'unfair', 'on', 'something', 'they', 'love/hate', 'and',
#        'i', 'grew', 'sick', 'of', 'it', '.', 'but', 'and', 'it', 'is', 'a', 'big', 'but', 'i', 'decided', 'for', 'something', 'this',
#        , ',', 'for', 'those', 'that', 'have', 'and', 'arnt', 'obsessed', 'with',
#      as', "madeleine's", 'own', 'ambivalent', 'handling', 'of',
#            'comedy', 'and', 'music', 'with', 'equal', 'ease', 'and', 'of', 'course', 'is', 'one', 'of', 'the', 'best', 'lookers', 'ever', '
#            in', 'movies', 'as', 'icing', 'on', 'the', 'cake', '.'], tags=['UNSUP_49538']),
#        TaggedDocument(words=["i've", 'already', 'watched', 'this', 'episode', 'twice', ',', 'and', 'love', 'it', '.', "it's",
#        'great', 'to', 'see', 'the', 'creators', 'still', 'know', 'how', 'to', 'have', 'some', 'fun', 'with', 'these', 'characters', '.', '.
#        ', '.', 'especially', 'the', 'interplay', 'between', 'brass', 'and', 'doc', 'robbins', '.', 'my', 'favorite', ',', 'by', 'far', ',',
#     'together', '.', 'my', 'opinion', ':', 'important', 'content', 'but', 'way', 'too', 'much', 'rob', 'stewart', 'in', 'it', '.'], tags=['UNSUP_49999'])]

