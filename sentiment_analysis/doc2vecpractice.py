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
https://blog.csdn.net/lenbow/article/details/52120230
'''
import logging
import os
import random
import sys

import gensim
import numpy as np
from gensim import utils
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
pos_corpus_dir = './corpus/aclImdb/train/pos/'
neg_corpus_dir = './corpus/aclImdb/train/neg/'
unsup_corpus_dir = './corpus/aclImdb/unsup/'


## the code for the doc2vec
class TaggedLineSentence(object):
    """
    sources: [file1 name: tag1 name, file2 name: tag2 name ...]
    privade two functions:
        to_array: transfer each line to a object of TaggedDocument and then add to a list
        perm: permutations
    """

    def __init__(self, sources):
        self.sources = sources

    def to_array(self):
        self.sentences = []

        for source, prefix in self.sources.items():
            print(source)
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    # TaggedDocument([word1, word2 ...], [tagx])
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(),
                                                         [prefix + '_%s' % item_no]))
        # print(self.sentences)
        '''

          [
          TaggedDocument(words=['i', 'must', 'admit', 'i', 'do', 'not', 'hold', 'much', 'of', 'new', 'age', 'mumbo', 'jumbo', '.', 
          'when', 'people', '"exchange', 'energy"', 'i', 'always', 'wonder', 'how', 'much', 'kj', 'is', 'actually', 'exchanged', 'and', 
          'how', 'it', 'may', 'contribute', 'to', 'solving', 'the', 'global', 'warming', 'problem', '.', 'when', 'energy', '"is',
                   '.', 'if', 'you', 'want', 'to', 'have', 'a', 'good', 'time', 'and', 'have', 'to', 'choose', 'between', 'this',
          'movie', 'and', 'sticking',            'safety', 'pins', 'in', 'your', 'eyelids', ',', 'take', 'my', 'advise', ':', 
          'choose', 'the', 'latter', '.'], tags=['TRAIN_19520']), 

          TaggedDocument(words=['philo', 'vance', '(', 'william', 'powell', ')', 'helps', 'solve', 'multiple', 'murders', 'among', 
          'the', 'wealthy', 'after',            'a', 'dog', 'show', '.', 'usually', 'i', 'hate', 'overly', 'convoluted', 'mysteries', 
          '(', 'like', 'this', ')',       'but', 'i', 'love', 'this', 'movie', '.', 'good', 'actor', 'with', 'a', 'very', 'distinctive',
           'voice', 'and', 'some', 'of', 'his', 'lines', 'were', 'hilarious', '.', 'basically', ',', 'an', 'excellent', '1930s', 'hollywood',
            'murder', 'mystery', '.', 'well', 'worth', 'seeing', '.'], tags=['TRAIN_19521'])]

        '''
        return self.sentences

    def perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)  # Note that this line does not return anything.
        return shuffled




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


x_train, x_test, y_train, y_test, unsup_reviews = get_dataset()
x_train = clean_text(x_train)
x_test = clean_text(x_test)
unsup_reviews = clean_text(unsup_reviews)

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


x_train = labelize_reviews(x_train, 'TRAIN')
x_test = labelize_reviews(x_train, 'TEST')
unsup_reviews = labelize_reviews(unsup_reviews, 'UNSUP')

import random

size = 400

#instantiate our DM and DBOW models
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

#build vocab over all reviews
# model_dm.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
model_dm.build_vocab(x_train+ x_test+unsup_reviews)
# model_dbow.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
model_dbow.build_vocab(x_train+ x_test+unsup_reviews)

#We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
all_train_reviews = np.concatenate((x_train, unsup_reviews))
for epoch in range(10):
    perm = np.random.permutation(all_train_reviews.shape[0])
    model_dm.train(all_train_reviews[perm])
    model_dbow.train(all_train_reviews[perm])

#Get training set vectors from our models
def getVecs(model, corpus, size):
    vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

train_vecs_dm = getVecs(model_dm, x_train, size)
train_vecs_dbow = getVecs(model_dbow, x_train, size)

train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

#train over test set
x_test = np.array(x_test)

for epoch in range(10):
    perm = np.random.permutation(x_test.shape[0])
    model_dm.train(x_test[perm])
    model_dbow.train(x_test[perm])

#Construct vectors for test reviews
test_vecs_dm = getVecs(model_dm, x_test, size)
test_vecs_dbow = getVecs(model_dbow, x_test, size)

test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))
'''
size = 400
# instantiate our DM and DBOW models

model_dm = Doc2Vec(min_count=1, window=10, vector_size=size, negative=5, workers=3, sample=1e-3)
model_dbow = Doc2Vec(min_count=1, window=10, vector_size=size, sample=1e-3, negative=5, dm=0, workers=3)
# build vocab over all reviews
# np.concatenate返回ndarray 正常是没有namedtuple的


[
    [
    ['picking', 'up', 'the', 'jacket', 'of', 'this', 'dvd', 'in', 'the', 'video', 'store', 'i', 'was', 'intrigued', 

      'bob', "thornton's", 'wig', 'all', 'about', '?', '?', '?', 'i', 'could', 'go', 'on', 'for', 'another', 'ten', 
      'lines', ',', 'but', 'this', 'film', 'just', "isn't", 'worth', 'the', 'bother', '.', 'anyone', 'who', 'hates', 
      'wasting', 'money', 'should', 'stay', 'well', 'away', 'from', 'this', 'stinker', '.'], ['UNSUP_49500']
      ], 
      [['.', '.', '.', 'which', 'makes', 'me', 'wonder', 'about', 'myself', '!', 'this', 'film', 'is', 'horrible', '.', '.', ], ['UNSUP_49999']]
 ]

# all_reviews=np.concatenate((x_train, x_test, unsup_reviews))
# all_reviews=np.concatenate((x_train, x_test, unsup_reviews)).tolist()

[

       TaggedDocument(words=['in', 'the', 'past', 'few', 'years', 'i', 'have', 'rarely', 'done', 'reviews', 'of', 'films', 'or', 
       'tv', 'on', 'here', 'for', 'one', 'simple', 'reason', ',', 'abuse', ',', 'abuse', 'from', 'people', 'who', 'think', 'you', 
       'are', 'wrong', 'or', 'are', 'thinking', 'you', 'have', 'been', 'unfair', 'on', 'something', 'they', 'love/hate', 'and', 
       'i', 'grew', 'sick', 'of', 'it', '.', 'but', 'and', 'it', 'is', 'a', 'big', 'but', 'i', 'decided', 'for', 'something', 'this', 
       , ',', 'for', 'those', 'that', 'have', 'and', 'arnt', 'obsessed', 'with', 
     as', "madeleine's", 'own', 'ambivalent', 'handling', 'of',
           'comedy', 'and', 'music', 'with', 'equal', 'ease', 'and', 'of', 'course', 'is', 'one', 'of', 'the', 'best', 'lookers', 'ever', '
           in', 'movies', 'as', 'icing', 'on', 'the', 'cake', '.'], tags=['UNSUP_49538']),
       TaggedDocument(words=["i've", 'already', 'watched', 'this', 'episode', 'twice', ',', 'and', 'love', 'it', '.', "it's", 
       'great', 'to', 'see', 'the', 'creators', 'still', 'know', 'how', 'to', 'have', 'some', 'fun', 'with', 'these', 'characters', '.', '.
       ', '.', 'especially', 'the', 'interplay', 'between', 'brass', 'and', 'doc', 'robbins', '.', 'my', 'favorite', ',', 'by', 'far', ',', 
    'together', '.', 'my', 'opinion', ':', 'important', 'content', 'but', 'way', 'too', 'much', 'rob', 'stewart', 'in', 'it', '.'], tags=['UNSUP_49999'])]

# TypeError: unhashable type: 'list'
# all_reviews = x_train  # + x_test+ unsup_reviews
print(type(x_train))
# 此处有坑，extend方法拼接是在x_train的基础上做的
# x_train.extend(x_test)
# x_train.extend(unsup_reviews)
all_reviews = x_train

model_dm.build_vocab(all_reviews)
# model_dm.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
# model_dbow.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
# We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
all_train_reviews = np.concatenate((x_train, unsup_reviews))
# for epoch in range(10):
#     perm = np.random.permutation(all_train_reviews.shape[0])
#     # print(all_train_reviews[perm])
#     model_dm.train(all_train_reviews[perm])
#     # model_dbow.train(all_train_reviews[perm])
'''
