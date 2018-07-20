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
data:http://www.cs.cornell.edu/people/pabo/movie-review-data/
'''
import logging
import os
import random
import sys

import numpy as np
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.linear_model import LogisticRegression

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
pos_corpus_dir = "./corpus/aclImdb/{}/pos/"
neg_corpus_dir = './corpus/aclImdb/{}/neg/'
unsup_corpus_dir = './corpus/aclImdb/{}/'
pos='./corpus/aclImdb/alldata/{}-pos.txt'
neg='./corpus/aclImdb/alldata/{}-neg.txt'
unsup='./corpus/aclImdb/alldata/unsup.txt'
corpus_path='./corpus/aclImdb/alldata/'

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
        # self.create_dataset(pos,neg,unsup,'train')
        # self.create_dataset(pos,neg,unsup,'test')
        # self.create_dataset(pos,neg,unsup,'unsup')

    def create_dataset(self, pos, neg, unsup,name):
        '''
        str()出来的值是给人看的。。。repr()出来的值是给python看的，可以通过eval()重新变回一个Python对象
        :param pos:
        :param neg:
        :param unsup:
        :param name:
        :return:
        '''
        unsup_file=eval(repr(unsup_corpus_dir).format(name))
        if name == 'unsup':
            unsup_corpus_files = os.listdir(unsup_file)
            unsup_reviews = open(eval(repr(unsup).format(name)), 'w')
            for file in unsup_corpus_files:
                with open(unsup_file + file, 'r') as f:
                    unsup_reviews.write(''.join(f.readlines())+'\n')
            return
        pos_file=eval(repr(pos_corpus_dir).format(name))
        pos_corpus_files = os.listdir(pos_file)
        pos_reviews = open(eval(repr(pos).format(name)), 'w')
        for file in pos_corpus_files:
            with open(pos_file + file, 'r') as f:
                pos_reviews.write(''.join(f.readlines())+'\n')
        neg_file=eval(repr(neg_corpus_dir).format(name))
        neg_corpus_files = os.listdir(neg_file)
        neg_reviews = open(eval(repr(neg).format(name)), 'w')
        for file in neg_corpus_files:
            with open(neg_file + file, 'r') as f:
                neg_reviews.write(''.join(f.readlines())+'\n')



    def to_array(self):
        self.sentences = []

        for source, prefix in self.sources.items():
            print(source)
            with utils.smart_open(corpus_path+source) as fin:
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


sources = {'test-neg.txt': 'TEST_NEG', 'test-pos.txt': 'TEST_POS',
           'train-neg.txt': 'TRAIN_NEG', 'train-pos.txt': 'TRAIN_POS',
           'unsup.txt': 'TRAIN_UNS'}
sentences = TaggedLineSentence(sources)

# set the parameter and get a model.
# by default dm=1, PV-DM is used. Otherwise, PV-DBOW is employed.
model = Doc2Vec(min_count=1, window=10, size=100,
                sample=1e-4, negative=5, dm=1, workers=7)
# print(type(sentences))
model.build_vocab(sentences.to_array())

# train the model
for epoch in range(20):
    logger.info('epoch %d' % epoch)
    model.train(sentences.perm(),
                total_examples=model.corpus_count,
                epochs=model.iter
                )

logger.info('model saved')
model.save('./model/imdb.d2v')

# load and test the model
logger.info('model loaded')
model = Doc2Vec.load('./model/imdb.d2v')

logger.info('Sentiment Analysis...')

logger.info('transfer the train document to the vector')
train_arrays = np.zeros((25000, 100))
train_labels = np.zeros(25000)

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    # note that the attribute is model.docvecs
    train_arrays[i], train_arrays[12500 + i] = \
        model.docvecs[prefix_train_pos], model.docvecs[prefix_train_neg]
    train_labels[i], train_labels[12500 + i] = 1, 0

logger.info('transfer the test document to the vector')
test_arrays = np.zeros((25000, 100))
test_labels = np.zeros(25000)

for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i], test_arrays[12500 + i] = \
        model.docvecs[prefix_test_pos], model.docvecs[prefix_test_neg]
    test_labels[i], test_labels[12500 + i] = 1, 0

logger.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

print("accuracy: " + str(classifier.score(test_arrays, test_labels)))
