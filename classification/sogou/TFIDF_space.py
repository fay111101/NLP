#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Time    : 18-4-23 下午4:52
@Author  : fay
@Email   : fay625@sina.cn
@File    : newsclassifier.py
@Software: PyCharm
"""

from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from Tools import readfile, readbunchobj, writebunchobj


def vector_space(stopword_path, bunch_path, space_path, train_tfidf_path=None):
    '''
    构造词向量
    :param stopword_path:
    :param bunch_path:
    :param space_path:
    :param train_tfidf_path:
    :return:
    '''
    stpwrdlst = readfile(stopword_path).splitlines()
    bunch = readbunchobj(bunch_path)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})

    if train_tfidf_path is not None:
        trainbunch = readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
                                     vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

    else:
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        print(tfidfspace.tdm)
        print(vectorizer.vocabulary_)
        tfidfspace.vocabulary = vectorizer.vocabulary_

    writebunchobj(space_path, tfidfspace)
    print("if-idf词向量空间实例创建成功！！！")


if __name__ == '__main__':
    path='../corpus/'
    stopword_path =path+ "train_word_bag/hlt_stop_words.txt"
    bunch_path = path+"train_word_bag/train_set.dat"
    space_path = path+"train_word_bag/tfdifspace.dat"
    vector_space(stopword_path, bunch_path, space_path)

    bunch_path = path+"test_word_bag/test_set.dat"
    space_path = path+"test_word_bag/testspace.dat"
    train_tfidf_path = path+"train_word_bag/tfdifspace.dat"
    vector_space(stopword_path, bunch_path, space_path, train_tfidf_path)
