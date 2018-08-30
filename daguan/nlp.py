#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-7-22
@Author  : fay
@Email   : fay625@sina.cn
@File    : nlp.py
@Software: PyCharm
"""
'''
此数据集用于训练模型，每一行对应一篇文章。文章分别在“字”和“词”的级别上做了脱敏处理。共有四列：
第一列是文章的索引(id)，第二列是文章正文在“字”级别上的表示，即字符相隔正文(article)；
第三列是在“词”级别上的表示，即词语相隔正文(word_seg)；第四列是这篇文章的标注(class)。
注：每一个数字对应一个“字”，或“词”，或“标点符号”。“字”的编号与“词”的编号是独立的！

'''
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
data_path='./data/'
result='./result/'
column = "word_seg"
train = pd.read_csv('{}train_set.csv'.format(data_path))
# print(train)
test = pd.read_csv('{}test_set.csv'.format(data_path))
print(test)
test_id = test["id"].copy()
#
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])
fid0=open('%sbaseline.csv'%result,'w')

y=(train["class"]-1).astype(int)
lin_clf = svm.LinearSVC()
lin_clf.fit(trn_term_doc,y)
preds = lin_clf.predict(test_term_doc)
i=0
fid0.write("id,class"+"\n")
for item in preds:
    fid0.write(str(i)+","+str(item+1)+"\n")
    i=i+1
fid0.close()