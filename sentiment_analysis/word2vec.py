#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-7-19
@Author  : fay
@Email   : fay625@sina.cn
@File    : word2vec.py
@Software: PyCharm
"""
'''
2015
https://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis
'''
import gensim.models
corpus_path='./corpus/'
# 词向量是基于谷歌新闻数据（大约一千亿个单词）训练所得。需要注意的是，这个文件解压后的大小是 3.5 GB
model=gensim.models.KeyedVectors.load_word2vec_format('%sGoogleNews-vectors-negative300.bin'%corpus_path,binary=True)
result=model.most_similar(positive=['woman','king'],negative=['man'],topn=5)
print(result)
# 可以发现语法关系,比如识别出最高级或单词形态的单词 “biggest”-“big”+“small”=“smallest”
model.most_similar(positive=['biggest','small'],negative=['big'],topn=5)
