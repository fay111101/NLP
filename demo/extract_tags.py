#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-17 下午3:40
@Author  : fay
@Email   : fay625@sina.cn
@File    : extract_tags.py
@Software: PyCharm
"""


# 基于TF-IDF算法的关键字抽取
"""
jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
sentence 为待提取的文本
topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
withWeight 为是否一并返回关键词权重值，默认值为 False
allowPOS 仅包括指定词性的词，默认值为空，即不筛选
jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件
"""


from os import path
import jieba.analyse as analyse

d = path.dirname(__file__)

text_path = 'data/test.txt' #设置要分析的文本路径
text = open(path.join(d, text_path)).read()
# for key in analyse.extract_tags(text,50, withWeight=False):
for key in analyse.extract_tags(text,10, withWeight=True):
# 使用jieba.analyse.extract_tags()参数提取关键字,默认参数为50
# 当withWeight=True时,将会返回number类型的一个权重值(TF-IDF)
    print (key)


# import jieba
# TestStr = "2010年底部队友谊篮球赛结束"
# # 因为在汉语中没有空格进行词语的分隔，所以经常会出现中文歧义，比如年底-底部-部队-队友
# # jieba 默认启用了HMM（隐马尔科夫模型）进行中文分词，实际效果不错
#
# seg_list = jieba.cut(TestStr, cut_all=True)
# print ("Full Mode:", "/ ".join(seg_list)) # 全模式
#
# seg_list = jieba.cut(TestStr, cut_all=False)
# print ("Default Mode:", "/ ".join(seg_list)) # 默认模式
# # 在默认模式下有对中文歧义有较好的分类方式
#
