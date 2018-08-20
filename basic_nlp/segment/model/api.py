#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-8-20 下午1:12
@Author  : fay
@Email   : fay625@sina.cn
@File    : api.py
@Software: PyCharm
"""

"""
接口
"""

from .calculatecorpus import get_corpus
from .hmmmodel import get_model
__all__=["train","cut","test"]
def train():
    """
    用于处理语料及hmm模型概率计算
    :return:
    """
    corpus=get_corpus()
    corpus.initialize()
    corpus.cal_state()

def test():
    """
    模型测试
    :return:
    """
    model=get_model()
    return model.test()

def cut(sentence):
    """
    分词
    :param sentence:
    :return:
    """
    model=get_model()
    return model.cut(sentence)