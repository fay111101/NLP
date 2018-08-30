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

from corpus import get_corpus
from hmmmodel import get_model
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

if __name__ == '__main__':
    sentence = '新华社北京十二月三十一日电(中央人民广播电台记者刘振英、新华社记者张宿堂)今天是一九九七年的最后一天。' \
               '辞旧迎新之际,国务院总理李鹏今天上午来到北京石景山发电总厂考察,向广大企业职工表示节日的祝贺,' \
               '向将要在节日期间坚守工作岗位的同志们表示慰问'
    cut(sentence)