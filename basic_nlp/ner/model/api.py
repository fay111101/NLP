#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-8-20 下午4:03
@Author  : fay
@Email   : fay625@sina.cn
@File    : api.py
@Software: PyCharm
"""
from corpus import get_corpus
from crfmodel import get_model

__all__ = ["pre_process", "train", "recognize"]


def pre_process():
    """
    抽取语料特征
    """
    corpus = get_corpus()
    corpus.pre_process()


def train():
    """
    训练模型
    """
    model = get_model()
    model.train()


def recognize(sentence):
    """
    命名实体识别
    """
    model = get_model()
    return model.predict(sentence)

if __name__ == '__main__':
    sentence =  u'新华社北京十二月三十一日电(中央人民广播电台记者刘振英、新华社记者张宿堂)今天是一九九七年的最后一天。' \
           u'辞旧迎新之际,国务院总理李鹏今天上午来到北京石景山发电总厂考察,向广大企业职工表示节日的祝贺,' \
           u'向将要在节日期间坚守工作岗位的同志们表示慰问'
    result = recognize(sentence)
    print(result)
