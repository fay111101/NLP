#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-8-20 下午4:03
@Author  : fay
@Email   : fay625@sina.cn
@File    : api.py
@Software: PyCharm
"""
from .corpus import get_corpus
from .crfmodel import get_model

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