#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-7-18 下午5:21
@Author  : fay
@Email   : fay625@sina.cn
@File    : word2vector_basic.py
@Software: PyCharm
"""
'''
data http://mattmahoney.net/dc/
'''
import tensorflow as tf
import os

url = 'http://mattmahoney.net/dc/'
def maybe_download(filename,expected_bytes):
    '''

    :param filename:
    :param expected_bytes:
    :return:
    '''
    pass
batch_size = 32
train_input = tf.placeholder(tf.int32, shape=[batch_size])
train_label = tf.placeholder(tf.int32, shape=[batch_size, 1])
