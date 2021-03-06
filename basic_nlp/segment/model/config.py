#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-8-20 上午11:20
@Author  : fay
@Email   : fay625@sina.cn
@File    : config.py
@Software: PyCharm
"""
from configparser import ConfigParser

__config=None

def get_config(config_file_path='./conf/config.conf'):
    """

    :param config_file_path:
    :return:
    """
    global __config
    if not __config:
        config=ConfigParser()
        config.read(config_file_path)
    else:
        config=__config

    return config