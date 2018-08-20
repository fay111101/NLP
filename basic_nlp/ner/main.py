#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-8-20 下午4:04
@Author  : fay
@Email   : fay625@sina.cn
@File    : main.py
@Software: PyCharm
"""
import sys

from model.api import *


def main():
    # arg=sys.argv[1]
    # arg = 'process'
    arg = 'train'
    if arg == 'train':
        train()
    elif arg == 'process':
        pre_process()
    else:
        print('Args must in ["process", "train"].')
    sys.exit()


if __name__ == '__main__':
    main()
