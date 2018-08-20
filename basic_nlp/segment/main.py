#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-8-20 上午11:48
@Author  : fay
@Email   : fay625@sina.cn
@File    : main.py
@Software: PyCharm
"""


import sys
from model.api import *
def main():
    # arg = sys.argv[1]
    # arg='test'
    arg='cut'
    if arg == 'train':
        train()
    elif arg == 'test':
        test()
    elif arg=='cut':
        cut("扬帆起航")
    else:
        print('Args must in ["train", "test"].')
    sys.exit()

if __name__ == '__main__':
    main()