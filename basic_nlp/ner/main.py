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
    # main()
    from model.api import recognize

    sentence = u'新华社北京十二月三十一日电(中央人民广播电台记者刘振英、新华社记者张宿堂)今天是一九九七年的最后一天。' \
               u'辞旧迎新之际,国务院总理李鹏今天上午来到北京石景山发电总厂考察,向广大企业职工表示节日的祝贺,' \
               u'向将要在节日期间坚守工作岗位的同志们表示慰问'
    predict = recognize(sentence)
