#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-23 下午1:31
@Author  : fay
@Email   : fay625@sina.cn
@File    : extract_tags_idfpath.py
@Software: PyCharm
"""
# 基于TF-IDF算法的关键字抽取,自定义语料
import sys
sys.path.append('../')

import jieba
import jieba.analyse
from optparse import OptionParser

USAGE = "usage:    python extract_tags_idfpath.py [file name] -k [top k]"

parser = OptionParser(USAGE)
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args()


if len(args) < 1:
    print(USAGE)
    sys.exit(1)

file_name = args[0]

if opt.topK is None:
    topK = 10
else:
    topK = int(opt.topK)

content = open(file_name, 'rb').read()

jieba.analyse.set_idf_path("../data/idf.txt.big");

tags = jieba.analyse.extract_tags(content, topK=topK)

print(",".join(tags))
