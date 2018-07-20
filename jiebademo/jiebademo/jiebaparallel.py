#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-4-17 下午4:41
@Author  : fay
@Email   : fay625@sina.cn
@File    : jiebaparallel.py
@Software: PyCharm
"""
import sys
import time
import os
print(os.name)
sys.path.append("../../")
import jieba

jieba.enable_parallel() # 关闭并行分词
jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数

# url = sys.argv[1]
url='../corpus/test.txt'
content = open(url,"rb").read()
t1 = time.time()
words = "/ ".join(jieba.cut(content))

t2 = time.time()
tm_cost = t2-t1

log_f = open("1.log","wb")
log_f.write(words.encode('utf-8'))

print('speed %s bytes/second' % (len(content)/tm_cost))