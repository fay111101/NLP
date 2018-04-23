#!usr/bin/env python  
# -*- coding:utf-8 _*-  
"""
@Time    : 18-4-23 下午4:52
@Author  : fay
@Email   : fay625@sina.cn
@File    : newsclassifier.py
@Software: PyCharm
"""
import pickle


# 保存至文件
def savefile(savepath, content):
    with open(savepath, "wb") as fp:
        fp.write(content)


# 读取文件
def readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


def writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)


# 读取bunch对象
def readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch
