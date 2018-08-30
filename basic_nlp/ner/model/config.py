#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-8-20 下午3:49
@Author  : fay
@Email   : fay625@sina.cn
@File    : config.py
@Software: PyCharm
"""
from configparser import ConfigParser

__config = None


def get_config(config_file_path='./conf/config.conf'):
    """
    单例模式实现
    :param config_file_path:
    :return:
    """
    global __config
    if not __config:
        config = ConfigParser()
        config.read(config_file_path)
    else:
        config = __config
    return config


class single_instance(object):
    __instance = None

    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if cls.__instance == None:
            cls.__instance = object.__new__(cls, *args, **kwargs)
        return cls.__instance


class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instanc



# class A(object):
#     def __init__(self):
#         print('initA')
#
#     # def __new__(cls, *args, **kwargs):
#     #     print('new')
#     #     return object.__new__(cls, *args, **kwargs)
#
#     def get_object(self):
#         print('ooooA')
#
#
# class B(A):
#     def __init__(self):
#         print('initB')
#
#     def get_object(self):
#         print('ooooB')
#
#
# class C(A):
#     def __init__(self):
#         print('initC')
#
#     def get_object(self):
#         print('ooooC')
#
#
# class D(B,C):
#     def __init__(self):
#         print('initD')
#
#     def get_object(self):
#         print('ooooD')
#
# if __name__ == '__main__':
#     # obj1=single_instance()
#     # obj2=single_instance()
#     # print(obj1)
#     # print(obj2)
#     a = A()
#     a1 = A()
#     print(a)
#     print(a1)
#
#     b = B()
#     b1 = B()
#     print(b)
#     print(b1)
#
#     c = C()
#     c.get_object()
#
#     d=D()
#     d.get_object()
