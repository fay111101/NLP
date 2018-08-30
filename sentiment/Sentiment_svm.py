#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-7-19
@Author  : fay
@Email   : fay625@sina.cn
@File    : word2vecdemo.py
@Software: PyCharm
"""
import jieba
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC

corpus_path = './corpus/'
model_path = './model/'
data_path = './data/'


def load_file_and_preprocessing():
    neg = pd.read_excel('{}neg.xls'.format(data_path), header=None, index=None)
    pos = pd.read_excel('{}pos.xls'.format(data_path), header=None, index=None)

    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    # print (pos['words'])
    # use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)

    np.save('{}y_train.npy'.format(data_path), y_train)
    np.save('{}y_test.npy'.format(data_path), y_test)
    return x_train, x_test


def build_sentence_vector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def get_train_vecs(x_train, x_test):
    n_dim = 300
    # 初始化模型和词表
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    imdb_w2v.build_vocab(x_train)

    # 在评论训练集上建模(可能会花费几分钟)
    imdb_w2v.train(x_train, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)
    train_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_train])
    train_vecs = scale(train_vecs)
    np.save('{}train_vecs.npy'.format(data_path), train_vecs)
    print(train_vecs.shape)

    # 在测试集上训练
    imdb_w2v.train(x_test, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)
    # Build test tweet vectors then scale
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_test])
    test_vecs = scale(test_vecs)
    np.save('{}test_vecs.npy'.format(data_path), test_vecs)
    print(test_vecs.shape)


def get_data():
    train_vecs = np.load('{}train_vecs.npy'.format(data_path))
    y_train = np.load('{}y_train.npy'.format(data_path))
    test_vecs = np.load('{}test_vecs.npy'.format(data_path))
    y_test = np.load('{}y_test.npy'.format(data_path))
    return train_vecs, y_train, test_vecs, y_test


def svm_train(train_vecs, y_train, test_vecs, y_test):
    param = {'C': [0.1, 0.2, 0.3]}
    clf = GridSearchCV(SVC(kernel='rbf', verbose=True), param_grid=param, verbose=-1)
    clf.fit(train_vecs, y_train)
    print(clf.best_estimator_)

    svm = SVC(kernel='rbf', verbose=True, C=clf.best_params_['C'])
    svm.fit(train_vecs, y_train)
    pred = svm.predict(test_vecs)

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    # print(y_test, pred)
    fpr, tpr, _ = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='area=%.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')

    plt.show()
    """
    optimization finished, #iter = 6215
    obj = -9831.082663, rho = -1.971551
    nSV = 10722, nBSV = 10682
    Total nSV = 10722
    0.7863065624259654
    
    其中，#iter为迭代次数，nu是你选择的核函数类型的参数，obj为SVM文件转换为的二次规划求解得到的最小值，
    rho为判决函数的偏置项b，nSV为标准支持向量个数(0<a[i]<c)，nBSV为边界上的支持向量个数(a[i]=c)，
    Total nSV为支持向量总个数（对于两类来说，因为只有一个分类模型Total nSV = nBSV，但是对于多类，这个是各个分类模型的nSV之和）。
    """

    """
    使用了scale
     optimization
    finished,  # iter = 9284
    obj = -6133.684440, rho = -0.149794
    nSV = 7588, nBSV = 6483
    Total
    nSV = 7588
    [LibSVM]
    0.8441127694859039
    """
    joblib.dump(clf, 'data/svm_model/model.pkl')
    print(clf.score(test_vecs, y_test))


def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('data/w2v_model/w2v_model.pkl')
    # imdb_w2v.train(words)
    train_vecs = build_sentence_vector(words, n_dim, imdb_w2v)
    # print train_vecs.shape
    return train_vecs


def svm_predict(sentence):
    words = jieba.lcut(sentence)
    words_vecs = get_predict_vecs(words)
    clf = joblib.load('{}svm_model/model.pkl'.format(data_path))

    pred = clf.predict(words_vecs)

    if int(pred[0]) == 1:
        print(sentence, ' positive')
    else:
        print(sentence, ' negative')


if __name__ == '__main__':
    ##对输入句子情感进行判断
    sentence = '电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    # sentence='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    # x_train, x_test = load_file_and_preprocessing()
    # get_train_vecs(x_train, x_test)
    train_vecs, y_train, test_vecs, y_test = get_data()
    svm_train(train_vecs, y_train, test_vecs, y_test)
    # svm_predict(sentence)
