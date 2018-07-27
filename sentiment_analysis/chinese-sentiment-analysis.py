# coding: utf-8

# ### 载入所需的库
# 我们依旧会用gensim去做word2vec的处理，会用sklearn当中的SVM进行建模

# In[2]:


# -*- coding: utf-8 -*-
"""
@author: fay
"""
import jieba
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# ### 载入数据，做预处理(分词)，切分训练集与测试集

# In[3]:


def load_file_and_preprocessing():
    neg = pd.read_excel('data/neg.xls', header=None, index=None)
    pos = pd.read_excel('data/pos.xls', header=None, index=None)

    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    # print pos['words']
    # use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))

    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)

    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    return x_train, x_test


# ### 对每个句子的所有词向量取均值，来生成一个句子的vector

# In[4]:


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


# ### 计算词向量

# In[5]:


# 计算词向量
def get_train_vecs(x_train, x_test):
    n_dim = 300
    # 初始化模型和词表
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    imdb_w2v.build_vocab(x_train)

    # 在评论训练集上建模(可能会花费几分钟)
    imdb_w2v.train(x_train, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)

    train_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_train])
    # train_vecs = scale(train_vecs)

    np.save('data/train_vecs.npy', train_vecs)
    print(train_vecs.shape)
    # 在测试集上训练
    imdb_w2v.train(x_test, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)
    imdb_w2v.save('data/w2v_model/w2v_model.pkl')
    # Build test tweet vectors then scale
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_test])
    # test_vecs = scale(test_vecs)
    np.save('data/test_vecs.npy', test_vecs)
    print(test_vecs.shape)


# In[6]:


def get_data():
    train_vecs = np.load('data/train_vecs.npy')
    y_train = np.load('data/y_train.npy')
    test_vecs = np.load('data/test_vecs.npy')
    y_test = np.load('data/y_test.npy')
    return train_vecs, y_train, test_vecs, y_test


# ### 训练svm模型

# In[8]:


def svm_train(train_vecs, y_train, test_vecs, y_test):
    clf = SVC(kernel='rbf', verbose=True)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf, 'data/svm_model/model.pkl')
    print(clf.score(test_vecs, y_test))


# ### 构建待预测句子的向量

# In[9]:


def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('data/w2v_model/w2v_model.pkl')
    # imdb_w2v.train(words)
    train_vecs = build_sentence_vector(words, n_dim, imdb_w2v)
    # print train_vecs.shape
    return train_vecs


# ### 对单个句子进行情感判断 

# In[10]:


def svm_predict(string):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words)
    clf = joblib.load('data/svm_model/model.pkl')

    result = clf.predict(words_vecs)

    if int(result[0]) == 1:
        print(string, ' positive')
    else:
        print(string, ' negative')


# In[13]:

if __name__=='__main__':
    ##对输入句子情感进行判断
    string = '电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    # string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    # x_train, x_test = load_file_and_preprocessing()
    # get_train_vecs(x_train, x_test)
    train_vecs, y_train, test_vecs, y_test = get_data()
    svm_train(train_vecs, y_train, test_vecs, y_test)
    svm_predict(string)
