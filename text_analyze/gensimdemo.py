#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-5-29 下午4:15
@Author  : fay
@Email   : fay625@sina.cn
@File    : gensimdemo.py
@Software: PyCharm
"""
from gensim import corpora
import gensim
import multiprocessing
from gensim.models import word2vec
import jieba
import re

raw_corpus = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]
stoplist = set('for a of the and to in'.split(' '))

# 去停用词之后的文本
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in raw_corpus]
print(texts)
from collections import defaultdict

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
# 移除了一些英文虚词以及在预料中仅仅出现一次的词
precessed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
dictionary = corpora.Dictionary(precessed_corpus)
print(precessed_corpus)
print(dictionary.token2id)

new_doc = "human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

bow_corpus = [dictionary.doc2bow(text) for text in precessed_corpus]
print(bow_corpus)
# 以上用于对语料进行向量化表示
# 以下用模型表示文档
from gensim import models

tfidf = models.TfidfModel(bow_corpus)
string = "system minors"
string_bow = dictionary.doc2bow(string.lower().split())
string_tfidf = tfidf[string_bow]
print(string_bow)
print(string_tfidf)


def segment_text(source_corpus, train_corpus, coding, punctuation):
    '''
    切词,去除标点符号
    :param source_corpus: 原始语料
    :param train_corpus: 切词语料
    :param coding: 文件编码
    :param punctuation: 去除的标点符号
    :return:
    '''
    with open(source_corpus, 'r', encoding=coding) as f, open(train_corpus, 'w', encoding=coding) as w:
        for line in f:
            # 去除标点符号
            line = re.sub('[{0}]+'.format(punctuation), '', line.strip())
            # 切词
            words = jieba.cut(line)
            w.write(' '.join(words))


if __name__ == '__main__':
    # 严格限制标点符号
    strict_punctuation = '。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼'
    # 简单限制标点符号
    simple_punctuation = '’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    # 去除标点符号
    punctuation = simple_punctuation + strict_punctuation

    # 文件编码
    coding = 'utf-8'
    # 原始语料
    source_corpus_text = 'source.txt'

    # 是每个词的向量维度
    size = 400
    # 是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词
    window = 5
    # 设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃
    min_count = 1
    # 是训练的进程数，默认是当前运行机器的处理器核数。
    workers = multiprocessing.cpu_count()
    # 切词语料
    train_corpus_text = 'words.txt'
    # w2v模型文件
    model_text = 'w2v_size_{0}.model'.format(size)

    # 切词 @TODO 切词后注释
    # segment_text(source_corpus_text, train_corpus_text, coding, punctuation)

    # w2v训练模型 @TODO 训练后注释
    sentences = word2vec.Text8Corpus(train_corpus_text)
    model = word2vec.Word2Vec(sentences=sentences, size=size, window=window, min_count=min_count, workers=workers)
    model.save(model_text)

    # 加载模型
    model = gensim.models.Word2Vec.load(model_text)
    # print(model['运动会'])

    # 计算一个词的最近似的词，倒序
    # similar_words = model.most_similar('球队')
    # for word in similar_words:
    #     print(word[0], word[1])

    # 计算两词之间的余弦相似度
    # sim1 = model.similarity('运动会', '总成绩')
    # sim2 = model.similarity('排名', '运动会')
    # sim3 = model.similarity('展示', '学院')
    # sim4 = model.similarity('学院', '体育')
    # print(sim1)
    # print(sim2)
    # print(sim3)
    # print(sim4)

    # 计算两个集合之间的余弦似度
    list1 = ['运动会', '总成绩']
    list2 = ['排名', '运动会']
    list3 = ['学院', '体育']
    list_sim1 = model.n_similarity(list1, list2)
    print(list_sim1)
    list_sim2 = model.n_similarity(list1, list3)
    print(list_sim2)

    # 选出集合中不同类的词语
    list = ['队员', '足球比赛', '小组', '代表队']
    print(model.doesnt_match(list))
    list = ['队员', '足球比赛', '小组', '西瓜']
    print(model.doesnt_match(list))
