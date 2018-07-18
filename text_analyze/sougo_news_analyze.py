#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-5-29 下午12:57
@Author  : fay
@Email   : fay625@sina.cn
@File    : sougo_news_analyze.py
@Software: PyCharm
"""
import jieba
import pandas as pd

"""Data Source
http://www.sogou.com/labs/resource/ca.php"""


df_news = pd.read_table('./data/val.txt', names=['category', 'theme', 'URL', 'content'], encoding='utf-8')
df_news = df_news.dropna()
df_news.head()
# print(df_news.shape)
content = df_news.content.values.tolist()
# print(content[1000])

content_S = []
for line in content:
    current_segment_list = jieba.cut(line)
    current_segment=' '.join(current_segment_list)
    if len(current_segment) > 1 and current_segment != '\r\n':
        content_S.append(current_segment)

print(content_S[1000])
df_content = pd.DataFrame({'content_S': content_S})
stopwords = pd.read_csv("./data/stopwords.txt", index_col=False, sep='\t', quoting=3, names=['stopword'],
                        encoding='utf-8')
stopwords.head(20)

global contents_clean
global all_words

contents_clean = []
all_words = []
def drop_stopwords(contents, stopwords):
    """
    对内容去停用词
    :param contents:
    :param stopwords:
    :return:
    """
    global contents_clean
    global all_words
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(word)
        contents_clean.append(line_clean)
    return contents_clean, all_words


df_content = pd.DataFrame({'contents_clean': contents_clean})
df_content.head()
df_all_words = pd.DataFrame({'all_words': all_words})
df_all_words.head()
#
# import numpy as np
#
# words_count = df_all_words.groupby(by=['all_words']).agg({'count': np.size})
# words_count = words_count.reset_index().sort_values(by=['count'], ascending=False)
# words_count.head()
#
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import matplotlib
#
# matplotlib.rcParams['figure.figsize'] = {10.0, 5.0}
# wordcloud = WordCloud(font_path='./data/simhei.ttf', background_color='white', max_font_size=80)
# word_frequence = {x[0]: x[1] for x in words_count.head(100).values}
# wordcloud = wordcloud.fit_words(word_frequence)
# plt.imshow(wordcloud)

"""
关键字抽取tfidf

"""
from jieba import analyse

index = 1000
# print(df_news['content'][index])
tfidf = analyse.extract_tags
content_S_str = "".join(content_S[index])
keywords = tfidf(content_S_str, topK=5, withWeight=False)
# for keyword in keywords:
#     print(keyword)
"""
关键字抽取textrank

"""
textrank = analyse.textrank
keywords = textrank(content_S_str, topK=5, withWeight=False)
for keyword in keywords:
    # print(keyword)
    pass

from gensim import corpora
import gensim

contents_clean, all_words = drop_stopwords(content, stopwords)
dictinary = corpora.Dictionary(contents_clean)
print(dictinary.items())
# 二元组对（0,1）代表一篇文档中 id为0的单词出现了1次
corpus = [dictinary.doc2bow(sentence) for sentence in contents_clean]
for i,sentence in enumerate(contents_clean):
    # print(sentence)
    # print(dictinary.doc2bow(sentence))
    if i==1:
        break

# print(corpus)
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictinary, num_topics=20)
# print(lda.print_topic(1, topn=5))
for topic in lda.print_topics(num_topics=20, num_words=5):
    # print(topic[1])
    pass

df_train = pd.DataFrame({'contents_clean': contents_clean, 'label': df_news['category']})
df_train.tail()
df_train.label.unique()
label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育": 5, "教育": 6, "文化": 7, "军事": 8, "娱乐": 9, "时尚": 0}
df_train['label'] = df_train['label'].map(label_mapping)
df_train.head()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values,
                                                    random_state=1)
words_train = []
for line_index in range(len(x_train)):
    try:
        words_train.append(' '.join(x_train[line_index]))
    except:
        print(line_index)

# print(len(words_train))
from sklearn.feature_extraction.text import CountVectorizer

texts = ["dog cat fish", "dog cat cat", "fish bird", 'bird']
cv = CountVectorizer()
cv_fit = cv.fit_transform(texts)
# print(cv.get_feature_names())
# print(cv_fit.toarray())
# print(cv_fit.toarray().sum(axis=0))




def classifier_countvectorizer():
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)
    vec.fit(words_train)
    print(words_train)
    from sklearn.naive_bayes import MultinomialNB
    classifier = MultinomialNB()
    classifier.fit(vec.transform(words_train), y_train)
    test_words = []
    for line_index in range(len(x_test)):
        try:
            # x_train[line_index][word_index] = str(x_train[line_index][word_index])
            test_words.append(' '.join(x_test[line_index]))
        except:
            print(line_index)
    print(test_words[0])
    classifier.score(vec.transform(test_words), y_test)


def classifier_tfidfvectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer='word', max_features=4000, lowercase=False)
    vectorizer.fit(words_train)
    from sklearn.naive_bayes import MultinomialNB
    classifier_tfidf = MultinomialNB()
    classifier_tfidf.fit(vectorizer.transform(words_train), y_train)
    test_words = []
    for line_word in range(len(x_test)):
        try:
            # x_train[line_index][word_index] = str(x_train[line_index][word_index])
            test_words.append(' '.join(x_test[line_index]))
        except:
            print(line_index)
    print(test_words[0])
    classifier_tfidf.score(vectorizer.transform(test_words), y_test)
