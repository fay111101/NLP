#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time    : 18-8-20 下午3:53
@Author  : fay
@Email   : fay625@sina.cn
@File    : crfmodel.py
@Software: PyCharm
"""

import sklearn_crfsuite
from config import get_config
from corpus import get_corpus
from sklearn.externals import joblib
from sklearn_crfsuite import metrics
from util import q_to_b

__model = None


class NER:

    def __init__(self):
        self.corpus = get_corpus()
        self.corpus.initialize()
        self.config = get_config()
        self.model = None

    def initialize_model(self):
        """
        初始化
        """
        algorithm = self.config.get('model', 'algorithm')
        c1 = float(self.config.get('model', 'c1'))
        c2 = float(self.config.get('model', 'c2'))
        max_iterations = int(self.config.get('model', 'max_iterations'))
        self.model = sklearn_crfsuite.CRF(algorithm=algorithm, c1=c1, c2=c2,
                                          max_iterations=max_iterations, all_possible_transitions=True)

    def train(self):
        """
        训练
        """
        self.initialize_model()
        # <class 'list'>: [{'w-1': '<BOS>', 'w:w+1': '迈向', 'bias': 1.0, 'w-1:w': '<BOS>迈', 'w': '迈', 'w+1': '向'},
        # {'w-1': '迈', 'w:w+1': '向充', 'bias': 1.0, 'w-1:w': '迈向', 'w': '向', 'w+1': '充'},
        # {'w-1': '向', 'w:w+1': '充满', 'bias': 1.0, 'w-1:w': '向充', 'w': '充', 'w+1': '满'},
        # {'w-1': '充', 'w:w+1': '满希', 'bias': 1.0, 'w-1:w': '充满', 'w': '满', 'w+1': '希'},
        #  {'w-1': '1', 'w:w+1': '张)', 'bias': 1.0, 'w-1:w': '1张', 'w': '张', 'w+1': ')'},
        #  {'w-1': '张', 'w:w+1': ')<EOS>', 'bias': 1.0, 'w-1:w': '张)', 'w': ')', 'w+1': '<EOS>'}]
        x, y = self.corpus.generator()

        x_train, y_train = x[500:], y[500:]
        x_test, y_test = x[:500], y[:500]
        self.model.fit(x_train, y_train)
        labels = list(self.model.classes_)

        labels.remove('O')
        y_predict = self.model.predict(x_test)
        metrics.flat_f1_score(y_test, y_predict, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        metrics.flat_classification_report(y_test, y_predict, labels=sorted_labels, digits=3)
        self.save_model()

    def predict(self, sentence):
        """
        预测
        # TODO 加入标签到序列的转换？
        """
        self.load_model()
        u_sent = q_to_b(sentence)
        word_lists = [[u'<BOS>'] + [c for c in u_sent] + [u'<EOS>']]
        # <class 'list'>: ['<BOS>', '新', '华', '社', '北', '京', '十', '二', '月', '三', '十', '一', '日', '电', '(', '中',
        # '央', '人', '民', '广', '播', '电', '台', '记', '者', '刘', '振', '英', '、', '新', '华', '社', '记', '者', '张',
        # '宿', '堂', ')', '今', '天', '是', '一', '九', '九', '七', '年', '的', '最', '后', '一', '天', '。', '辞', '旧',
        # '迎', '新', '之', '际', ',', '国', '务', '院', '总', '理', '李', '鹏', '今', '天', '上', '午', '来', '到', '北',
        # '京', '石', '景', '山', '发', '电', '总', '厂', '考', '察', ',', '向', '广', '大', '企', '业', '职', '工', '表',
        # '示', '节', '日', '的', '祝', '贺', ',', '向', '将', '要', '在', '节', '日', '期', '间', '坚', '守', '工', '作',
        # '岗', '位', '的', '同', '志', '们', '表', '示', '慰', '问', '<EOS>']
        word_grams = [self.corpus.segment_by_window(word_list) for word_list in word_lists]
        features = self.corpus.extract_feature(word_grams)
        y_predict = self.model.predict(features)
        entity = u''
        for index in range(len(y_predict[0])):
            if y_predict[0][index] != u'O':
                if index > 0 and y_predict[0][index][-1] != y_predict[0][index - 1][-1]:
                    entity += u' '
                entity += u_sent[index]
            elif entity[-1] != u' ':
                entity += u' '
        return entity

    def load_model(self, name='model'):
        """
        加载模型
        """
        model_path = self.config.get('model', 'model_path').format(name)
        self.model = joblib.load(model_path)

    def save_model(self, name='model'):
        """
        保存模型
        """
        model_path = self.config.get('model', 'model_path').format(name)
        joblib.dump(self.model, model_path)


def get_model():
    """
    单例模型获取
    """
    global __model
    if not __model:
        __model = NER()
    return __model


if __name__ == '__main__':
    model = get_model()
    sentence = '新华社北京十二月三十一日电(中央人民广播电台记者刘振英、新华社记者张宿堂)今天是一九九七年的最后一天。' \
               '辞旧迎新之际,国务院总理李鹏今天上午来到北京石景山发电总厂考察,向广大企业职工表示节日的祝贺,' \
               '向将要在节日期间坚守工作岗位的同志们表示慰问'
    model.train()
    # result = model.predict(sentence)
    # print(result)
