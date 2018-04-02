import numpy as np
from functools import reduce
from .metrics import r2_score

class Perceptron(object):

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None
        self.intercept_ = None
        self.activator = None

    def fit(self, X_train, y_train, iteration, rate):
        """根据训练数据集X_train, y_train训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        for i in range(iteration):
            samples = zip(X_train,y_train)
            for (X_train,y_train) in samples:
                output = self.predict(X_train)
                delta = y_train - output
                self.coef_ = map(lambda x_w:x_w[1]+rate * delta * x_w[0], zip(X_train, self.coef_))
                self.intercept_ += rate * delta

        return self


    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        return self.activator(reduce(lambda a, b: a + b, map(lambda x_w: x_w[0] * x_w[1], zip(X_predict, self.weights)), 0.0) + self.intercept_)

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "Perceptron()"
