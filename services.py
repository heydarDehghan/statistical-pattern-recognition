import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
import nltk
from nltk.corpus import stopwords

from sympy import *
import re


class Info:
    def __init__(self, label, predict, data, cost_value):
        self.label = label
        self.predict = predict
        self.data = data
        self.cost_value = cost_value


class Data:
    trainData: np.array
    testData: np.array
    trainLabel: np.array
    testLabel: np.array

    def __init__(self, data, split_rate=0.2, bias=True, normal=False):
        self.trainData: np.array
        self.testData: np.array
        self.trainLabel: np.array
        self.testLabel: np.array
        self.split_rare = split_rate
        self.data = data
        self.bias = bias
        self.normal = normal
        self.prepare_data()

    def prepare_data(self):
        if self.normal:
            self.normalizer()

        # self.data[:, -1] = np.unique(self.data[:, -1], return_inverse=True)[1]
        if self.bias:
            self.data = np.insert(self.data, 0, 1, axis=1)

        self.trainData, self.testData, self.trainLabel, self.testLabel = train_test_split(self.data[:, :-1],
                                                                                          self.data[:, -1],
                                                                                          test_size=self.split_rare,
                                                                                          random_state=42)

    def normalizer(self):
        norm = np.linalg.norm(self.data[:, :-1])
        self.data[:, :-1] = self.data[:, :-1] / norm


def calculate_metrics(predicted, gold):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for p, g in zip(predicted, gold):

        if p == 1 and g == 1:
            true_pos += 1
        if p == 0 and g == 0:
            true_neg += 1
        if p == 1 and g == 0:
            false_pos += 1
        if p == 0 and g == 1:
            false_neg += 1

    recall = true_pos / float(true_pos + false_neg)

    precision = true_pos / float(true_pos + false_pos)

    fscore = 2 * precision * recall / (precision + recall)


    # accuracy = (true_pos + true_neg) / float(len(gold)) if gold else 0

    return  precision, recall, fscore


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def mse(actual, predict):
    diff = np.subtract(predict, actual)
    ms = np.power(diff, 2)
    return np.mean(ms)


def init_weights(size):
    x = np.random.uniform(-0.01, 0.01, size=size)
    return x


def load_data(path, array=True):
    train = pd.read_csv(path)
    if array:
        train = train.to_numpy()
    return train


def preprocess(line):
    line = cleanpunc(line)
    line = cleanstop(line)
    line = [stemming(w) for w in line]
    return line


def cleanstop(line):
    stop = set(stopwords.words('english'))
    filtered_sentence = []
    for w in line.split():
        if w.isalpha() and len(w) > 2:
            w = w.lower()
            if (w not in stop):
                filtered_sentence.append(w)
    return filtered_sentence


def stemming(word):
    snowball = nltk.stem.SnowballStemmer('english')
    return (snowball.stem(word.lower())).encode('utf8')


def cleanpunc(line):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', line)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    return cleaned


def load_word_data(path):
    with open(path, "r") as text_file:
        lines = text_file.read().split('\n')
    lines = [line.split("\t") for line in lines if len(line.split("\t")) == 2 and line.split("\t")[1] != '']
    train_sentences = [preprocess(line[0]) for line in lines]
    train_labels = [int(line[1]) for line in lines]
    return pd.DataFrame({'line': train_sentences, 'label': train_labels})


def plot_pdfs(data, mus, sigmas, labels, priors, y_train):
    ax = plt.axes(projection="3d")

    colors = [('b', 'viridis'), ('r', 'plasma')]

    fig2 = plt.figure()
    ax2 = fig2.gca()

    for cls in range(len(labels)):
        ax.scatter3D(data[y_train == cls][:, 0], data[y_train == cls][:, 1], np.ones(1) * -0.03, c=colors[cls][0])
        mu = np.asarray(mus[cls]).flatten()
        x = np.linspace(mu[0] - 3 * sigmas[cls][0, 0], mu[0] + 3 * sigmas[cls][0, 0], 40).flatten()
        y = np.linspace(mu[1] - 3 * sigmas[cls][1, 1], mu[1] + 3 * sigmas[cls][1, 1], 40).flatten()

        X, Y = np.meshgrid(x, y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        # print(pos.shape)
        rv = multivariate_normal([mu[0], mu[1]], sigmas[cls])
        Z = rv.pdf(pos)
        ax.plot_surface(X, Y, Z, cmap=colors[cls][1], linewidth=0.2, alpha=0.9, shade=True)

        ax2.contour(X, Y, Z, cmap='coolwarm')
    x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 40).flatten().T
    y = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 40).flatten().T
    b0 = 0.5 * mus[0].T.dot(np.linalg.pinv(sigmas[0])).dot(mus[0])
    b1 = -0.5 * mus[1].T.dot(np.linalg.pinv(sigmas[1])).dot(mus[1])
    b = b0 + b1 + np.log(priors[0] / priors[1])
    a = np.linalg.pinv(sigmas[0]).dot(mus[1] - mus[0])
    x1 = -(b + a[0] * x) / a[1]
    ax2.plot(x, x1)


def plot_dec_boundary(train_data, test_x, prediction, test_y, mus, sigmas, priors, title):
    missed_0 = np.take(test_x, np.setdiff1d(np.where(prediction == 0), np.where(test_y == 0)), axis=0)
    missed_1 = np.take(test_x, np.setdiff1d(np.where(prediction == 1), np.where(test_y == 1)), axis=0)
    cl0 = np.delete(test_x, np.where(prediction != 0), axis=0)
    cl1 = np.delete(test_x, np.where(prediction != 1), axis=0)
    plt.plot(cl0[:, 0], cl0[:, 1], '.c')
    plt.plot(missed_0[:, 0], missed_0[:, 1], '.r')
    plt.plot(cl1[:, 0], cl1[:, 1], '.y')
    plt.plot(missed_1[:, 0], missed_1[:, 1], '.k')
    x = np.linspace(np.min(train_data[:, 0]), np.max(train_data[:, 0]), 40).flatten().T
    y = np.linspace(np.min(train_data[:, 1]), np.max(train_data[:, 1]), 40).flatten().T
    b0 = 0.5 * mus[0].T.dot(np.linalg.pinv(sigmas[0])).dot(mus[0])
    b1 = -0.5 * mus[1].T.dot(np.linalg.pinv(sigmas[1])).dot(mus[1])
    b = b0 + b1 + np.log(priors[0] / priors[1])
    a = np.linalg.pinv(sigmas[0]).dot(mus[1] - mus[0])
    x1 = -(b + a[0] * x) / a[1]
    plt.plot(x, x1)
    plt.title(title)
    plt.show()
    return


def plot_pdfs1(data, db, predicted, mus, sigmas, labels, priors, y_train):
    ax = plt.axes(projection="3d")

    colors = [('b', 'viridis'), ('r', 'plasma'), ('g', 'inferno')]

    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.gca()

    for cls in range(len(labels)):
        mu = np.asarray(mus[cls]).flatten()
        x = np.linspace(mu[0] - 3 * sigmas[cls][0, 0], mu[0] + 3 * sigmas[cls][0, 0], 40).flatten()
        y = np.linspace(mu[1] - 3 * sigmas[cls][1, 1], mu[1] + 3 * sigmas[cls][1, 1], 40).flatten()
        X, Y = np.meshgrid(x, y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        rv = multivariate_normal([mu[0], mu[1]], sigmas[cls])
        Z = rv.pdf(pos)
        ax.plot_surface(X, Y, Z, cmap=colors[cls][1], linewidth=0.2, alpha=0.9, shade=True)
        ax2.contour(X, Y, Z, cmap='coolwarm')
    #     db = decision_boundaries(data, predicted, mus, sigmas, priors)
    for i in range(len(labels)):
        x1 = db[i][0]
        x2 = db[i][1]
        ax2.plot(x1, x2)


def plot_dec_boundary1(data_x, db, prediction, data_y, means, cov_matrices, priors, title):
    missed_0 = np.take(data_x, np.setdiff1d(np.where(prediction == 0), np.where(data_y == 0)), axis=0)
    missed_1 = np.take(data_x, np.setdiff1d(np.where(prediction == 1), np.where(data_y == 1)), axis=0)
    missed_2 = np.take(data_x, np.setdiff1d(np.where(prediction == 2), np.where(data_y == 2)), axis=0)
    cl0 = np.delete(data_x, np.where(prediction != 0), axis=0)
    cl1 = np.delete(data_x, np.where(prediction != 1), axis=0)
    cl2 = np.delete(data_x, np.where(prediction != 2), axis=0)
    plt.plot(cl0[:, 0], cl0[:, 1], '.c')
    plt.plot(missed_0[:, 0], missed_0[:, 1], '.r')
    plt.plot(cl1[:, 0], cl1[:, 1], '1y')
    plt.plot(missed_1[:, 0], missed_1[:, 1], '1k')
    plt.plot(cl2[:, 0], cl2[:, 1], '^g')
    plt.plot(missed_2[:, 0], missed_2[:, 1], '^m')
    #     db = decision_boundaries(data_x, prediction, means, cov_matrices, priors)
    for i in range(3):
        x1 = db[i][0]
        x2 = db[i][1]
        plt.plot(x1, x2)
    plt.title(title)
    plt.show()
    return


def decision_boundaries1(data_x, data_y, means, cov_matrices, priors):
    items = [(0, 1), (0, 2), (1, 2)]
    decision_boundaries = []
    for item in items:
        i, j = item
        X = data_x[(data_y == i) | (data_y == j)]
        x = np.arange(X.min(axis=0)[0], X.max(axis=0)[0], 0.2)
        det_i = np.linalg.det(cov_matrices[i])
        det_j = np.linalg.det(cov_matrices[j])
        inv_i = np.linalg.pinv(cov_matrices[i])
        inv_j = np.linalg.pinv(cov_matrices[j])
        mu_i = means[i]
        mu_j = means[j]
        a = -0.5 * (inv_i - inv_j)
        b = (inv_i @ mu_i.T) - (inv_j @ mu_j.T)
        c = np.log(priors[i] / priors[j]) - 1 / 2 * np.log(det_i / det_j)
        c = c + (-0.5 * (mu_i @ inv_i) @ mu_i.T) + (0.5 * (mu_j @ inv_j) @ mu_j.T)
        a1 = a[0, 0]
        a2 = a[0, 1]
        a3 = a[1, 0]
        a4 = a[1, 1]
        b1 = b[0, 0]
        b2 = b[1, 0]
        c = c[0, 0]
        result = []
        for x2 in x:
            x1 = symbols('x')
            equation = x1 ** 2 * a1 + x1 * x2 * a3 + x1 * x2 * a2 + x2 ** 2 * a4 + b1 * x1 + b2 * x2 + c
            r = solve(Eq(equation, 0), x1, domain=S.Reals)
            if (i == 0 and j == 1):
                r = r[0]
            else:
                r = r[1]
            if (type(r) is Float):
                result.append(r)
        result = list(zip(x, result))
        result = [(x1, x2) for x1, x2 in result if type(x2) is Float]
        x = []
        r = []
        for x1, x2 in result:
            x.append(x1)
            r.append(x2)
        decision_boundaries.append((x, r))
    #         decision_boundaries.append((x, result))
    return decision_boundaries

