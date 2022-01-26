import collections
import numpy as np

from services import Data


class Bayesian:

    def __init__(self, data: Data):
        self.data = data
        self.class_name_list = np.unique(data.trainLabel)
        self.class_name_list.sort()
        self.means = None
        self.priors = None
        self.covariance_list = None

    def calculate_prior(self):
        classes = np.unique(self.data.trainLabel)
        classes.sort()
        prior = np.zeros(classes.size)
        for index, className in enumerate(classes):
            prior[index] = self.data.trainLabel[self.data.trainLabel == className].size \
                           / self.data.trainLabel.size
        self.priors = prior

    def calculate_mean(self):
        classes = np.unique(self.data.trainLabel)
        classes.sort()
        means = np.zeros((classes.size, self.data.trainData.shape[1]))
        for index, className in enumerate(classes):
            x = self.data.trainData[self.data.trainLabel == className]
            means[index] = np.mean(x, axis=0)
        self.means = means

    def calculate_covariance(self):
        classNameList = np.unique(self.data.trainLabel)
        classNameList.sort()
        covariance_list = [None] * len(classNameList)
        cov = np.zeros(shape=(self.data.trainData.shape[1], self.data.trainData.shape[1]))
        for index, className in enumerate(classNameList):
            classRows = self.data.trainData[self.data.trainLabel == className]
            dist = (classRows - self.means[index]).T.dot(classRows - self.means[index])
            cov += dist
            cov /= self.data.trainData.shape[0]
            covariance_list[index] = cov

        self.covariance_list = covariance_list

    def probability(self, data, i):
        xm = data - self.means[i]
        xm_covariance = (xm.dot(np.linalg.pinv(self.covariance_list[i]))) * xm
        xm_covariance_sum = xm_covariance.sum(axis=1)
        return -0.5 * xm_covariance_sum + np.log(self.priors[i])

    def predict(self, data):
        predict_list = np.zeros((data.shape[0], self.priors.size))
        for index, class_name in enumerate(self.class_name_list):
            predict_list[:, index] = self.probability(data, index)

        return np.argmax(predict_list, axis=1)

    def fit(self):
        self.calculate_prior()
        self.calculate_mean()
        self.calculate_covariance()


class Quadratic:
    def __init__(self, data):
        self.data = data
        self.class_name_list = np.unique(data.trainLabel)
        self.class_name_list.sort()
        self.means = None
        self.priors = None
        self.covariance_matrix = None

    def calculate_prior(self):
        prior = np.zeros(self.class_name_list.size)
        for index, className in enumerate(self.class_name_list):
            prior[index] = self.data.trainLabel[self.data.trainLabel == className].size \
                           / self.data.trainLabel.size
        self.priors = prior

    def calculate_mean(self):
        means = np.zeros((self.class_name_list.size, self.data.trainData.shape[1]))
        covariance_matrix = []
        for index, className in enumerate(self.class_name_list):
            row_data = self.data.trainData[self.data.trainLabel == className]
            mean = np.asmatrix(np.mean(row_data, axis=0))
            means[index] = mean
            cov_matrix = (row_data - mean).T @ (row_data - mean) / self.data.trainData.shape[0]
            covariance_matrix.append(cov_matrix)

        self.means = means
        self.covariance_matrix = covariance_matrix

    def predict(self, data):
        probs = np.asmatrix(np.zeros((data.shape[0], self.priors.size)))
        for index, class_abel in enumerate(self.class_name_list):
            probs[:, index] = self.probability(data, index)
        return np.argmax(probs, axis=1)

    def probability(self, data, index):
        X = np.asmatrix(data)
        cov_matrix_det = np.linalg.det(self.covariance_matrix[index])
        cov_matrix_inv = np.linalg.pinv(self.covariance_matrix[index])
        Xm = X - self.means[index]
        Xm_covariance = (Xm @ cov_matrix_inv) @ Xm.T
        Xm_covariance_sum = Xm_covariance.sum(axis=1)
        return -0.5 * Xm_covariance_sum - 0.5 * np.log(cov_matrix_det) + np.log(self.priors[index])

    def fix(self):
        self.calculate_prior()
        self.calculate_mean()


class NaiveBayes:

    def __init__(self, data):
        self.data = data
        self.class_label_list = np.unique(self.data.trainLabel)
        self.class_label_list.sort()
        self.priors = np.zeros(len(self.class_label_list))
        self.pos_word_list = {0: dict(), 1: dict()}
        self.word_count_collection = {}
        self.word_counter()

    def word_counter(self):
        for line, label in zip(self.data.trainData, self.data.trainLabel):
            word_counts = collections.Counter(line)
            for word, count in sorted(word_counts.items()):
                if word not in self.word_count_collection.keys():
                    self.word_count_collection[word] = count
                else:
                    self.word_count_collection[word] += count
                if word not in self.pos_word_list[label].keys():
                    self.pos_word_list[label][word] = count
                else:
                    self.pos_word_list[label][word] += count

    def prior(self):
        for index, label in enumerate(self.class_label_list):
            self.priors[index] = len(self.data.trainLabel[self.data.trainLabel == label]) / len(self.data.trainLabel)

    def p_x_y(self, data, label):
        result = []
        word_total_in_label = 0
        for w in self.pos_word_list[label].keys():
            word_total_in_label += self.pos_word_list[label][w]
        for j, line in enumerate(data):
            result.append(self._p_x_Y(line, label, word_total_in_label))
        return np.array(result)

    def _p_x_Y(self, line, label, word_total_in_label):
        prob = 0
        for w in line:
            if w in self.pos_word_list[label].keys():
                p = (self.pos_word_list[label][w] / word_total_in_label) + 0.1
            else:
                p = 0.1
            prob += np.log(p)
        return prob

    def fit(self):
        self.prior()
        result = np.zeros((self.data.trainData.shape[0], len(self.class_label_list)))
        for index, label in enumerate(self.class_label_list):
            pxy_py = self.p_x_y(self.data.trainData, label) * self.priors[index]
            result[:, index] = pxy_py
        return np.argmax(result, axis=1)

    def predict_prior(self, y_data, classes):
        priors = np.zeros(len(classes))
        for i, c in enumerate(classes):
            priors[i] = len(y_data[y_data == c]) / len(y_data)
        return priors

    def predict(self, data):
        result = np.zeros((data.shape[0], len(self.class_label_list)))
        for index, label in enumerate(self.class_label_list):
            pxy_py = self.p_x_y(data, label) * self.priors[index]
            result[:, index] = pxy_py
        return np.argmax(result, axis=1)


