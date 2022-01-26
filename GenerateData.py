from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mu_s_1 = [[3, 6],
          [5, 4],
          [6, 6]]
cov_s_1 = [[[1.5, 0], [0, 1.5]],
           [[2, 0], [0, 2]],
           [[1, 0], [0, 1]]]

mu_s_2 = [[3, 6],
          [5, 4],
          [6, 6]]
cov_s_2 = [[[1.5, 0.1], [0.1, 1.5]],
           [[1, -0.20], [-0.20, 2]],
           [[2, -0.25], [-0.25, 1.5]]]

SIZE = 500


def generate_dataset(mu_s, cov_s, label_sampels_size):
    dataset = pd.DataFrame(data={'X1': [], 'X2': [], 'Y': []})
    for i, (mu, cov) in enumerate(zip(mu_s, cov_s)):
        x1, x2 = np.random.multivariate_normal(mu, cov, label_sampels_size).T
        temp = pd.DataFrame(data={'X1': x1, 'X2': x2, 'Y': [i] * label_sampels_size})
        dataset = pd.concat([dataset, temp], axis=0)
    return dataset


dataset1 = generate_dataset(mu_s_1, cov_s_1, SIZE)
dataset2 = generate_dataset(mu_s_2, cov_s_2, SIZE)

dataset1.to_csv('dataset1.csv', index=False)
dataset2.to_csv('dataset2.csv', index=False)
