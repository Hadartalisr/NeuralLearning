import numpy as np
# from scipy.io import loadmat
#
# M = loadmat('data_10D.mat')
# print(M)
#
#
#

a = np.array([[1, 2], [3, 4]])

print(a @ a)


def he(x):
    return x**2

print(he(2))