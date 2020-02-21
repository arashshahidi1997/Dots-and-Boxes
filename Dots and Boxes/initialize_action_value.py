import numpy as np

w = 3
h = 3
dim = h * (w - 1) + (h - 1) * w
index_list = np.load('symmetries/index_list' + str(h) + '_' + str(w) + '.npy')
reduced_index_list = list(set(index_list[:, 0]))
folder = 'action_value_function/'


def vectorize(num):
    vector = np.zeros(dim, dtype='int')
    v = np.array([int(s) for s in bin(num)[2:]])
    vector[dim - np.size(v):] = v
    return vector


Q = np.ones((len(reduced_index_list), dim))

k = 0
for s in reduced_index_list:
    vector = vectorize(s)
    Q[k, np.where(vector == 1)] = -100
    k += 1

np.save(folder + 'initial_Q' + str(h) + '-' + str(w) + '.npy', Q)