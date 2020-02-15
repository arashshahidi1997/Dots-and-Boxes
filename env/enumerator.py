import numpy as np
a = 5
b = 5
enumerator = np.ndarray((a, b), dtype='int')
x = 1
for i in range(a):
    y = 1
    for j in range(b):
        enumerator[i, j] = x * y
        y *= 2

    x *= y

np.save('enumerator_matrix.npy', enumerator)


class Distribution:

    def __init__(distr, state_vector, terminal_state_vector):
        distr.h_state_num = state_vector[0]
        distr.h_array_len = terminal_state_vector[0] - state_vector[0]
        distr.h_array = np.ones(distr.h_array_len)
        distr.v_state_num = state_vector[1]
        distr.v_array_len = terminal_state_vector[1] - state_vector[1]
        distr.v_array = np.ones(distr.v_array_len)
        distr.p = np.concatenate((distr.h_array, distr.v_array))
        distr.normalize()

    def normalize(distr):
        distr.p = np.concatenate((distr.h_array, distr.v_array))
        distr.p = distr.p / np.sum(distr.p)

    def generate(distr):
        s = np.random.choice(np.size(distr.p), p=distr.p)
        if s >= distr.h_array_len:
            return [distr.h_state_num, distr.v_state_num + s - distr.h_array_len]

        else:
            return [distr.h_state_num + s, distr.v_state_num]


class Policy:

    def __init__(policy, w, h):
        policy.w = w
        policy.h = h
        policy.terminal_state_vector = [enumerate(np.ones(h, w - 1)), enumerate(np.ones(h - 1, w))]
        policy.distribution = []

        for i in range(policy.terminal_state_vector[0]):
            for j in range(policy.terminal_state_vector[1]):
                policy.distribution.append(distribution())
