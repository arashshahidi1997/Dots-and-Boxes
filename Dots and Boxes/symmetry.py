import numpy as np

w = 4
h = 3
dim = h * (w - 1) + (h - 1) * w


def enumerate(vector):
    return int("".join(str(v) for v in vector), 2)


def vectorize(num):
    vector = np.zeros(dim, dtype='int')
    v = np.array([int(s) for s in bin(num)[2:]])
    vector[dim - np.size(v):] = v
    return vector


def vector_form(array_list):
    return np.concatenate((np.concatenate(array_list[0]), np.concatenate(array_list[1])))


def matrix_form(vector):
    a = h * (w - 1)
    vector_list = [vector[:a], vector[a:]]
    array_list = [np.ndarray((h, w - 1), dtype='int'), np.ndarray((h - 1, w), dtype='int')]

    for i in range(w - 1):
        array_list[0][:, i] = vector_list[0][i::(w - 1)]

    for j in range(w):
        array_list[1][:, j] = vector_list[1][j::w]

    return array_list


def Symmetry_Group():

    # identity
    def id(arr_list):
        return arr_list

    # horizontal symmetry
    def hor(arr_list):
        return [arr_list[0][:, ::-1], arr_list[1][:, ::-1]]

    # vertical symmetry
    def ver(arr_list):
        return [arr_list[0][::-1], arr_list[1][::-1]]

    # 180 degree rotation:
    def r180(arr_list):
        return [arr_list[0][::-1, ::-1], arr_list[1][::-1, ::-1]]

    # diagonal symmetry if w = h
    def d1(arr_list):
        return [np.transpose(arr_list[1][::-1, ::-1]), np.transpose(arr_list[0][::-1, ::-1])]

    def d2(arr_list):
        return [np.transpose(arr_list[1]), np.transpose(arr_list[0])]

    def r90(arr_list):
        return [np.transpose(arr_list[1])[::-1], np.transpose(arr_list[0])[::-1]]

    def r270(arr_list):
        return [np.transpose(arr_list[1])[:, ::-1], np.transpose(arr_list[0])[:, ::-1]]

    def c1(arr_list):
        arr_list[0][0, 0], arr_list[1][0, 0] = arr_list[1][0, 0], arr_list[0][0, 0]
        return arr_list

    def c2(arr_list):
        arr_list[0][0, -1], arr_list[1][0, -1] = arr_list[1][0, -1], arr_list[0][0, -1]
        return arr_list

    def c3(arr_list):
        arr_list[0][-1, 0], arr_list[1][-1, 0] = arr_list[1][-1, 0], arr_list[0][-1, 0]
        return arr_list

    def c4(arr_list):
        arr_list[0][-1, -1], arr_list[1][-1, -1] = arr_list[1][-1, -1], arr_list[0][-1, -1]
        return arr_list

    return [id, hor, ver, r180, d1, d2, r90, r270, c1, c2, c3, c4]


transformation = Symmetry_Group()
inverse = [0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 10, 11]
multiply = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                     [1, 0, 3, 2, 6, 7, 4, 5],
                     [2, 3, 0, 1, 7, 6, 5, 4],
                     [3, 2, 1, 0, 5, 4, 7, 6],
                     [4, 7, 6, 5, 0, 3, 2, 1],
                     [5, 6, 7, 4, 3, 0, 1, 2],
                     [6, 5, 4, 7, 1, 2, 3, 0],
                     [7, 4, 5, 6, 2, 1, 0, 3]])

terminal_vector = np.ones(dim, dtype='int')
terminal_index = enumerate(terminal_vector)

index_list = np.ndarray((terminal_index + 1, 2), dtype='int')

s_list = list(range(terminal_index + 1))

while s_list:
    print(len(s_list))
    array_list = matrix_form(vectorize(s_list[0]))
    v = []

    for i in range(4):
        v.append(enumerate(vector_form(transformation[i](array_list))))

    if h == w:
        for i in range(4, 8):
            v.append(enumerate(vector_form(transformation[i](array_list))))

    # for i in range(8, 12):
    #    v.append(enumerate((vector_form(transformation[i](array_list)))))

    v_f = min(v)
    v_f_complement = terminal_index - max(v)
    t_f_complement = np.argmax(v)
    v_set = set(v)

    for x in v_set:
        y = terminal_index - x
        if x in s_list:
            s_list.remove(x)

        if y in s_list:
            s_list.remove(y)

        t = v.index(x)
        index_list[x] = [v_f, t]
        index_list[y] = [v_f_complement, multiply[t, inverse[t_f_complement]]]

np.save('symmetries/index_list' + str(h) + '_' + str(w) + '.npy', index_list)

# print(index_list)
# k = 0
# for i in reduced_index_list:
#     print(k, ':', vectorize(i, (w-1)*h + (h-1)*w))
#     k += 1
