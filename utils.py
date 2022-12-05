import numpy as np
from matplotlib import pyplot as plt, image as mpimg
from config import path_pngs
from math import sqrt
from random import choice


def view(pic, after):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(pic)
    axs[1].imshow(after)
    axs[0].set_title("Образ")
    axs[1].set_title("Распознано")
    plt.show()


def create_input_vectors():
    result = []
    for i in range(10):
        pic = mpimg.imread(path_pngs + f"{i}.png")
        input_vec = []
        for row in pic:
            for el in row:
                if el[0] == 0:
                    input_vec.append(-1.)
                else:
                    input_vec.append(1.)
        result.append([input_vec])
    return np.array(result)


def create_input(pic):
    input_vec = []
    for row in pic:
        for el in row:
            if el[0] == 0:
                input_vec.append(-1.)
            else:
                input_vec.append(1.)
    return np.array([input_vec])


def create_pic(vector):
    pic = np.zeros((int(sqrt(len(vector[0]))), int(sqrt(len(vector[0]))), 3))
    for i in range(len(vector[0])):
        pic[i // int(sqrt(len(vector[0])))][i % int(sqrt(len(vector[0])))] = np.array([0., 0., 0.]) \
            if vector[0][i] == -1 else np.array([1., 1., 1.])
    return pic


def distort(vector, percent):
    part = int(len(vector[0]) * percent)
    ls = []
    while len(ls) != part:
        cho = choice(range(784))
        if cho not in ls:
            ls.append(cho)
    for i in ls:
        vector[0][i] *= -1
    return vector
