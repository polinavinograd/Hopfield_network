from random import uniform

import numpy as np
from tqdm import tqdm

from config import weights_path
from utils import create_input_vectors


class HopfieldNetwork:
    def __init__(self, weights=None):
        self.inputs = create_input_vectors()
        if weights is None:
            self.weight = np.zeros((len(self.inputs[0][0]), len(self.inputs[0][0])))
        else:
            self.weight = np.load(weights)

    def train(self):
        for input in tqdm(self.inputs):
            rate = 1 / ((input @ input.T) - (input @ self.weight @ input.T))
            temp_matrix = (self.weight @ input.T) - input.T
            matrix = temp_matrix @ temp_matrix.T
            self.weight += rate * matrix
        for i in range(len(self.weight)):
            self.weight[i][i] = 0.
        np.save(weights_path, self.weight)

    def test(self, vector):
        new_vector = np.array([])
        while True:
            new_vector = self.activate(vector @ self.weight)
            if np.array_equal(new_vector, vector):
                break
            vector = self.asynchrono(vector, new_vector)
        return new_vector

    @staticmethod
    def asynchrono(old_vector, new_vector):
        result = old_vector
        for i in range(len(result[0])):
            result[0][i] = old_vector[0][i] if uniform(0, 65555) % 2 == 1 else new_vector[0][i]
        return result

    @staticmethod
    def activate(vector):
        result = []
        for el in vector[0]:
            result.append(1. if el > 0 else -1.)
        return np.array([result])
