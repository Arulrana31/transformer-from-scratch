import numpy as np
from Functions import function


class Layer:
    def __init__(self, nodes, w, b, f_name):  # w is a matrix, b is a column numpy array
        self.nodes = nodes
        self.w = w
        self.b = b
        self.function = function(f_name)

    def compute(self, prev_value, switch):  # prev_value must be a column numpy array
        if switch == 0:
            value = self.function.normal(np.dot(self.w, prev_value) + self.b)
            return value
        elif switch == 1:
            k = np.dot(self.w, prev_value) + self.b
            value = self.function.normal(k)
            return value, k
