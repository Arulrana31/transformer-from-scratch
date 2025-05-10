import numpy as np


class function:
    def __init__(self, name, alpha=0.1):
        self.name = name
        self.alpha = alpha

    def normal(self, value):  # value should be column numpy array
        if self.name == "ReLU":
            return np.maximum(0, value)

        elif self.name == "identity":
            return value

        elif self.name == "LeakyReLU":
            return np.maximum(self.alpha * value, value)

        elif self.name == "SoftMax":
            shifted = value - np.max(value)  # Subtract max to avoid overflow
            exp_values = np.exp(shifted)
            return exp_values / np.sum(exp_values, axis=0, keepdims=True)

        elif self.name == "Sigmoid":
            return 1 / (1 + np.exp(-value))

        else:
            raise ValueError("No function found")

    def derivative(self, value):
        if self.name == "ReLU":
            return np.where(value > 0, 1.0, 0.0)

        elif self.name == "identity":
            return np.ones_like()

        elif self.name == "LeakyReLU":
            return np.where(value > 0, 1.0, self.alpha)

        elif self.name == "Sigmoid":
            k = self.normal(value)
            return k * (1 - k)

        else:
            raise ValueError("No function found")

    def cost_normal(self, value, wanted_value):
        if self.name == "MSE":
            if value.shape != (1, 1) or wanted_value.shape != (1, 1):
                raise ValueError("MSE can only take single values")
            else:
                return (value[0][0] - wanted_value[0][0]) ** 2

        elif self.name == "MAE":
            if value.shape != (1, 1) or wanted_value.shape != (1, 1):
                raise ValueError("MAE can only take single values")
            else:
                return abs(value[0][0] - wanted_value[0][0])

        elif self.name == "cross_entropy":
            if value.shape != wanted_value.shape:
                raise ValueError("CrossEntropy requires value and wanted_value to have the same shape")
            else:
                loss = -np.sum(wanted_value * np.log(value))
                return loss

    def cost_der(self, value, wanted_value):

        if self.name == "MSE":
            if value.shape != (1, 1) or wanted_value.shape != (1, 1):
                raise ValueError("MSE can only take single values")
            else:
                return 2 * (value[0][0] - wanted_value[0][0])

        elif self.name == "MAE":
            if value.shape != (1, 1) or wanted_value.shape != (1, 1):
                raise ValueError("MAE can only take single values")
            else:
                return np.where(value[0][0] - wanted_value[0][0] > 0, 1.0, -1.0)

        elif self.name == "cross_entropy":
            k = function("SoftMax").normal(value) - wanted_value
            if value.shape != k.shape:
                raise ValueError("Please subtract column arrays in cross entropy derivative")
            else:
                return function("SoftMax").normal(value) - wanted_value

        else:
            raise ValueError("No function found")
