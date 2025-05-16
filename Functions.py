import numpy as np


class function:
    def __init__(self, name: str, alpha = 0.1):
        """
        Args:
            name (str): Name of the activation or cost function.
            alpha (float): Slope for LeakyReLU (if applicable).
        """
        self.name = name
        self.alpha = alpha

    def normal(self, value: np.ndarray) -> np.ndarray:
        """
        Compute the activation function output.
        """
        if self.name == "ReLU":
            return np.maximum(0, value)
        elif self.name == "identity":
            return value
        elif self.name == "LeakyReLU":
            return np.maximum(self.alpha * value, value)
        elif self.name == "SoftMax":
            shifted = value - np.max(value, axis=0, keepdims=True)
            exp_values = np.exp(shifted)
            return exp_values / np.sum(exp_values, axis=0, keepdims=True)
        elif self.name == "Sigmoid":
            return 1 / (1 + np.exp(-value))
        else:
            raise ValueError(f"Unknown activation function: {self.name}")

    def derivative(self, value: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation function.
        """
        if self.name == "ReLU":
            return (value > 0).astype(float)
        elif self.name == "identity":
            return np.ones_like(value)
        elif self.name == "LeakyReLU":
            return np.where(value > 0, 1.0, self.alpha)
        elif self.name == "Sigmoid":
            sig = self.normal(value)
            return sig * (1 - sig)
        else:
            raise ValueError(f"Unknown activation function for derivative: {self.name}")

    def cost_normal(self, value: np.ndarray, target: np.ndarray) -> float:
        """
        Compute the cost function.
        """
        if self.name == "MSE":
            self._check_scalar_shape(value, target)
            return float((value - target) ** 2)
        elif self.name == "MAE":
            self._check_scalar_shape(value, target)
            return float(abs(value - target))
        elif self.name == "cross_entropy":
            self._check_same_shape(value, target)
            return float(-np.sum(target * np.log(value + 1e-9)))  # Add epsilon to prevent log(0)
        else:
            raise ValueError(f"Unknown cost function: {self.name}")

    def cost_der(self, value: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cost function.
        """
        if self.name == "MSE":
            self._check_scalar_shape(value, target)
            return 2 * (value - target)
        elif self.name == "MAE":
            self._check_scalar_shape(value, target)
            return np.where(value > target, 1.0, -1.0)
        elif self.name == "cross_entropy":
            softmax = function("SoftMax").normal(value)
            self._check_same_shape(softmax, target)
            return softmax - target
        else:
            raise ValueError(f"Unknown cost function for derivative: {self.name}")

    @staticmethod
    def _check_scalar_shape(value: np.ndarray, target: np.ndarray):
        if value.shape != (1, 1) or target.shape != (1, 1):
            raise ValueError("This cost function requires scalar values of shape (1, 1)")

    @staticmethod
    def _check_same_shape(a: np.ndarray, b: np.ndarray):
        if a.shape != b.shape:
            raise ValueError("Value and target must have the same shape")