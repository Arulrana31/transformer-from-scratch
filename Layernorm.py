import numpy as np


class LayerNorm:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon

        self.beta = None
        self.gamma = None
        self.cache = None
        self.t = None
        self.m_beta = None
        self.m_gamma = None
        self.v_beta = None
        self.v_gamma = None

    def LayerNorm_initialize(self, embed_dimension):
        self.gamma = np.ones((embed_dimension, 1), dtype=np.float64)
        self.beta = np.zeros((embed_dimension, 1), dtype=np.float64)

        self.m_gamma = np.zeros_like(self.gamma)
        self.v_gamma = np.zeros_like(self.gamma)
        self.m_beta = np.zeros_like(self.beta)
        self.v_beta = np.zeros_like(self.beta)

        self.t = 0

    def LayerNorm_compute(self, X):
        mean = np.mean(X, axis=0, keepdims=True)  # (1, num_samples)
        variance = np.var(X, axis=0, keepdims=True)  # (1, num_samples)

        X_norm = (X - mean) / np.sqrt(variance + self.epsilon)

        Y = self.gamma * X_norm + self.beta

        return Y

    def LayerNorm_Backprop(self, X, delta):
        mean = np.mean(X, axis=0, keepdims=True)
        variance = np.var(X, axis=0, keepdims=True)

        X_norm = (X - mean) / np.sqrt(variance + self.epsilon)
        D, N = X.shape

        d_beta = np.sum(delta, axis=1, keepdims=True)  # (D, 1)
        d_gamma = np.sum(delta * X_norm, axis=1, keepdims=True)  # (D, 1)

        std_inv = 1.0 / np.sqrt(variance + self.epsilon)  # (1, N)
        d_X_norm = delta * self.gamma  # (D, N)

        d_var = np.sum(d_X_norm * (X - mean) * -0.5 * ((variance + self.epsilon) ** (-1.5)),
                       axis=0, keepdims=True)  # (1, N)

        d_mean = (np.sum(d_X_norm * -std_inv, axis=0, keepdims=True) + d_var * np.mean(-2.0 * (X - mean), axis=0,
                                                                                       keepdims=True))

        d_X = (d_X_norm * std_inv) + \
              (d_var * 2.0 * (X - mean) / D) + \
              (d_mean / D)

        return d_X, d_gamma, d_beta

    def LayerNorm_Update(
            self,
            X,  # List/array of input samples (column vectors)
            delta,
            learning_rate=0.01,
            beta1=0.9,  # Adam: exponential decay rate for 1st moment estimates
            beta2=0.999,  # Adam: exponential decay rate for 2nd moment estimates
            epsilon=1e-8,  # Adam: small constant for numerical stability
            l2_lambda=0.01,
            reg="L2",
    ):
        d_X, d_gamma, d_beta = self.LayerNorm_Backprop(X, delta)
        # L2 regularization
        if reg == "L2":
            d_gamma += l2_lambda * self.gamma
            d_beta += l2_lambda * self.beta

        # Adam time step
        self.t += 1

        self.m_gamma = beta1 * self.m_gamma + (1 - beta1) * d_gamma
        self.v_gamma = beta2 * self.v_gamma + (1 - beta2) * (d_gamma ** 2)

        self.m_beta = beta1 * self.m_beta + (1 - beta1) * d_beta
        self.v_beta = beta2 * self.v_beta + (1 - beta2) * (d_beta ** 2)

        m_gamma_hat = self.m_gamma / (1 - beta1 ** self.t)
        v_gamma_hat = self.v_gamma / (1 - beta2 ** self.t)

        m_beta_hat = self.m_beta / (1 - beta1 ** self.t)
        v_beta_hat = self.v_beta / (1 - beta2 ** self.t)

        self.gamma -= learning_rate * m_gamma_hat / (np.sqrt(v_gamma_hat) + epsilon)
        self.beta -= learning_rate * m_beta_hat / (np.sqrt(v_beta_hat) + epsilon)

        return d_X
