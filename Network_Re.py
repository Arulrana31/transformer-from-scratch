import numpy as np
from Functions import function
from Layers import Layer


class Network:
    def __init__(self, n_layers, nodes_per_layer, function_list):
        self.n_layers = n_layers
        self.nodes_per_layer = nodes_per_layer
        self.function_list = function_list

        self.layers = None
        self.bias_correction = None
        self.m_w = [None]
        self.v_w = [None]
        self.m_b = [None]
        self.v_b = [None]

    def initialize(self):

        self.bias_correction = 0

        layers = np.full(self.n_layers, None, dtype=object)
        layers[0] = Layer(self.nodes_per_layer[0], 0, 0, "ReLU")

        for i in range(1, self.n_layers):
            prev_nodes = self.nodes_per_layer[i - 1]
            current_nodes = self.nodes_per_layer[i]
            activation = self.function_list[i - 1]

            if activation == "ReLU" or activation == "identity":
                # He initialization for ReLU
                scale = np.sqrt(2.0 / prev_nodes)
            elif activation == "sigmoid" or activation == "tanh":
                # Xavier/Glorot initialization
                scale = np.sqrt(1.0 / prev_nodes)
            elif activation == "LeakyReLU":
                # He initialization variant for LeakyReLU
                alpha = function(activation).alpha
                scale = np.sqrt(2.0 / (1 + alpha ** 2)) / np.sqrt(
                    prev_nodes
                )  # alpha=0.01
            else:
                # Default: He initialization
                scale = np.sqrt(2.0 / prev_nodes)

            weight_matrix = (
                                np.random.randn(current_nodes, prev_nodes)
                            ) * scale  # (current_layer_nodes, previous_layer_nodes)
            bias_matrix = (
                    np.random.randn(current_nodes, 1) * 0.01
            )  # (current_layer_nodes, 1)

            layers[i] = Layer(
                current_nodes,
                weight_matrix,
                bias_matrix,
                activation,
            )

        self.layers = layers

        self.m_w = [None] * self.n_layers
        self.v_w = [None] * self.n_layers
        self.m_b = [None] * self.n_layers
        self.v_b = [None] * self.n_layers

        for k in range(1, self.n_layers):
            self.m_w[k] = np.zeros_like(self.layers[k].w)
            self.v_w[k] = np.zeros_like(self.layers[k].w)
            self.m_b[k] = np.zeros_like(self.layers[k].b)
            self.v_b[k] = np.zeros_like(self.layers[k].b)

    def compute_network(self, X):
        row, column = X.shape
        empty_matrix = np.empty((self.nodes_per_layer[-1], column))

        for i in range(column):
            value = X[:, i:i + 1]

            for j in range(1, self.n_layers):
                value = self.layers[j].compute(value, 0)

            empty_matrix[:, i:i + 1] = value

        return empty_matrix + X

    def forward_pass(self, X):
        num_samples = X.shape[1]

        # Initialize activation and pre-activation arrays
        a = np.full(self.n_layers, None, dtype=object)
        z = np.full(self.n_layers, None, dtype=object)
        a[0] = X  # Input layer activations

        for layer_idx in range(1, self.n_layers):
            output_dim = self.nodes_per_layer[layer_idx]
            a[layer_idx] = np.empty((output_dim, num_samples))
            z[layer_idx] = np.empty((output_dim, num_samples))

        # Process each sample column-wise
        for sample_idx in range(num_samples):
            current_activation = X[:, sample_idx:sample_idx + 1]  # Maintain column vector

            for layer_idx in range(1, self.n_layers):
                new_activation, pre_activation = self.layers[layer_idx].compute(current_activation, 1)

                a[layer_idx][:, sample_idx:sample_idx + 1] = new_activation
                z[layer_idx][:, sample_idx:sample_idx + 1] = pre_activation

                current_activation = new_activation

        return a, z

    def back_prop(self, X, delta):
        embed_size, batch_size = X.shape
        a, z = self.forward_pass(X)

        d_X = np.empty_like(X, dtype=np.float64)

        w_sum = np.full(self.n_layers, None, dtype=object)
        b_sum = np.full(self.n_layers, None, dtype=object)
        for layer_index in range(1, self.n_layers):
            w_sum[layer_index] = np.zeros((self.nodes_per_layer[layer_index], self.nodes_per_layer[layer_index - 1]))
            b_sum[layer_index] = np.zeros((self.nodes_per_layer[layer_index], 1))

        for batch_index in range(batch_size):
            delta_w = np.full(self.n_layers, None, dtype=object)
            delta_b = np.full(self.n_layers, None, dtype=object)

            delta_temp = delta[:, batch_index:batch_index + 1]

            for i in range(self.n_layers - 1):
                delta_w[-1 - i] = np.dot(delta_temp, a[-2 - i][:, batch_index:batch_index + 1].T)
                delta_b[-1 - i] = delta_temp.copy()

                if i != self.n_layers - 2:
                    delta_temp = np.dot(delta_temp.T, self.layers[-1 - i].w).T * self.layers[
                        -2 - i
                        ].function.derivative(z[-2 - i][:, batch_index:batch_index + 1])

            delta_temp2 = np.dot(delta_temp.T, self.layers[1].w).T
            d_X[:, batch_index:batch_index + 1] = delta_temp2

            for j in range(1, self.n_layers):
                if delta_w[j] is not None:  # Skip input layer
                    w_sum[j] += delta_w[j]
                if delta_b[j] is not None:  # Skip input layer
                    b_sum[j] += delta_b[j]

        return (
            [w_sum[i] / batch_size if w_sum[i] is not None else None for i in range(self.n_layers)],
            [b_sum[i] / batch_size if b_sum[i] is not None else None for i in range(self.n_layers)],
            d_X
        )

    def Network_Update(
            self,
            X_train,  # List/array of input samples (column vectors)
            delta,
            learning_rate=0.01,
            beta1=0.9,  # Adam: exponential decay rate for 1st moment estimates
            beta2=0.999,  # Adam: exponential decay rate for 2nd moment estimates
            epsilon=1e-8,  # Adam: small constant for numerical stability
            lambda_=1e-4,
            reg="L2",
    ):
        n = X_train.shape[1]
        batch_size = X_train.shape[1]

        self.bias_correction += 1

        X_used = X_train
        delta_used = delta
        batch_w, batch_b, d_X = self.back_prop(X_used, delta_used)
        for k in range(1, self.n_layers):
            if batch_w[k] is not None:  # Skip input layer
                # Add L2 regularization term to gradient
                if reg == "L2":
                    batch_w[k] = batch_w[k] + lambda_ * self.layers[k].w

                elif reg == "L1":
                    batch_w[k] = batch_w[k] + lambda_ * np.sign(
                        self.layers[k].w
                    )

                self.m_w[k] = beta1 * self.m_w[k] + (1 - beta1) * batch_w[k]
                self.v_w[k] = beta2 * self.v_w[k] + (1 - beta2) * (batch_w[k] ** 2)

                m_hat_w = self.m_w[k] / (1 - beta1 ** self.bias_correction)
                v_hat_w = self.v_w[k] / (1 - beta2 ** self.bias_correction)

                self.layers[k].w -= (
                        learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
                )

            if batch_b[k] is not None:
                self.m_b[k] = beta1 * self.m_b[k] + (1 - beta1) * batch_b[k]
                self.v_b[k] = beta2 * self.v_b[k] + (1 - beta2) * (batch_b[k] ** 2)

                m_hat_b = self.m_b[k] / (1 - beta1 ** self.bias_correction)
                v_hat_b = self.v_b[k] / (1 - beta2 ** self.bias_correction)

                self.layers[k].b -= (
                        learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
                )

        return d_X
