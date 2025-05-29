import numpy as np
from Functions import function


class AttentionHead:
    def __init__(self, embed_dimension: int, output_dimension: int, mask: bool):
        """
        Initializes an AttentionHead.

        Args:
            embed_dimension (int): Input embedding size (d_model).
            output_dimension (int): Output size per head (typically d_model / num_heads).
        """
        self.embed_dimension = embed_dimension
        self.output_dimension = output_dimension
        self.maxlen = 512
        self.mask = mask

        self.number_embeds = None
        self.pe = self._init_positional_encoding()

        self.w_q = self.w_k = self.w_v = None
        self.beta_q_vector = self.beta_k_vector = self.beta_v_vector = None

        self.t = 0
        self._init_optimizer_states()

        self.output = None

    def _init_optimizer_states(self):
        """
        Initializes the optimizer states elsewhere to avoid clutter.
        """
        self.m_w_q = self.v_w_q = None
        self.m_w_k = self.v_w_k = None
        self.m_w_v = self.v_w_v = None
        self.m_beta_q = self.v_beta_q = None
        self.m_beta_k = self.v_beta_k = None
        self.m_beta_v = self.v_beta_v = None

    def initialize(self):

        d = self.embed_dimension
        d_ = self.output_dimension

        self.w_q = np.random.randn(d_, d)
        self.w_k = np.random.randn(d_, d)
        self.w_v = np.random.randn(d_, d)

        self.beta_q_vector = np.random.randn(d_, 1)
        self.beta_k_vector = np.random.randn(d_, 1)
        self.beta_v_vector = np.random.randn(d_, 1)

        self.m_w_q = np.zeros_like(self.w_q)
        self.v_w_q = np.zeros_like(self.w_q)
        self.m_w_k = np.zeros_like(self.w_k)
        self.v_w_k = np.zeros_like(self.w_k)
        self.m_w_v = np.zeros_like(self.w_v)
        self.v_w_v = np.zeros_like(self.w_v)

        self.m_beta_q = np.zeros_like(self.beta_q_vector)
        self.v_beta_q = np.zeros_like(self.beta_q_vector)
        self.m_beta_k = np.zeros_like(self.beta_k_vector)
        self.v_beta_k = np.zeros_like(self.beta_k_vector)
        self.m_beta_v = np.zeros_like(self.beta_v_vector)
        self.v_beta_v = np.zeros_like(self.beta_v_vector)

        self.t = 0

    def _init_positional_encoding(self) -> np.ndarray:
        """
        Generates sinusoidal positional encoding matrix.

        Returns:
            np.ndarray: A matrix of shape (embed_dimension, maxlen).
        """
        position = np.arange(self.maxlen)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dimension, 2) *
                          (-np.log(10000.0) / self.embed_dimension))  # [d_model//2]

        pe = np.zeros((self.maxlen, self.embed_dimension))  # [max_len, d_model]
        pe[:, 0::2] = np.sin(position * div_term)  # even indices
        pe[:, 1::2] = np.cos(position * div_term[:pe.shape[1] // 2]) # odd indices

        return pe.T  # [d_model, max_len]

    def compute(self, X: np.ndarray, switch=0, position=True):
        """
        Computes the self-attention output.

        Args:
            X (np.ndarray): Input tensor of shape (embed_dimension, sequence_length).
            switch (int): If 1, returns intermediate values for backprop.
            position (bool): Whether to add positional encoding.
            mask (bool): Whether to apply causal mask.

        Returns:
            np.ndarray or tuple: Output tensor or tuple with intermediate results.
        """
        if X.shape[0] != self.embed_dimension:
            raise ValueError("Input dimension does not match embed_dimension")

        self.number_embeds = X.shape[1]
        n = self.number_embeds

        if position:
            pos_encoding = self.pe[:, :self.number_embeds]
            X_pos = X + pos_encoding
        else:
            X_pos = X

        Q = np.dot(self.w_q, X_pos) + np.tile(self.beta_q_vector, (1, self.number_embeds))
        K = np.dot(self.w_k, X_pos) + np.tile(self.beta_k_vector, (1, self.number_embeds))
        V = np.dot(self.w_v, X) + np.tile(self.beta_v_vector, (1, self.number_embeds))

        dot_product = np.dot(Q.T, K) / np.sqrt(self.output_dimension)  # nxn

        if self.mask:
            mask_matrix = np.tri(n, dtype=bool)  # Lower-triangular boolean matrix (n × n)
            mask_matrix = np.where(mask_matrix, 0, -np.inf) # Upper triangle becomes minus infinity
            dot_product = mask_matrix + dot_product

        soft_dot_product = np.zeros(dot_product.shape)

        for i in range(n):
            temp_list = function("SoftMax").normal(dot_product[i])
            for j in range(n):
                soft_dot_product[i, j] = temp_list[j]

        Sa = V @ soft_dot_product.T

        self.output = Sa

        if switch == 1:
            return Sa, Q, K, V, soft_dot_product, X_pos
        else:
            return Sa

    def backprop(self, delta: np.ndarray, X: np.ndarray):
        """
        Backward pass through the attention head.

        Args:
            delta (np.ndarray): Gradient from next layer, shape (output_dimension, sequence_length).
            X (np.ndarray): Input tensor.

        Returns:
            dict: Gradients for weights, biases, and input.
        """
        # Forward pass
        Sa, Q, K, V, A, X_pos = self.compute(X, switch=1)

        dL_dV = delta @ A.T  # [d_out, n]
        dL_dA = V.T @ delta  # [n, n]

        dL_dScores = A * (dL_dA - np.sum(A * dL_dA, axis=1, keepdims=True))  # [n, n]

        dL_dQ = (K @ dL_dScores.T) / np.sqrt(self.output_dimension)  # [d_out, n]
        dL_dK = (Q @ dL_dScores) / np.sqrt(self.output_dimension)  # [d_out, n]

        dL_dw_q = dL_dQ @ X_pos.T  # [d_out, d]
        dL_dw_k = dL_dK @ X_pos.T  # [d_out, d]
        dL_dw_v = dL_dV @ X.T  # [d_out, d]

        dL_dbeta_q = np.sum(dL_dQ, axis=1, keepdims=True)  # [d_out, 1]
        dL_dbeta_k = np.sum(dL_dK, axis=1, keepdims=True)  # [d_out, 1]
        dL_dbeta_v = np.sum(dL_dV, axis=1, keepdims=True)  # [d_out, 1]

        dL_dX = (
                self.w_q.T @ dL_dQ +  # [d_out, n]
                self.w_k.T @ dL_dK +  # [d_out, n]
                self.w_v.T @ dL_dV  # [d_out, n]
        )

        return {
            'w_q': dL_dw_q,
            'w_k': dL_dw_k,
            'w_v': dL_dw_v,
            'beta_q': dL_dbeta_q,
            'beta_k': dL_dbeta_k,
            'beta_v': dL_dbeta_v,
            'X': dL_dX
        }

    def batchprop(self, X_set, delta_set):
        """
        Perform batch-wise backpropagation.

        Args:
            X_set (list of np.ndarray): List of input instances.
            delta_set (list of np.ndarray): List of corresponding deltas.

        Returns:
            dict: Averaged gradients and list of input gradients.
        """
        if len(X_set) != len(delta_set):
            raise ValueError("X_set and delta_set must have the same length")

        # Initialize accumulators
        grad_accumulator = {
            'w_q': 0,
            'w_k': 0,
            'w_v': 0,
            'beta_q': 0,
            'beta_k': 0,
            'beta_v': 0,
        }
        dX_list = []
        total_tokens = 0

        for X, delta in zip(X_set, delta_set):
            grads = self.backprop(delta, X)
            tokens = X.shape[1]
            total_tokens += tokens
            for key in grad_accumulator:
                grad_accumulator[key] += grads[key] * tokens
            dX_list.append(grads['X'])

        # Average the gradients weighted by token count
        for key in grad_accumulator:
            grad_accumulator[key] /= total_tokens

        grad_accumulator['X'] = dX_list
        return grad_accumulator

    def update(self, X_set, delta_set, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Updates parameters using Adam optimizer.

        Args:
            X (np.ndarray): Input tensor.
            delta (np.ndarray): Gradient from next layer.
            learning_rate (float): Learning rate.
            beta1 (float): First moment decay rate.
            beta2 (float): Second moment decay rate.
            epsilon (float): Small constant to avoid division by zero.

        Returns:
            np.ndarray: Gradient with respect to input X.
        """
        self.t += 1

        # Get gradients from backprop
        grads = self.batchprop(X_set, delta_set)

        self.m_w_q = beta1 * self.m_w_q + (1 - beta1) * grads['w_q']
        self.v_w_q = beta2 * self.v_w_q + (1 - beta2) * (grads['w_q'] ** 2)
        m_hat_q = self.m_w_q / (1 - beta1 ** self.t)
        v_hat_q = self.v_w_q / (1 - beta2 ** self.t)
        self.w_q -= learning_rate * m_hat_q / (np.sqrt(v_hat_q) + epsilon)

        self.m_w_k = beta1 * self.m_w_k + (1 - beta1) * grads['w_k']
        self.v_w_k = beta2 * self.v_w_k + (1 - beta2) * (grads['w_k'] ** 2)
        m_hat_k = self.m_w_k / (1 - beta1 ** self.t)
        v_hat_k = self.v_w_k / (1 - beta2 ** self.t)
        self.w_k -= learning_rate * m_hat_k / (np.sqrt(v_hat_k) + epsilon)

        self.m_w_v = beta1 * self.m_w_v + (1 - beta1) * grads['w_v']
        self.v_w_v = beta2 * self.v_w_v + (1 - beta2) * (grads['w_v'] ** 2)
        m_hat_v = self.m_w_v / (1 - beta1 ** self.t)
        v_hat_v = self.v_w_v / (1 - beta2 ** self.t)
        self.w_v -= learning_rate * m_hat_v / (np.sqrt(v_hat_v) + epsilon)

        self.m_beta_q = beta1 * self.m_beta_q + (1 - beta1) * grads['beta_q']
        self.v_beta_q = beta2 * self.v_beta_q + (1 - beta2) * (grads['beta_q'] ** 2)
        m_hat_bq = self.m_beta_q / (1 - beta1 ** self.t)
        v_hat_bq = self.v_beta_q / (1 - beta2 ** self.t)
        self.beta_q_vector -= learning_rate * m_hat_bq / (np.sqrt(v_hat_bq) + epsilon)

        self.m_beta_k = beta1 * self.m_beta_k + (1 - beta1) * grads['beta_k']
        self.v_beta_k = beta2 * self.v_beta_k + (1 - beta2) * (grads['beta_k'] ** 2)
        m_hat_bk = self.m_beta_k / (1 - beta1 ** self.t)
        v_hat_bk = self.v_beta_k / (1 - beta2 ** self.t)
        self.beta_k_vector -= learning_rate * m_hat_bk / (np.sqrt(v_hat_bk) + epsilon)

        self.m_beta_v = beta1 * self.m_beta_v + (1 - beta1) * grads['beta_v']
        self.v_beta_v = beta2 * self.v_beta_v + (1 - beta2) * (grads['beta_v'] ** 2)
        m_hat_bv = self.m_beta_v / (1 - beta1 ** self.t)
        v_hat_bv = self.v_beta_v / (1 - beta2 ** self.t)
        self.beta_v_vector -= learning_rate * m_hat_bv / (np.sqrt(v_hat_bv) + epsilon)

        return grads["X"]


class AttentionLayer:
    def __init__(self, embed_dimension: int, number_heads: int, mask: bool):
        """
        Multi-head attention layer composed of multiple AttentionHead instances.

        Args:
            embed_dimension (int): Total embedding dimension.
            number_heads (int): Number of attention heads.
        """
        if embed_dimension % number_heads != 0:
            raise ValueError("embed_dimension must be divisible by number_heads")

        self.embed_dimension = embed_dimension
        self.number_embeds = None
        self.number_heads = number_heads
        self.mask = mask

        self.attention_list = None

    def initialize(self):
        """Initializes all attention heads."""
        output_dimension = int(self.embed_dimension / self.number_heads)

        attention_list = np.empty(self.number_heads, dtype=object)

        for i in range(self.number_heads):
            attention_list[i] = AttentionHead(self.embed_dimension, output_dimension, self.mask)
            attention_list[i].initialize()

        self.attention_list = attention_list

    def compute(self, X: np.ndarray):
        """
        Applies multi-head attention.

        Args:
            X (np.ndarray): Input tensor of shape (embed_dimension, sequence_length).
            mask_layer (bool): Whether to apply attention masking.

        Returns:
            np.ndarray: Output tensor after attention (same shape as input).
        """
        if X.shape[0] != self.embed_dimension:
            raise ValueError("Input shape does not match embed_dimension")

        self.number_embeds = X.shape[1]

        outputs = [head.compute(X) for head in self.attention_list]
        return X + np.vstack(outputs)

    def update(self, X_set, delta_set, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Backpropagates through all attention heads and updates parameters.

        Args:
            X (np.ndarray): Input tensor.
            delta (np.ndarray): Output gradient from the next layer.
            learning_rate (float): Learning rate.
            beta1 (float): Adam beta1.
            beta2 (float): Adam beta2.
            epsilon (float): Small constant to avoid division by zero.

        Returns:
            np.ndarray: Gradient with respect to input X.
        """
        output_dimension = int(self.embed_dimension / self.number_heads)

        delta_head = []
        for i in range(self.number_heads):
            delta_head.append([d[i * output_dimension: (i + 1) * output_dimension, :]for d in delta_set])

        dL_dX_list = [self.attention_list[i].update(X_set, delta_head[i],
                                            learning_rate=learning_rate,
                                            beta1=beta1,
                                            beta2=beta2,
                                            epsilon=epsilon) for i in range(self.number_heads)]


        # Now vertically stack each group
        stacked_outputs = [np.sum(arrays, axis=0) for arrays in zip(*dL_dX_list)]

        return stacked_outputs
