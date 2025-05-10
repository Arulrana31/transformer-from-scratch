import numpy as np
from Functions import function


class AttentionHead:
    def __init__(self, embed_dimension, output_dimension):
        self.embed_dimension = embed_dimension
        self.number_embeds = None
        self.output_dimension = output_dimension
        self.maxlen = 512
        self.pe = self.positional_encoding()

        self.w_q = None
        self.w_k = None
        self.w_v = None
        self.beta_q_vector = None
        self.beta_k_vector = None
        self.beta_v_vector = None
        self.output = None

        self.t = None
        self.m_w_q = None
        self.m_w_k = None
        self.m_w_v = None
        self.m_beta_q = None
        self.m_beta_k = None
        self.m_beta_v = None
        self.v_w_q = None
        self.v_w_k = None
        self.v_w_v = None
        self.v_beta_q = None
        self.v_beta_k = None
        self.v_beta_v = None

    def initialize_attention(self):

        d = self.embed_dimension
        d_ = self.output_dimension

        w_q = np.random.randn(d_, d)
        w_k = np.random.randn(d_, d)
        w_v = np.random.randn(d_, d)
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v

        beta_q_vector = np.random.randn(d_, 1)
        beta_k_vector = np.random.randn(d_, 1)
        beta_v_vector = np.random.randn(d_, 1)
        self.beta_q_vector = beta_q_vector
        self.beta_k_vector = beta_k_vector
        self.beta_v_vector = beta_v_vector

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

    def positional_encoding(self):
        position = np.arange(self.maxlen)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embed_dimension, 2) *
                          (-np.log(10000.0) / self.embed_dimension))  # [d_model//2]

        pe = np.zeros((self.maxlen, self.embed_dimension))  # [max_len, d_model]
        pe[:, 0::2] = np.sin(position * div_term)  # even indices
        pe[:, 1::2] = np.cos(position * div_term)  # odd indices

        return pe.T  # [d_model, max_len]

    def compute_attention(self, X, switch=0, position=True, mask=True):
        self.number_embeds = X.shape[1]
        n = self.number_embeds

        if position:
            pos_encoding = self.pe[:, :self.number_embeds]
            X_pos = X + pos_encoding
        else:
            X_pos = X

        Q = np.dot(self.w_q, X_pos) + np.tile(self.beta_q_vector, (1, n))
        K = np.dot(self.w_k, X_pos) + np.tile(self.beta_k_vector, (1, n))
        V = np.dot(self.w_v, X) + np.tile(self.beta_v_vector, (1, n))

        dot_product = np.dot(Q.T, K) / np.sqrt(self.output_dimension)  # nxn

        if mask:
            mask_matrix = np.tri(n, dtype=bool)  # Lower-triangular boolean matrix (n × n)
            mask_matrix = np.where(mask_matrix, 0, -np.inf)
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

    def attention_backprop(self, delta, X):
        # Forward pass
        Sa, Q, K, V, A, X_pos = self.compute_attention(X, switch=1)

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
                self.w_q.T @ dL_dQ +  # [d, n]
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

    def Head_Update(self, X, delta, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1

        # Get gradients from backprop
        grads = self.attention_backprop(delta, X)

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
    def __init__(self, embed_dimension, number_heads):
        self.embed_dimension = embed_dimension
        self.number_embeds = None
        self.number_heads = number_heads

        self.attention_list = None

    def initialize_attention_layer(self):
        output_dimension = int(self.embed_dimension / self.number_heads)

        attention_list = np.empty(self.number_heads, dtype=object)

        for i in range(self.number_heads):
            attention_list[i] = AttentionHead(self.embed_dimension, output_dimension)
            attention_list[i].initialize_attention()

        self.attention_list = attention_list

    def compute_attention_layer(self, X, mask_layer=True):
        self.number_embeds = X.shape[1]
        Sa_concatenated = self.attention_list[0].compute_attention(X, mask=mask_layer)

        for i in range(1, self.number_heads):
            temp_Sa = self.attention_list[i].compute_attention(X, mask=mask_layer)
            Sa_concatenated = np.vstack((Sa_concatenated, temp_Sa))

        return X + Sa_concatenated

    def Update(self, X, delta, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        output_dimension = int(self.embed_dimension / self.number_heads)

        dL_dX = np.zeros_like(X, dtype=np.float64)  # Initialize gradient accumulator

        for i in range(self.number_heads):
            delta_head = delta[i * output_dimension: (i + 1) * output_dimension, :]

            dL_dX += self.attention_list[i].Head_Update(
                X,
                delta_head,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon
            )

        return dL_dX
