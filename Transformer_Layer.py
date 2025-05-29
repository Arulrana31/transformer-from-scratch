import numpy as np
from Network import Network
from Attention import AttentionLayer
from Layernorm import LayerNorm
from Logits_Layer import Logit_Layer


class Transformer_Layer:
    def __init__(self, embed_size, heads_number, vocab_size, mask):
        """
        Initialize a Transformer Layer with multi-head attention and feed-forward components.

        Args:
            embed_size (int): Dimensionality of input embeddings.
            heads_number (int): Number of attention heads.
        """
        if embed_size <= 0:
            raise ValueError("embed_size must be a positive integer.")
        if heads_number <= 0:
            raise ValueError("heads_number must be a positive integer.")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer.")

        self.embed_size = embed_size
        self.heads_number = heads_number
        self.mask = mask

        self.Attention_Layer = None
        self.LayerNorm_Layer1 = None
        self.LayerNorm_Layer2 = None
        self.NeuralNetwork = None
        self.Logits_Layer = None
        self.Logit = None

    def initialize(self, Logit = 0):

        self.Logit = Logit

        """Initialize sub-layers (Attention, LayerNorm, Feed-Forward Network)."""
        N = self.embed_size
        self.Attention_Layer = AttentionLayer(N, self.heads_number, self.mask)
        self.Attention_Layer.initialize()

        self.LayerNorm_Layer1 = LayerNorm()
        self.LayerNorm_Layer2 = LayerNorm()

        self.LayerNorm_Layer1.initialize(self.embed_size)
        self.LayerNorm_Layer2.initialize(self.embed_size)

        self.NeuralNetwork = Network(3, [N, 4 * N, N], ["ReLU", "identity"])
        self.NeuralNetwork.initialize()

        if self.Logit:
            self.Logits_Layer = Logit_Layer(self.embed_size, self.vocab_size)

    def compute(self, X):
        """
        Forward pass through the Transformer Layer.

        Args:
            X (np.ndarray): Input tensor of shape (batch_size, sequence_length, embed_size).

        Returns:
            np.ndarray: Output tensor after attention and feed-forward operations.
        """
        value = self.Attention_Layer.compute(X)
        value = self.LayerNorm_Layer1.compute(value)
        value = self.NeuralNetwork.compute(value)
        value = self.LayerNorm_Layer2.compute(value)

        if self.Logit:
            value = self.Logits_Layer.compute(value)

        return value

    def update(self, X_set, delta_set, learning_rate=0.01,
               beta1=0.9,  # Adam: exponential decay rate for 1st moment estimates
               beta2=0.999,  # Adam: exponential decay rate for 2nd moment estimates
               epsilon=1e-8,  # Adam: small constant for numerical stability
               lambda_=1e-4,
               reg="L2", ):
        """
        Update layer parameters using backpropagation (Adam optimizer).

        Args:
            X (np.ndarray): Input tensor.
            delta (np.ndarray): Gradient from the next layer.
            learning_rate (float): Learning rate for updates.
            beta1 (float): Adam decay rate for 1st-moment estimates.
            beta2 (float): Adam decay rate for 2nd-moment estimates.
            epsilon (float): Small constant for numerical stability.
            lambda_ (float): Regularization strength.
            reg (str): Regularization type ("L2" or "L1").

        Returns:
            np.ndarray: Updated gradient for backpropagation.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")


        X_N1 = []
        for X in X_set:
            X_N1.append(self.Attention_Layer.compute(X))
        X_NN = []
        for X in X_N1:
            X_NN.append(self.LayerNorm_Layer1.compute(X))
        X_N2 = []
        for X in X_NN:
            X_N2.append(self.NeuralNetwork.compute(X))

        X_L = []
        if self.Logit:
            for X in X_N2:
                X_L.append(self.Logits_Layer.compute(X))


        if self.Logit:
            delta_set = self.Logits_Layer.compute(X_L, delta_set, learning_rate=learning_rate)
        delta_set = self.LayerNorm_Layer2.update(X_N2, delta_set, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                                       epsilon=epsilon, l2_lambda=lambda_, reg=reg)
        delta_set = self.NeuralNetwork.update(X_NN, delta_set, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                                  epsilon=epsilon, lambda_=lambda_, reg=reg)
        delta_set = self.LayerNorm_Layer1.update(X_N1, delta_set, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                                       epsilon=epsilon, l2_lambda=lambda_, reg=reg)
        delta_set = self.Attention_Layer.update(X_set, delta_set, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                            epsilon=epsilon)
        return delta_set


