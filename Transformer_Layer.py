import numpy as np
from Network import Network
from Attention import AttentionLayer
from Layernorm import LayerNorm


class Transformer_Layer:
    def __init__(self, embed_size, heads_number):
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

        self.embed_size = embed_size
        self.heads_number = heads_number

        self.Attention_Layer = None
        self.LayerNorm_Layer1 = None
        self.LayerNorm_Layer2 = None
        self.NeuralNetwork = None

    def initialize(self):
        """Initialize sub-layers (Attention, LayerNorm, Feed-Forward Network)."""
        N = self.embed_size
        self.Attention_Layer = AttentionLayer(N, self.heads_number)
        self.Attention_Layer.initialize()

        self.LayerNorm_Layer1 = LayerNorm()
        self.LayerNorm_Layer2 = LayerNorm()

        self.LayerNorm_Layer1.initialize(self.embed_size)
        self.LayerNorm_Layer2.initialize(self.embed_size)

        self.NeuralNetwork = Network(3, [N, 4 * N, N], ["ReLU", "identity"])
        self.NeuralNetwork.initialize()

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

        return value

    def update(self, X, delta, learning_rate=0.01,
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
        if not isinstance(X, np.ndarray) or not isinstance(delta, np.ndarray):
            raise TypeError("X and delta must be numpy.ndarray.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        delta = self.LayerNorm_Layer2.update(X, delta, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                                       epsilon=epsilon, l2_lambda=lambda_, reg=reg)
        delta = self.NeuralNetwork.update(X, delta, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                                  epsilon=epsilon, lambda_=lambda_, reg=reg)
        delta = self.LayerNorm_Layer1.update(X, delta, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                                       epsilon=epsilon, l2_lambda=lambda_, reg=reg)
        delta = self.Attention_Layer.update(X, delta, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                            epsilon=epsilon)

        return delta
