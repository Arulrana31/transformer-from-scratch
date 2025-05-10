import numpy as np
from Network_Re import Network
from Attention import AttentionLayer
from Layernorm import LayerNorm


class Transformer_Layer:
    def __init__(self, embed_size, heads_number):
        self.embed_size = embed_size
        self.heads_number = heads_number

        self.Attention_Layer = None
        self.LayerNorm_Layer1 = None
        self.LayerNorm_Layer2 = None
        self.NeuralNetwork = None

    def initialize_Transformer(self):
        N = self.embed_size
        self.Attention_Layer = AttentionLayer(N, self.heads_number)
        self.Attention_Layer.initialize_attention_layer()

        self.LayerNorm_Layer1 = LayerNorm()
        self.LayerNorm_Layer2 = LayerNorm()

        self.LayerNorm_Layer1.LayerNorm_initialize(self.embed_size)
        self.LayerNorm_Layer2.LayerNorm_initialize(self.embed_size)

        self.NeuralNetwork = Network(3, [N, 4 * N, N], ["ReLU", "identity"])
        self.NeuralNetwork.initialize()

    def compute_Transformer_Output(self, X):
        value = self.Attention_Layer.compute_attention_layer(X)
        value = self.LayerNorm_Layer1.LayerNorm_compute(value)
        value = self.NeuralNetwork.compute_network(value)
        value = self.LayerNorm_Layer2.LayerNorm_compute(value)

        return value

    def Update(self, X, delta, learning_rate=0.01,
               beta1=0.9,  # Adam: exponential decay rate for 1st moment estimates
               beta2=0.999,  # Adam: exponential decay rate for 2nd moment estimates
               epsilon=1e-8,  # Adam: small constant for numerical stability
               lambda_=1e-4,
               reg="L2", ):
        delta = self.LayerNorm_Layer2.LayerNorm_Update(X, delta, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                                       epsilon=epsilon, l2_lambda=lambda_)
        delta = self.NeuralNetwork.Network_Update(X, delta, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                                  epsilon=epsilon, lambda_=lambda_)
        delta = self.LayerNorm_Layer1.LayerNorm_Update(X, delta, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                                       epsilon=epsilon, l2_lambda=lambda_)
        delta = self.Attention_Layer.Update(X, delta, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                            epsilon=epsilon)

        return delta
