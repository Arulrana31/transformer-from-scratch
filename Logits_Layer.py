import numpy as np
from Functions import function

class Logit_Layer:
    def __init__(self, embed_size, vocab):
        self.embed_size = embed_size
        self.vocab = vocab

        self.w = None
        self.b = None

    def initialize(self):
        self.w = np.random.randn(self.vocab, self.embed_size)
        self.b = np.random.randn(self.vocab, 1) * 0.01

    def compute(self, X, switch=0):
        embed_number = X.shape[1]
        X_ = np.dot(self.w, X) + self.b
        Xf = np.empty_like(X_)
        for i in range(embed_number):
            Xf[:, i:i+1] = function("SoftMax").normal(X_[:, i:i+1])
        if switch == 0:
            return Xf
        else:
            return Xf, X_
    def backprop(self, X, delta):
        """
        Args:
            dZ: gradient of loss wrt output logits, shape (vocab, embed_number)
            X: input embeddings to forward, shape (embed_size, embed_number)

        Returns:
            dX: gradient wrt input embeddings, shape (embed_size, embed_number)
        """
        # Gradient wrt weights
        dW = np.dot(delta, X.T)  # shape (vocab, embed_size)

        # Gradient wrt bias: sum over tokens axis (columns)
        db = np.sum(delta, axis=1, keepdims=True)  # shape (vocab, 1)

        # Gradient wrt input embeddings
        dX = np.dot(self.w.T, delta)  # shape (embed_size, embed_number)

        return dX, dW, db

    def update(self, X, delta, learning_rate=0.01):
        batch_size = len(X)
        dX = [None] * batch_size
        dW = np.zeros_like(self.w)
        db = np.zeros_like(self.b)

        for i in range(batch_size):
            a, b, c = self.backprop(X[i], delta[i])
            dX[i] = a
            dW += b
            db += c

        dW = dW/batch_size
        db = db/batch_size

        self.w -= learning_rate * dW
        self.b -= learning_rate * db

        return dX


X = np.random.randn(10, 4)
X1 = np.random.randn(10, 2)
list = [X, X1]
L = Logit_Layer(embed_size=10, vocab=2)
delta = np.random.randn(2, 4)
delta1 = np.random.randn(2, 2)
dlist = [delta, delta1]
L.initialize()
print(L.compute(X))
k = L.update(list, dlist)
print(L.compute(X))
