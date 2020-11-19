import numpy as np
from numpy.random import randn

class RNN:
    def __init__(self, input_size, output_size, hidden_dim=100):

        self.Whh = randn(hidden_dim, hidden_dim) / 1000
        self.Wxh = randn(hidden_dim, input_size) / 1000
        self.Why = randn(output_size, hidden_dim) / 1000

        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_size, 1))

    def forward_pass(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))

        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
        y = self.Why @ h + self.by
        return y, h