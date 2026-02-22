import torch
import torch.nn as nn



class LiquidTank(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

        self.tau = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, h, dt=0.01):
        dh = (-h + torch.tanh(self.W(h)) + self.U(x)) / self.tau
        h_next = h + dt * dh

        y = self.out(h_next)

        return y, h_next