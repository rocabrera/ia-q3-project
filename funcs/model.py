import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        # first_layer
        out = self.l1(X)
        out = self.relu(out)
        # second_layer
        out = self.l2(out)
        return out