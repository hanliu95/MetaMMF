import torch
import torch.nn as nn


class MetaLayer(nn.Module):
    def __init__(self, s_dim, in_dim, out_dim):
        super(MetaLayer, self).__init__()
        self.s_dim = s_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.T_1 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(self.s_dim, self.in_dim * self.out_dim)))
        self.M_1 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(self.in_dim, self.out_dim)))

    def forward(self, signal, feature):
        w_1 = torch.reshape(torch.matmul(signal, self.T_1), (-1, self.in_dim, self.out_dim))
        feature = torch.reshape(feature, (-1, 1, self.in_dim))
        x = torch.bmm(feature, (w_1 + self.M_1))
        x = torch.reshape(x, (-1, self.out_dim))
        return x

