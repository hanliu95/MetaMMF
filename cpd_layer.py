import torch
import torch.nn as nn


class CPDLayer(nn.Module):
    def __init__(self, R, s_dim, in_dim, out_dim):
        super(CPDLayer, self).__init__()
        self.R = R
        self.s_dim = s_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(self.in_dim, self.R)))
        self.B = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(self.R, self.out_dim)))
        self.C = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(self.R * self.s_dim, 1)))
        self.M_1 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(self.in_dim, self.out_dim)))

    def forward(self, signal, feature):
        AB = torch.reshape(torch.matmul(self.A, self.B), (1, -1))
        ABC = torch.reshape(torch.matmul(self.C, AB), (self.R, self.s_dim, -1))
        T_1 = torch.sum(ABC, dim=0, keepdim=False)
        w_1 = torch.reshape(torch.matmul(signal, T_1), (-1, self.in_dim, self.out_dim))
        feature = torch.reshape(feature, (-1, 1, self.in_dim))
        x = torch.bmm(feature, (w_1 + self.M_1))
        x = torch.reshape(x, (-1, self.out_dim))
        return x
