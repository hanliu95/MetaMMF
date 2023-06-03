import torch
import torch.nn as nn
from meta_layer import MetaLayer
from cpd_layer import CPDLayer


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.R = 16
        self.s_dim = 5
        self.f_dim = 2276
        self.n_output = 32
        self.ITEM_NUM = 5986

        self.mlp_signal = nn.Sequential(
            nn.Linear(self.f_dim, self.f_dim),
            nn.ReLU(),
            nn.Linear(self.f_dim, self.s_dim)
        )
        self.item_E = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(self.ITEM_NUM, self.n_output)))
        self.layer_1 = CPDLayer(self.R, self.s_dim, self.f_dim, 1024)
        self.layer_2 = CPDLayer(self.R, self.s_dim, 1024, 512)
        self.layer_3 = CPDLayer(self.R, self.s_dim, 512, 256)
        self.layer_4 = CPDLayer(self.R, self.s_dim, 256, self.n_output)
        self.activation = nn.LeakyReLU()

    def forward(self, video, audio, title, item_id):
        feature = torch.cat((video, audio, title), dim=1)
        signal = self.mlp_signal(feature)
        x = self.layer_1(signal, feature)
        x = self.activation(x)
        x = self.layer_2(signal, x)
        x = self.activation(x)
        x = self.layer_3(signal, x)
        x = self.activation(x)
        x = self.layer_4(signal, x)
        x = torch.cat((x, self.item_E[item_id, :]), dim=1)
        return x
