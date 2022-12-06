import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from Fusion_model import Net
import numpy as np


class GCN(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.conv1 = SAGEConv(128, 128)
        self.conv2 = SAGEConv(128, 64)
        self.fus = Net()
        self.USER_NUM = 55485
        self.ITEM_NUM = 5986
        self.device = device
        self.user = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(self.USER_NUM, 128)))

    def forward(self, video, audio, title, edge_index):
        tensor_list = [self.user]
        for batch_it in np.array_split(np.array(list(range(self.ITEM_NUM))), indices_or_sections=1):
            video_it = torch.from_numpy(video[batch_it, :]).float().to(self.device)
            audio_it = torch.from_numpy(audio[batch_it, :]).float().to(self.device)
            title_it = torch.from_numpy(title[batch_it, :]).float().to(self.device)
            md_it = self.fus(video=video_it, audio=audio_it, title=title_it, item_id=batch_it)
            tensor_list.append(md_it)

        x = torch.cat(tensor_list, dim=0)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)

        x = F.dropout(x, p=0.2, training=self.training)

        return x
