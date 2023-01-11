import pickle
import torch
from torch import nn
import math
import numpy as np
import torch.utils.data as D
import torch.nn.functional as F
from Fusion_model import Net
from GCN_model import GCN
import scipy.sparse as sparse
from torch_geometric.data import Data

f_para = open('./pro_data/movielens_load.para', 'rb')
para_load = pickle.load(f_para)
user_num = para_load['user_num']  # total number of users
item_num = para_load['item_num']  # total number of items
train_ui = para_load['train_ui']
print('total number of users is ', user_num)
print('total number of items is ', item_num)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

video_feature_lookup = np.load('./pro_feature/movielens_v_64.npy')  # size=(item_num, 128)
audio_feature_lookup = np.load('./pro_feature/movielens_a_64.npy')
title_feature_lookup = np.load('./pro_feature/movielens_t_32.npy')

batch_size = 3000
step_threshold = 500
epoch_max = 60
data_block = 1

train_i = torch.empty(0).long()
train_j = torch.empty(0).long()
train_m = torch.empty(0).long()
for b_i in list(range(data_block)):
    triple_para = pickle.load(open('./pro_triple/movielens_triple_' + str(b_i) + '.para', 'rb'))
    train_i = torch.cat((train_i, torch.tensor(triple_para['train_i'])))  # 1-D tensor of user node ID
    train_j = torch.cat((train_j, torch.tensor(triple_para['train_j'])))  # 1-D tensor of pos item node ID
    train_m = torch.cat((train_m, torch.tensor(triple_para['train_m'])))  # 1-D tensor of neg item node ID

train_dataset = D.TensorDataset(train_i, train_j, train_m)
train_loader = D.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


gcn = GCN(device).to(device)
optimizer = torch.optim.Adam([{'params': gcn.parameters()}], lr=1e-3, weight_decay=1e-6)

train_ui = train_ui + [0, user_num]
edge_index = np.concatenate((train_ui, train_ui[:, [1, 0]]), axis=0)
edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_index = edge_index.t().contiguous().to(device)

gcn.train()

for epoch in range(epoch_max):
    running_loss = 0.0
    for step, (batch_i, batch_j, batch_m) in enumerate(train_loader):
        optimizer.zero_grad()

        out = gcn(video_feature_lookup, audio_feature_lookup, title_feature_lookup, edge_index)

        embedding_i = out[batch_i.numpy(), :]
        embedding_j = out[batch_j.numpy() + user_num, :]
        embedding_m = out[batch_m.numpy() + user_num, :]

        predict_ij = torch.sum(torch.mul(embedding_i, embedding_j), dim=1)  # 1-D
        predict_im = torch.sum(torch.mul(embedding_i, embedding_m), dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(predict_ij - predict_im) + 1e-10))
        loss = bpr_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % step_threshold == (step_threshold - 1):
            print('[%d, %d] loss: %.5f' % (epoch + 1, step + 1, running_loss / step_threshold))
            running_loss = 0.0

gcn.eval()

print('model training finished...')

with torch.no_grad():
    out = out = gcn(video_feature_lookup, audio_feature_lookup, title_feature_lookup, edge_index)
    out = out.cpu().numpy()
    np.save('./output/movielens_emb.npy', out)
    print('model training finished...')



