import math
import numpy as np
import pickle
import torch
from Fusion_model import Net
import torch.utils.data as D

f_1 = open('./pro_data/movielens_load.para', 'rb')
para_load = pickle.load(f_1)
user_num = para_load['user_num']  # total number of users
item_num = para_load['item_num']  # total number of items

train_matrix = para_load['train_matrix']
train_matrix.data = np.array(train_matrix.data, dtype=np.int8)
train_matrix = train_matrix.toarray()  # the 0-1 matrix of training set
test_matrix = para_load['test_matrix']
test_matrix.data = np.array(test_matrix.data, dtype=np.int8)
test_matrix = test_matrix.toarray()  # the 0-1 matrix of testing set

x = np.load('./output/movielens_emb.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


item_ids = np.array(list(range(item_num)))

user_matrix = x[: user_num, :]
item_matrix = x[user_num: user_num + item_num, :]

P = 0
R = 0
HR = 0
NDCG = 0
eva_size = 10


def IDCG(num):
    if num == 0:
        return 1
    idcg = 0
    for i in list(range(num)):
        idcg += 1/math.log(i+2, 2)
    return idcg


def descend_sort(array):
    return -np.sort(-array)


null_user = 0

for user_id, row in enumerate(test_matrix):
    if row.sum() == 0:
        null_user += 1
        continue
    can_item_ids = item_ids[~np.array(train_matrix[user_id], dtype=bool)]  # the id list of test items
    I = item_matrix[can_item_ids, :]
    u = user_matrix[user_id, :]
    inner_pro = np.matmul(u, I.T).reshape(-1)
    sort_index = np.argsort(-inner_pro)
    hit_num = 0
    dcg = 0
    for i, item_id in enumerate(can_item_ids[sort_index][0:eva_size]):
        if row[item_id] > 0:
            hit_num = hit_num + 1
            dcg = dcg + 1 / math.log(i + 2, 2)
    P += hit_num / eva_size
    R += hit_num / np.sum(row)
    HR += hit_num
    NDCG += dcg / IDCG(np.sum(descend_sort(row)[0:eva_size]))

P = P/(user_num - null_user)
R = R/(user_num - null_user)
HR = HR/np.sum(test_matrix)
NDCG = NDCG/(user_num - null_user)
print('P@%d: %.4f; R@%d: %.4f; HR@%d: %.4f; NDCG@%d: %.4f' % (eva_size, P, eva_size, R, eva_size, HR, eva_size, NDCG))
