import numpy as np
import random
import math
import scipy.sparse as sparse
import pickle
from sklearn.decomposition import PCA

USER_NUM = 55485
ITEM_NUM = 5986

video_feature_lookup = np.load('./dataset_sample/movielens/v_feat_sample.npy')  # 2048
audio_feature_lookup = np.load('./dataset_sample/movielens/a_feat_sample.npy')  # 128
title_feature_lookup = np.load('./dataset_sample/movielens/t_feat_sample.npy')  # 100

pca_v = PCA(n_components=64)
video_signal_lookup = pca_v.fit_transform(video_feature_lookup)  # size=(item_num, n_components)
pca_a = PCA(n_components=64)
audio_signal_lookup = pca_a.fit_transform(audio_feature_lookup)
pca_t = PCA(n_components=32)
title_signal_lookup = pca_t.fit_transform(title_feature_lookup)

np.save('./pro_feature/movielens_v_64.npy', video_signal_lookup)
np.save('./pro_feature/movielens_a_64.npy', audio_signal_lookup)
np.save('./pro_feature/movielens_t_32.npy', title_signal_lookup)

whole_dict = np.load('./dataset_sample/tiktok/user_item_dict_sample.npy', allow_pickle=True)
whole_dict = whole_dict.item()

train_ui = np.empty(shape=[0, 2], dtype=int)
test_ui = np.empty(shape=[0, 2], dtype=int)
val_ui = np.empty(shape=[0, 2], dtype=int)

for k in range(USER_NUM):
    whole_set = list(whole_dict[k])
    train_set = random.sample(whole_set, round(0.8 * len(whole_set)))
    test_val_set = list(set(whole_set) - set(train_set))
    test_set = random.sample(test_set, math.ceil(0.5 * len(test_set)))
    val_set = list(set(test_val_set) - set(test_set))
    for i in train_set:
        train_ui = np.append(train_ui, [[k, i]], axis=0)
    for j in test_set:
        test_ui = np.append(test_ui, [[k, j]], axis=0)
    for t in val_set:
        val_ui = np.append(val_ui, [[k, t]], axis=0)

train_ui = train_ui - [0, USER_NUM]
test_ui = test_ui - [0, USER_NUM]
val_ui = val_ui - [0, USER_NUM]

print('train_matrix test_matrix started...')
row = train_ui[:, 0]
col = train_ui[:, 1]
data = np.ones_like(row)
train_matrix = sparse.coo_matrix((data, (row, col)), shape=(USER_NUM, ITEM_NUM), dtype=np.int8)
row = test_ui[:, 0]
col = test_ui[:, 1]
data = np.ones_like(row)
test_matrix = sparse.coo_matrix((data, (row, col)), shape=(USER_NUM, ITEM_NUM), dtype=np.int8)
row = val_ui[:, 0]
col = val_ui[:, 1]
data = np.ones_like(row)
val_matrix = sparse.coo_matrix((data, (row, col)), shape=(USER_NUM, ITEM_NUM), dtype=np.int8)
print('train_matrix test_matrix constructed...')

para = {}
para['user_num'] = USER_NUM
para['item_num'] = ITEM_NUM
para['train_matrix'] = train_matrix
para['test_matrix'] = test_matrix
para['val_matrix'] = val_matrix
para['train_ui'] = train_ui
pickle.dump(para, open('./pro_data/tiktok_load.para', 'wb'))
print('data_load finished...')


