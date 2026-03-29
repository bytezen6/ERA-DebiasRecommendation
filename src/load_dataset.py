import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset


def seed_randomly_split(df, ratio, split_seed, shape):
    """
    Split based on a deterministic seed randomly
    """
    # Set the random seed for splitting
    np.random.seed(split_seed)

    # Randomly shuffle the data
    rows, cols, rating = df['uid'], df['iid'], df['rating']
    num_nonzeros = len(rows)
    permute_indices = np.random.permutation(num_nonzeros)
    rows, cols, rating = rows[permute_indices], cols[permute_indices], rating[permute_indices]

    # Convert to train/valid/test matrix
    idx = [int(ratio[0] * num_nonzeros), int(ratio[0] * num_nonzeros) + int(ratio[1] * num_nonzeros)]

    train = sp.csr_matrix((rating[:idx[0]], (rows[:idx[0]], cols[:idx[0]])), shape=shape, dtype='float32')

    valid = sp.csr_matrix((rating[idx[0]:idx[1]], (rows[idx[0]:idx[1]], cols[idx[0]:idx[1]])), shape=shape,
                          dtype='float32')

    test = sp.csr_matrix((rating[idx[1]:], (rows[idx[1]:], cols[idx[1]:])), shape=shape, dtype='float32')

    return train, valid, test


def load_dataset(data_name='yahooR3', type='explicit', unif_ratio=0.05, seed=0, threshold=4):
    """
    flag: whether to use interpolation model, TRUE means using interpolation model
    """
    if type not in ['explicit', 'implicit', 'list']:
        print('--------illegal type, please reset legal type.---------')
        return

    path = './datasets/' + data_name
    # if "uai" in data_name:
    #     path = './datasets/' + data_name + '/32/'
    if data_name == 'simulation':
        user_df = pd.read_csv(path + '/user.txt', sep=',', header=0, names=['uid', 'iid', 'position', 'rating'])
    else :
        user_df = pd.read_csv(path + '/user.txt', sep=',', header=0, names=['uid', 'iid', 'rating'])
        user_df['uid'] = user_df['uid'].astype('int64')
        user_df['iid'] = user_df['iid'].astype('int64')
    random_df = pd.read_csv(path + '/random.txt', sep=',', header=0, names=['uid', 'iid', 'rating'])


    # binirize the rating to -1/1
    neg_score = 0 if type == 'implicit' else 1
    user_df['rating'].loc[(user_df['rating']) < threshold] = neg_score
    user_df['rating'].loc[user_df['rating'] >= threshold] = 1

    random_df['rating'].loc[random_df['rating'] < threshold] = neg_score
    random_df['rating'].loc[random_df['rating'] >= threshold] = 1
    
    m, n = max(user_df['uid'].max(), random_df["uid"].max()) + 1, max(user_df['iid'].max(), random_df["iid"].max()) + 1
    # 划分比例 train/valid/test matrix
    ratio = (unif_ratio, 0.05, 1 - unif_ratio)
    print(random_df.shape, user_df.shape)
    unif_train, validation, test = seed_randomly_split(df=random_df, ratio=ratio,
                                                       split_seed=seed, shape=(m, n))
    if type == 'list':
        train_pos = sp.csr_matrix((user_df['position'], (user_df['uid'], user_df['iid'])), shape=(m, n), dtype='int64')
        train_rating = sp.csr_matrix((user_df['rating'], (user_df['uid'], user_df['iid'])), shape=(m, n),
                                     dtype='float32')
        train = {}
        train['position'] = sparse_mx_to_torch_sparse_tensor(train_pos)
        train['rating'] = sparse_mx_to_torch_sparse_tensor(train_rating)
    else:
        train = sp.csr_matrix((user_df['rating'], (user_df['uid'], user_df['iid'])), shape=(m, n), dtype='float32')
        train = sparse_mx_to_torch_sparse_tensor(train).cuda()

    unif_train = sparse_mx_to_torch_sparse_tensor(unif_train).cuda()
    validation = sparse_mx_to_torch_sparse_tensor(validation).cuda()
    test = sparse_mx_to_torch_sparse_tensor(test).cuda()
    # train: 真正 的 有偏的数据集
    # unif_train vaildation test 来自于random数据
    return train, unif_train, validation, test


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_dataset_specific(data_name='yahooR3', type='explicit', unif_ratio=0.05, seed=0, threshold=4):
    """
    flag: whether to use interpolation model, TRUE means using interpolation model
    """
    if type not in ['explicit', 'implicit', 'list']:
        print('--------illegal type, please reset legal type.---------')
        return

    path = './datasets/' + data_name

    if data_name == 'simulation':
        user_df = pd.read_csv(path + '/user.txt', sep=',', header=0, names=['uid', 'iid', 'position', 'rating'])
    else :
        user_df = pd.read_csv(path + '/user.txt', sep=',', header=0, names=['uid', 'iid', 'rating'])
        user_df['uid'] = user_df['uid'].astype('int64')
        user_df['iid'] = user_df['iid'].astype('int64')
    random_df = pd.read_csv(path + '/random.txt', sep=',', header=0, names=['uid', 'iid', 'rating'])


    # binirize the rating to -1/1
    neg_score = 0 if type == 'implicit' else 1
    user_df['rating'].loc[(user_df['rating']) < threshold] = neg_score
    user_df['rating'].loc[user_df['rating'] >= threshold] = 1

    random_df['rating'].loc[random_df['rating'] < threshold] = neg_score
    random_df['rating'].loc[random_df['rating'] >= threshold] = 1
    
    m, n = max(user_df['uid'].max(), random_df["uid"].max()) + 1, max(user_df['iid'].max(), random_df["iid"].max()) + 1
    # 划分比例 train/valid/test matrix
    ratio = (unif_ratio, 0.05, 1 - unif_ratio - 0.05 )
    print(random_df.shape, user_df.shape)
    unif_train, validation, test = seed_randomly_split(df=random_df, ratio=ratio,
                                                       split_seed=seed, shape=(m, n))
    if type == 'list':
        train_pos = sp.csr_matrix((user_df['position'], (user_df['uid'], user_df['iid'])), shape=(m, n), dtype='int64')
        train_rating = sp.csr_matrix((user_df['rating'], (user_df['uid'], user_df['iid'])), shape=(m, n),
                                     dtype='float32')
        train = {}
        train['position'] = sparse_mx_to_torch_sparse_tensor(train_pos)
        train['rating'] = sparse_mx_to_torch_sparse_tensor(train_rating)
    else:
        train = sp.csr_matrix((user_df['rating'], (user_df['uid'], user_df['iid'])), shape=(m, n), dtype='float32')
        # train = sparse_mx_to_torch_sparse_tensor(train).cuda()

    # unif_train = sparse_mx_to_torch_sparse_tensor(unif_train).cuda()
    
    train, unif_train = split_train_into_train_and_unif(user_df, m, n)
    
    train = sparse_mx_to_torch_sparse_tensor(train).cuda()
    unif_train = sparse_mx_to_torch_sparse_tensor(unif_train).cuda()
    validation = sparse_mx_to_torch_sparse_tensor(validation).cuda()
    test = sparse_mx_to_torch_sparse_tensor(test).cuda()
    # train: 真正 的 有偏的数据集
    # unif_train vaildation test 来自于random数据
    return train, unif_train, validation, test



def split_train_into_train_and_unif(df:pd.DataFrame, m:int=None, n:int=None):
    """
    Split based on a deterministic seed randomly
    """
    unif_user = pd.DataFrame() 
    unif_user_item = pd.DataFrame()
    
    for _, g in df.groupby('uid'):
        # print(g)
        k = min(len(g), 5) 
        unif_user = pd.concat([unif_user, g.sample(n=k)])
    
    for _, g in unif_user.groupby("iid"):
        k = min(len(g), 5) 
        unif_user_item = pd.concat([unif_user_item, g.sample(n=k)])
    
    train_df = df.drop(unif_user_item.index)
    unif_train_df = unif_user_item 

    print("train_df", train_df.shape)
    print("unif_train_df", unif_train_df.shape)

    train = sp.csr_matrix((train_df['rating'], (train_df['uid'], train_df['iid'])), shape=(m, n), dtype='float32')
    unif_train = sp.csr_matrix((unif_train_df['rating'], (unif_train_df['uid'], unif_train_df['iid'])), shape=(m, n), dtype='float32')

    return train, unif_train    