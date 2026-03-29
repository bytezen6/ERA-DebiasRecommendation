import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor


class Loader_list(Dataset):
    def __init__(self, lst):
        self.lst = lst

    def __getitem__(self, index):
        return self.lst[index]

    def __len__(self):
        return len(self.lst)


class Block:
    def __init__(self, mat: Tensor, u_batch_size=1000, i_batch_size=1000, device='cuda'):
        self.mat = mat
        self.n_users, self.n_items = self.mat.shape

        # 所有user / item 的序号 1~n_user / 1~n_item
        self.User_loader = DataLoader(Loader_list(torch.arange(self.n_users).to(device)), batch_size=u_batch_size,
                                      shuffle=True, num_workers=0)
        self.Item_loader = DataLoader(Loader_list(torch.arange(self.n_items).to(device)), batch_size=i_batch_size,
                                      shuffle=True, num_workers=0)

    def get_batch(self, batch_user: Tensor, batch_item: Tensor, device='cuda'):
        index_row = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())  # B中元素是否在A中
        index_col = np.isin(self.mat._indices()[1].cpu().numpy(), batch_item.cpu().numpy())

        temp = np.where(index_row * index_col)

        index = Tensor(temp[0]).to(device).long()  # 元素坐标 形成的元组
        # index_row = (mat._indices()[0][..., None] == batch_user[None, ...]).any(-1)
        # index_col = (mat._indices()[1][..., None] == batch_item[None, ...]).any(-1)
        # index = torch.where(index_row * index_col)[0]
        return self.mat._indices()[0][index], self.mat._indices()[1][index], self.mat._values()[index]
        # 返回user item y_train

    def get_batch_withneg(self, batch_user: Tensor, batch_item: Tensor, device='cuda'):
        index_row = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())  # B中元素是否在A中
        index_col = np.isin(self.mat._indices()[1].cpu().numpy(), batch_item.cpu().numpy())

        temp = np.where(index_row * index_col)  # True * True
        neg_items = np.where(index_col == False)

        index = Tensor(temp[0]).to(device)
        num = index.shape[0]
        li = random.sample(neg_items[0].tolist(), num)
        index_neg = Tensor(li).to(device)

        return self.mat._indices()[0][index], self.mat._indices()[1][index], \
            self.mat._values()[index], self.mat._indices()[1][index_neg]

class User:
    def __init__(self, mat_position: Tensor, mat_rating, u_batch_size=100, device='cuda'):
        self.mat_position = mat_position
        self.mat_rating = mat_rating
        self.n_users, self.n_items = self.mat_position.shape
        self.User_loader = DataLoader(Loader_list(torch.arange(self.n_users).to(device)), batch_size=u_batch_size,
                                      shuffle=True, num_workers=0)

    def get_batch(self, batch_user: Tensor, device='cuda'):
        index = np.isin(self.mat_position._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index = Tensor(np.where(index)[0]).to(device)
        # index = (self.mat_pisition._indices()[0][..., None] == batch_user[None, ...]).any(-1)
        # index = torch.where(index_row)[0]
        return self.mat_position._indices()[0][index], self.mat_position._indices()[1][index], \
        self.mat_position._values()[index], self.mat_rating._values()[index]


class Interactions(Dataset):
    """
    Hold data in the form of an interactions matrix.
    Typical use-case is like a ratings matrix:
    - Users are the rows
    - Items are the columns
    - Elements of the matrix are the ratings given by a user for an item.
    """

    def __init__(self, mat: Tensor):
        self.mat = mat
        self.n_users, self.n_items = self.mat.shape

    def __getitem__(self, index):
        row = self.mat._indices()[0][index]
        col = self.mat._indices()[1][index]
        val = self.mat._values()[index]
        # return Tensor(int(row)), Tensor(int(col)), Tensor(val)
        return row, col, val

    def __len__(self):
        return self.mat._nnz()


"""
建立一个抽取正样本和负样本的数据提取器
"""
class Block_PandN_Coat():
    def __init__(self, mat: Tensor, u_batch_size=1000, i_batch_size=1000, device='cuda'):
        self.mat = mat
        self.n_users, self.n_items = self.mat.shape
        self.device = device

        # 所有user / item 的序号 1~n_user / 1~n_item
        self.User_loader = DataLoader(Loader_list(torch.arange(self.n_users).to(device)), batch_size=u_batch_size,
                                      shuffle=True, num_workers=0)
        self.Item_loader = DataLoader(Loader_list(torch.arange(self.n_items).to(device)), batch_size=i_batch_size,
                                      shuffle=True, num_workers=0)

    def get_batch(self, batch_user: Tensor, batch_item: Tensor):
        device = self.device
        index_row = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())  # B中元素是否在A中
        index_col = np.isin(self.mat._indices()[1].cpu().numpy(), batch_item.cpu().numpy())

        idx = index_row * index_col

        pos = np.isin(self.mat._values().cpu().numpy(), 1)
        neg = np.isin(self.mat._values().cpu().numpy(), -1)

        pos_idx = idx * pos
        neg_idx = idx * neg

        temp = np.where(idx)

        index = Tensor(temp[0]).to(device)  # 元素坐标 形成的元组

        return self.mat._indices()[0][index], self.mat._indices()[1][index], self.mat._values()[index]
