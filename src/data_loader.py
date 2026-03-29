import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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

        # user / item indices 1~n_user / 1~n_item
        self.User_loader = DataLoader(Loader_list(torch.arange(self.n_users).to(device)), batch_size=u_batch_size,
                                      shuffle=True, num_workers=0)
        self.Item_loader = DataLoader(Loader_list(torch.arange(self.n_items).to(device)), batch_size=i_batch_size,
                                      shuffle=True, num_workers=0)

    def get_batch(self, batch_user: Tensor, batch_item: Tensor, device='cuda'):
        """
        # Get users, items, and ratings corresponding to user and item batches
        """
        index_row = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())  # check if elements are in array
        index_col = np.isin(self.mat._indices()[1].cpu().numpy(), batch_item.cpu().numpy())

        temp = np.where(index_row * index_col)
        index = torch.tensor(temp[0]).to(device)  # element coordinates tuple
        return self.mat._indices()[0][index], self.mat._indices()[1][index], self.mat._values()[index]


class User:
    def __init__(self, mat_position: Tensor, mat_rating, u_batch_size=100, device='cuda'):
        self.mat_position = mat_position
        self.mat_rating = mat_rating
        self.n_users, self.n_items = self.mat_position.shape
        self.User_loader = DataLoader(Loader_list(torch.arange(self.n_users).to(device)), batch_size=u_batch_size,
                                      shuffle=True, num_workers=0)

    def get_batch(self, batch_user: Tensor, device='cuda'):
        index = np.isin(self.mat_position._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index = torch.tensor(np.where(index)[0]).to(device)
        # index = (self.mat_pisition._indices()[0][..., None] == batch_user[None, ...]).any(-1)
        # index = torch.where(index_row)[0]
        return self.mat_position._indices()[0][index], self.mat_position._indices()[1][index], \
            self.mat_position._values()[index], self.mat_rating._values()[index]


class EnvInteractions(Dataset):

    def __init__(self, mat: Tensor, envs: Tensor, weight):
        self.mat = mat
        self.n_users, self.n_items = self.mat.shape
        self.envs = envs
        self.weight = weight

    def __getitem__(self, index):
        row = self.mat._indices()[0][index]
        col = self.mat._indices()[1][index]
        val = self.mat._values()[index]
        env = self.envs[index]
        w = self.weight[index]
        # return torch.tensor(int(row)), torch.tensor(int(col)), torch.tensor(val)
        return row, col, val, env, w

    def __len__(self):
        return self.mat._nnz()


class Interactions(Dataset):

    def __init__(self, mat: Tensor):
        self.mat = mat
        self.n_users, self.n_items = self.mat.shape

    def __getitem__(self, index):
        row = self.mat._indices()[0][index]
        col = self.mat._indices()[1][index]
        val = self.mat._values()[index]
        return row, col, val

    def __len__(self):
        return self.mat._nnz()

class User_With_Sep:
    """
    # Data extractor distinguishing positive and negative samples
    """

    def __init__(self, mat: Tensor, u_batch_size=100, device='cuda'):
        self.mat = mat
        self.device = device
        self.n_users, self.n_items = self.mat.shape

        user_data = Loader_list(torch.arange(self.n_users).to(device))

        self.User_Loader = DataLoader(user_data, batch_size=u_batch_size,
                                      shuffle=True, num_workers=0)

    def get_batch(self, batch_user: Tensor):
        device = self.device
        index_user = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index_val = np.isin(self.mat._values(), 1)
        index_pos = index_user * index_val

        index_user_n = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index_val_n = np.isin(self.mat._values(), -1)
        index_neg = index_user_n * index_val_n

        data_pos = (self.mat._indices()[0][index_pos], self.mat._indices()[1][index_pos], self.mat._values()[index_pos])
        data_neg = (self.mat._indices()[0][index_neg], self.mat._indices()[1][index_neg], self.mat._values()[index_neg])

        return data_pos, data_neg


class User_Single:
    """
    # Extract corresponding positive and negative samples based on user
    """

    def __init__(self, mat: Tensor, u_batch_size=1, device='cuda'):
        self.mat = mat
        self.device = device
        self.n_users, self.n_items = self.mat.shape

        user_data = Loader_list(torch.arange(self.n_users).to(device))
        self.item_popularity = torch.zeros(self.n_items).to(device)
        # for i in torch.arange(self.n_items):
        #     index = np.isin(self.mat._indices()[1].cpu().numpy(), i.cpu().numpy())
        #     self.item_popularity[i] = index.sum()
        # print("Item popularity statistics completed")
        self.User_Loader = DataLoader(user_data, batch_size=u_batch_size,
                                      shuffle=True, num_workers=0)

    def get_batch(self, batch_user: Tensor):
        device = self.device
        index_user = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index_val = np.isin(self.mat._values().cpu().numpy(), 1)
        index_pos = index_user * index_val

        index_user_n = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index_val_n = np.isin(self.mat._values().cpu().numpy(), -1)
        index_neg = index_user_n * index_val_n

        data_pos = (self.mat._indices()[0][index_pos], self.mat._indices()[1][index_pos], self.mat._values()[index_pos],
                    self.item_popularity[self.mat._indices()[1][index_pos]])
        data_neg = (self.mat._indices()[0][index_neg], self.mat._indices()[1][index_neg], self.mat._values()[index_neg],
                    self.item_popularity[self.mat._indices()[1][index_neg]])

        return data_pos, data_neg


class Block_Single:
    def __init__(self, mat: Tensor, u_batch_size=1000, i_batch_size=1000, device='cuda'):
        self.mat = mat
        self.n_users, self.n_items = self.mat.shape
        self.device = device

        # user / item indices 1~n_user / 1~n_item
        self.User_loader = DataLoader(Loader_list(torch.arange(self.n_users).to(device)), batch_size=u_batch_size,
                                      shuffle=True, num_workers=0)
        self.Item_loader = DataLoader(Loader_list(torch.arange(self.n_items).to(device)), batch_size=i_batch_size,
                                      shuffle=True, num_workers=0)

        pop_item = {}
        for i in torch.arange(self.n_items):
            index = np.isin(self.mat._indices()[1].cpu().numpy(), i.cpu().numpy())
            pop_item[i] = index.sum()
        print("Item popularity statistics completed")

        pop_user = {}
        for u in torch.arange(self.n_users):
            index = np.isin(self.mat._indices()[0].cpu().numpy(), u.cpu().numpy())
            pop_user[u] = index.sum()

        sorted_pop_user = list(set(list(pop_user.values())))
        sorted_pop_item = list(set(list(pop_item.values())))
        sorted_pop_item.sort()
        sorted_pop_user.sort()

        user_idx = {}
        item_idx = {}
        for i, item in enumerate(sorted_pop_user):
            user_idx[item] = i  #
        for i, item in enumerate(sorted_pop_item):
            item_idx[item] = i

        self.user_pop_idx = torch.zeros(self.n_users, dtype=torch.int).to(device)
        self.item_pop_idx = torch.zeros(self.n_items, dtype=torch.int).to(device)
        for key, value in pop_user.items():
            self.user_pop_idx[key] = user_idx[value]  #
        for key, value in pop_item.items():
            self.item_pop_idx[key] = item_idx[value]  #

        user_pop_max = max(self.user_pop_idx)
        item_pop_max = max(self.item_pop_idx)

        self.user_pop_max = user_pop_max
        self.item_pop_max = item_pop_max


    def get_batch_a(self, batch_user: Tensor, batch_item: Tensor):
        """
        # Get corresponding positive and negative samples
        """
        device = self.device

        ####  Positive samples info
        index_user = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index_col = np.isin(self.mat._indices()[1].cpu().numpy(), batch_item.cpu().numpy())
        index_val = np.isin(self.mat._values().cpu().numpy(), 1)
        index_pos = index_user * index_val * index_col

        user_pos = self.mat._indices()[0][index_pos]
        item_pos = self.mat._indices()[1][index_pos]
        y_pos = self.mat._values()[index_pos]
        item_pop_pos = self.item_pop_idx[item_pos]
        user_pop_pos = self.user_pop_idx[user_pos]

        ####  Negative samples info
        index_user_n = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index_val_n = np.isin(self.mat._values().cpu().numpy(), -1)
        index_neg = index_user_n * index_val_n * index_col

        user_neg = self.mat._indices()[0][index_neg]
        item_neg = self.mat._indices()[1][index_neg]
        y_neg = self.mat._values()[index_neg]
        item_pop_neg = self.item_pop_idx[item_neg]
        user_pop_neg = self.user_pop_idx[user_neg]

        data_pos = (user_pos, item_pos, y_pos, user_pop_pos, item_pop_pos)
        data_neg = (user_neg, item_neg, y_neg, user_pop_neg, item_pop_neg)

        return data_pos, data_neg


    def get_batch(self, batch_user: Tensor, batch_item: Tensor):
        """
        # Get users, items, and ratings corresponding to user and item batches
        """
        device = self.device
        index_row = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())  # check if elements are in array
        index_col = np.isin(self.mat._indices()[1].cpu().numpy(), batch_item.cpu().numpy())

        temp = np.where(index_row * index_col)
        index = torch.tensor(temp[0]).to(device)  # element coordinates tuple
        return self.mat._indices()[0][index], self.mat._indices()[1][index], self.mat._values()[index]


class FusionEnvInteractions():

    def __init__(self, mat: Tensor, envs: Tensor, u_batch_size=6000, i_batch_size=500):
        self.mat = mat.cuda()
        self.n_users, self.n_items = self.mat.shape
        self.envs = envs.cuda()
        self.User_loader = DataLoader(Loader_list(torch.arange(self.n_users).cuda()), batch_size=u_batch_size,
                                      shuffle=True, num_workers=0)
        self.Item_loader = DataLoader(Loader_list(torch.arange(self.n_items).cuda()), batch_size=i_batch_size,
                                      shuffle=True, num_workers=0)

    def getbatch(self, batch_user: Tensor, batch_item: Tensor):
        index_user = np.isin(self.mat._indices()[0].cpu().numpy(), batch_user.cpu().numpy())
        index_col = np.isin(self.mat._indices()[1].cpu().numpy(), batch_item.cpu().numpy())
        index = index_user * index_col
        index = torch.tensor(np.where(index)[0]).cuda()

        row = self.mat._indices()[0][index]
        col = self.mat._indices()[1][index]
        val = self.mat._values()[index]
        env = self.envs[index]

        return row, col, val, env


def var(a:pd.DataFrame)->Tensor:
    return torch.from_numpy(a.values).cuda()


class FusionEnvInteractionsNew():

    def __init__(self, mat: pd.DataFrame, envs: Tensor, weight:Tensor, u_batch_size=6000, i_batch_size=500):
        self.users = var(mat['uid'])
        self.items = var(mat['iid'])
        self.scores = var(mat['score'])
        self.item_count = var(mat['item_count'])
        self.user_count = var(mat['user_count'])
        self.user_score_mean = var(mat['user_score_mean'])
        self.item_score_mean = var(mat['item_score_mean'])
        self.user_score_std	= var(mat['user_score_std'])
        self.item_score_std = var(mat['item_score_std'])
        self.n_users = torch.unique(self.users).shape[0]
        self.n_items = torch.unique(self.items).shape[0]
        self.envs = envs.cuda()
        self.weight = weight.cuda()
        self.User_loader = DataLoader(Loader_list(torch.arange(self.n_users).cuda()), batch_size=u_batch_size,
                                      shuffle=True, num_workers=0)
        self.Item_loader = DataLoader(Loader_list(torch.arange(self.n_items).cuda()), batch_size=i_batch_size,
                                      shuffle=True, num_workers=0)
        pop_item = {}
        for i in torch.arange(self.n_items):
            index = np.isin(self.items.values, i.cpu().numpy())
            pop_item[i] = index.sum()

        pop_user = {}
        for u in torch.arange(self.n_users):
            index = np.isin(self.users.values, u.cpu().numpy())
            pop_user[u] = index.sum()

        sorted_pop_user = list(set(list(pop_user.values())))
        sorted_pop_item = list(set(list(pop_item.values())))
        sorted_pop_item.sort()
        sorted_pop_user.sort()

        user_idx = {}
        item_idx = {}
        for i, item in enumerate(sorted_pop_user):
            user_idx[item] = i 
        for i, item in enumerate(sorted_pop_item):
            item_idx[item] = i

        self.user_pop_idx = torch.zeros(self.n_users, dtype=torch.int).cuda()
        self.item_pop_idx = torch.zeros(self.n_items, dtype=torch.int).cuda()
        for key, value in pop_user.items():
            self.user_pop_idx[key] = user_idx[value] 
        for key, value in pop_item.items():
            self.item_pop_idx[key] = item_idx[value] 

        user_pop_max = max(self.user_pop_idx)
        item_pop_max = max(self.item_pop_idx)

        self.user_pop_max = user_pop_max
        self.item_pop_max = item_pop_max
        # print(user_pop_max, item_pop_max)
        # print("物品流行度统计完成")

    
    def getbatch(self, batch_user: Tensor, batch_item: Tensor):
        index_user = np.isin(self.users.cpu().numpy(), batch_user.cpu().numpy())
        index_col = np.isin(self.items.cpu().numpy(), batch_item.cpu().numpy())
        index = index_user * index_col
        index = torch.tensor(np.where(index)[0]).cuda()

        row = self.users[index]
        col = self.items[index]
        val = self.scores[index]

        item_count = self.item_count[index]
        user_count= self.user_count[index]
        user_score_mean	= self.user_score_mean[index]
        item_score_mean	= self.item_score_mean[index]
        user_score_std = self.user_score_std[index]
        item_score_std = self.item_score_std[index]

        env = self.envs[index]
        w = self.weight[index]
        users_pop = self.user_pop_idx[row]
        items_pop = self.item_pop_idx[col]
        return row, col, val, item_count, user_count, user_score_mean, \
               item_score_mean, user_score_std, item_score_std, env, w, users_pop, items_pop


class PureInteractionsNew():

    def __init__(self, mat: pd.DataFrame, u_batch_size=6000, i_batch_size=500):
        self.users = var(mat['uid'])
        self.items = var(mat['iid'])
        self.scores = var(mat['score'])
        self.item_count = var(mat['item_count'])
        self.user_count = var(mat['user_count'])
        self.user_score_mean = var(mat['user_score_mean'])
        self.item_score_mean = var(mat['item_score_mean'])
        self.user_score_std	= var(mat['user_score_std'])
        self.item_score_std = var(mat['item_score_std'])
        self.n_users = torch.unique(self.users).shape[0]
        self.n_items = torch.unique(self.items).shape[0]

        self.User_loader = DataLoader(Loader_list(torch.arange(self.n_users).cuda()), batch_size=u_batch_size,
                                      shuffle=True, num_workers=0)
        self.Item_loader = DataLoader(Loader_list(torch.arange(self.n_items).cuda()), batch_size=i_batch_size,
                                      shuffle=True, num_workers=0)
        pop_item = {}
        for i in torch.arange(self.n_items):
            index = np.isin(self.items.values, i.cpu().numpy())
            pop_item[i] = index.sum()

        pop_user = {}
        for u in torch.arange(self.n_users):
            index = np.isin(self.users.values, u.cpu().numpy())
            pop_user[u] = index.sum()

        sorted_pop_user = list(set(list(pop_user.values())))
        sorted_pop_item = list(set(list(pop_item.values())))
        sorted_pop_item.sort()
        sorted_pop_user.sort()

        user_idx = {}
        item_idx = {}
        for i, item in enumerate(sorted_pop_user):
            user_idx[item] = i 
        for i, item in enumerate(sorted_pop_item):
            item_idx[item] = i

        self.user_pop_idx = torch.zeros(self.n_users, dtype=torch.int).cuda()
        self.item_pop_idx = torch.zeros(self.n_items, dtype=torch.int).cuda()
        for key, value in pop_user.items():
            self.user_pop_idx[key] = user_idx[value] 
        for key, value in pop_item.items():
            self.item_pop_idx[key] = item_idx[value] 

        user_pop_max = max(self.user_pop_idx)
        item_pop_max = max(self.item_pop_idx)

        self.user_pop_max = user_pop_max
        self.item_pop_max = item_pop_max
        # print(user_pop_max, item_pop_max)
        # print("物品流行度统计完成")

    
    def getbatch(self, batch_user: Tensor, batch_item: Tensor):
        index_user = np.isin(self.users.cpu().numpy(), batch_user.cpu().numpy())
        index_col = np.isin(self.items.cpu().numpy(), batch_item.cpu().numpy())
        index = index_user * index_col
        index = torch.tensor(np.where(index)[0]).cuda()

        row = self.users[index]
        col = self.items[index]
        val = self.scores[index]

        item_count = self.item_count[index]
        user_count= self.user_count[index]
        user_score_mean	= self.user_score_mean[index]
        item_score_mean	= self.item_score_mean[index]
        user_score_std = self.user_score_std[index]
        item_score_std = self.item_score_std[index]

        users_pop = self.user_pop_idx[row]
        items_pop = self.item_pop_idx[col]
        return row, col, val, item_count, user_count, user_score_mean, \
               item_score_mean, user_score_std, item_score_std, users_pop, items_pop

