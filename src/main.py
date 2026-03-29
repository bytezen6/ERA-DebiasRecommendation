"""
Era: Automatic Debiasing Recommendation System Implementation
Main functions: Implemented recommendation system debiasing method based on meta-learning and adversarial training
Supported datasets: Coat, Yahoo!R3, KuaiRand-1K
"""
import os
import time
import numpy as np
import random
import sys
import itertools
import argparse
import pandas as pd
import copy
sys.path.append('./')

import torch
from torch import Tensor, nn, optim
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
from src.model import dc2, DisBC, GradReverseLayer, zw2
from src.load_dataset import load_dataset_specific as load_dataset
import src.data_loader as dl
import utils.metrics
from utils.early_stop import EarlyStopping, Stop_args


def setup_seed(seed):
    """
    Set random seed to ensure reproducible experiments
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class train_and_eval:
    """
    Train and evaluation class, encapsulates the entire training process
    """
    def __init__(self, train_data, unif_data, test_data, args):
        """
        Initialize the trainer
        Args:
            train_data: Biased training data (sparse tensor)
            unif_data: Uniformly sampled training data (for debiasing)
            test_data: Test data
            args: Parameter dictionary
        """
        self.args = args

        # Load biased training data
        self.train_data = train_data
        self.train_dense = train_data.to_dense()
        self.train_length = train_data._nnz()
        self.train_user:Tensor = train_data._indices()[0]
        self.train_item:Tensor = train_data._indices()[1]
        self.train_y:Tensor = train_data._values()

        # Load uniformly sampled data
        self.unif_user:Tensor = unif_data._indices()[0]
        self.unif_item:Tensor = unif_data._indices()[1]
        self.unif_y:Tensor = unif_data._values()

        # Initialize test data loader
        self.test_loader = dl.DataLoader(dl.Interactions(test_data), batch_size=102400, shuffle=False,
                                         num_workers=0)

        # Get number of users and items
        n_user, n_item = train_data.shape
        self.n_user = n_user
        self.n_item = n_item
        print("train num :{0} test num:{1}".format(train_data._nnz(), test_data._nnz()))

        # Model hyperparameter settings
        emb_dim = 64
        self.embdim = emb_dim

        # Initialize recommendation model
        self.model = DisBC(n_user, n_item, emb_dim).cuda()
        self.optimizer = optim.Adam(self.model.params(), lr=1e-1, weight_decay=1e-6)

        # Initialize weight network (for meta-learning to adjust sample weights)
        self.w = zw2(n_user=n_user, n_item=n_item, n_dim=9).cuda()
        self.w_optim = optim.Adam(self.w.parameters(), lr=1e-1, weight_decay=0)

        # Initialize discriminator (for adversarial training)
        self.dc = dc2(emb_dim, emb_dim).cuda()
        self.dc_optimizer = torch.optim.Adam(self.dc.parameters(), lr=5e-2, weight_decay=1e-6)

        # Initialize early stopping mechanism
        stopping_args = Stop_args(patience=self.args['patience'], max_epochs=self.args['epochs'])
        self.early_stopping = EarlyStopping(self.model, **stopping_args)

        # Initialize TensorBoard log
        self.log_path = './logs/YahooR3/' + time.strftime('%m-%d-%H-%M-%S')
        self.writer = SummaryWriter(self.log_path)

        return


    def train(self):
        """
        Execute a single training epoch
        Returns:
            loss_val: Dictionary containing various loss values
        """
        loss_func = nn.BCELoss(reduction='none').forward
        min_value = 1e-10

        # Generate pseudo-uniform samples (for adversarial training)
        pseudo_unif_index = torch.randint(0, self.n_item*self.n_user, [self.train_y.shape[0]]).cuda()
        pseudo_unif_user = torch.div(pseudo_unif_index, self.n_item, rounding_mode='floor').long()
        pseudo_unif_item = pseudo_unif_index % self.n_item

        # Get embedding representations
        train_z = self.model.get_z(self.train_user, self.train_item)
        pseudo_unif_z = self.model.get_z(pseudo_unif_user, pseudo_unif_item)

        # Train discriminator: distinguish between biased samples and uniform samples
        for _ in range(1):
            self.dc.train()
            train_prob = self.dc.forward(train_z.detach(), False)
            unif_prob = self.dc.forward(pseudo_unif_z.detach(), False)

            # 判别器损失：尽可能正确区分两类样本
            dc_train_loss = torch.mean(torch.log(train_prob + min_value)) + torch.mean(torch.log(1 - unif_prob + min_value))

            self.dc_optimizer.zero_grad()
            dc_train_loss.backward()
            self.dc_optimizer.step()


        # Meta-learning update weight network
        unif_prob = self.model.forward(pseudo_unif_user, pseudo_unif_item).detach()
        unif_user = self.unif_user
        unif_item = self.unif_item
        unif_y = self.unif_y

        # Create model copy for meta-learning update
        model = DisBC(self.n_user, self.n_item, self.embdim).cuda()
        model.train()
        model.load_state_dict(self.model.state_dict().copy())

        # Calculate weighted loss for biased samples
        train_z = model.get_z(self.train_user, self.train_item)
        w1 = self.w.forward(self.train_user, self.train_item, self.train_y.long())
        train_loss1 = w1 * loss_func(model.from_z_to_predict(train_z), self.train_y)

        # Calculate weighted loss for pseudo-uniform samples
        pseudo_unif_z = model.get_z(pseudo_unif_user, pseudo_unif_item)
        y_idx = np.ones(shape=(pseudo_unif_z.shape[0],), dtype=np.int64) * 2
        y_idx = torch.from_numpy(y_idx).cuda()
        w2 = self.w.forward(pseudo_unif_user, pseudo_unif_item, y_idx.long())
        train_loss2 = w2 * loss_func(model.from_z_to_predict(pseudo_unif_z), unif_prob.round())

        train_loss = torch.mean(train_loss1) + train_loss2.mean()

        # Calculate gradients and update model copy
        model.zero_grad()
        grads = torch.autograd.grad(train_loss, (model.params()), create_graph=True)
        model.update_params(lr_inner=0.1, source_params=grads)

        # Update weight network: minimize loss on uniform data
        self.w_optim.zero_grad()
        r_p = model.forward(unif_user, unif_item)
        loss = loss_func(r_p, unif_y).mean()
        loss.backward()
        if self.epoch_cnt > -1:
            self.w_optim.step()


        # Recalculate main model loss using updated weights
        train_z = self.model.get_z(self.train_user, self.train_item)
        w1 = self.w.forward(self.train_user, self.train_item, self.train_y.long())
        train_loss1 = w1 * loss_func(self.model.from_z_to_predict(train_z), self.train_y)

        pseudo_unif_z = self.model.get_z(pseudo_unif_user, pseudo_unif_item)
        w2 = self.w.forward(pseudo_unif_user, pseudo_unif_item, y_idx)
        train_loss2 = w2 * loss_func(self.model.from_z_to_predict(pseudo_unif_z), unif_prob.round())

        train_loss = torch.mean(train_loss1) + torch.mean(train_loss2)



        with torch.autograd.detect_anomaly():
            # Calculate clustering loss: make embeddings of positive and negative samples more separable in feature space
            train_pos_index = (self.train_y).bool()
            train_neg_index = (1 - self.train_y).bool()

            # Calculate prototypes (mean vectors) of positive and negative samples in training set
            train_pos_proto = torch.mean(train_z[train_pos_index], dim=0, keepdim=True)
            train_neg_proto = torch.mean(train_z[train_neg_index], dim=0, keepdim=True)

            # Intra-class loss: make samples of the same class closer to their prototype
            train_intra_loss = (
                (train_z[train_pos_index] - train_pos_proto.detach()).pow(2).sum(dim=1).sum()  + \
                (train_z[train_neg_index] - train_neg_proto.detach()).pow(2).sum(dim=1).sum()
                ) / train_z.shape[0]

            # Inter-class loss: make prototypes of different classes as far apart as possible
            train_inter_loss = - (train_neg_proto - train_pos_proto.detach()).pow(2).sum()

            train_cluster_loss = train_intra_loss + train_inter_loss

            # Calculate clustering loss for pseudo-uniform samples
            unif_z = self.model.get_z(pseudo_unif_user, pseudo_unif_item)
            with torch.no_grad():
                unif_r = self.model.forward(pseudo_unif_user, pseudo_unif_item).round()
                pos_num = unif_r.sum()
                unif_pos_index = (unif_r == 1).bool()
                unif_neg_index = (unif_r == 0).bool()
            if pos_num.bool() and (311704 - pos_num).bool() and self.epoch_cnt > 1:
                unif_pos_proto = torch.mean(unif_z[unif_pos_index], dim=0, keepdim=True)
                unif_neg_proto = torch.mean(unif_z[unif_neg_index], dim=0, keepdim=True)

                unif_intra_loss = (
                    (unif_z[unif_pos_index] - unif_pos_proto.detach()).pow(2).sum(dim=1).sum() + \
                    (unif_z[unif_neg_index] - unif_neg_proto.detach()).pow(2).sum(dim=1).sum()
                ) / unif_z.shape[0]

                unif_inter_loss = - (unif_neg_proto - unif_pos_proto.detach()).pow(2).sum()

                unif_cluster_loss = unif_intra_loss + unif_inter_loss
            else:
                unif_cluster_loss = torch.tensor([0.1]).cuda()


            # Domain adversarial training: update generator so discriminator cannot distinguish origin
            train_z = self.model.get_z(self.train_user, self.train_item)
            pseudo_unif_z = self.model.get_z(pseudo_unif_user, pseudo_unif_item)

            self.model.train()
            self.dc.eval()
            # Use gradient reversal layer to update generator in direction that confuses discriminator
            train_prob = self.dc.forward(GradReverseLayer.apply(train_z), True)
            unif_prob = self.dc.forward(GradReverseLayer.apply(pseudo_unif_z), True)

            # Adversarial loss: make discriminator unable to distinguish between biased and uniform samples
            dc_train_loss = torch.mean(torch.log(train_prob + min_value)) + torch.mean(torch.log(1 - unif_prob + min_value))

            # Calculate distribution alignment loss: make mean predictions of two distributions as close as possible
            y_pseudo_unif_output = self.model.forward(pseudo_unif_user, pseudo_unif_item)
            y_train_output = self.model.forward(self.train_user, self.train_item)

            st_loss = - torch.log((y_train_output.detach().mean() - y_pseudo_unif_output.mean()).sigmoid() + min_value)

            # Calculate total loss, different weights for different parts
            total_loss = 0.01 * (train_cluster_loss + unif_cluster_loss) + \
                         0.10 * dc_train_loss + \
                         1.00 * st_loss + \
                         1.00 * train_loss

            loss_val = {
                'train_loss': train_loss.item(),
                'cluster_loss': train_cluster_loss.item() + unif_cluster_loss.item(),
                'dc_train_loss': dc_train_loss.item(),
                'st_loss': st_loss.item(),
            }

            # Update main model parameters
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            return loss_val





    @torch.no_grad()
    def test(self):
        # Start testing
        self.model.eval()
        with torch.no_grad():
            test_users = torch.empty(0, dtype=torch.int64).cuda()
            test_items = torch.empty(0, dtype=torch.int64).cuda()
            test_pre_ratings = torch.empty(0).cuda()
            test_ratings = torch.empty(0).cuda()
            for _, (users, items, ratings) in enumerate(self.test_loader):
                users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
                pre_ratings = self.model.predict(users, items)
                test_users = torch.cat((test_users, users))
                test_items = torch.cat((test_items, items))
                test_pre_ratings = torch.cat((test_pre_ratings, pre_ratings))
                test_ratings = torch.cat((test_ratings, ratings))

            # train_results = utils.metrics.evaluate(train_pre_ratings, train_ratings, ['MSE', 'NLL'])
            test_results = utils.metrics.evaluate(test_pre_ratings, test_ratings,
                                                  ['MSE', 'NLL', 'AUC', 'Recall_Precision_NDCG@'], users=test_users,
                                                  items=test_items)
        return {}, test_results



    def run(self):
        data_record = {}
        for epo in range(self.args['epochs']):
            self.epoch_cnt = epo

            train_loss = self.train()


            if epo % 1 == 0:
                train_results, test_results = self.test()
                test_results.update(train_loss)
                data_record[epo] = test_results

                print('Epoch:{0:2d}/{1}, TRAIN:{2}, TEST:{3}'.
                    format(epo, self.args['epochs'],
                            ' '.join([key + ':' + '%.7f' % train_results[key] for key in train_results]),
                            ' '.join([key + ':' + '%.7f' % test_results[key] for key in test_results])))

                for key in test_results:
                    self.writer.add_scalar(key, test_results[key], epo)
                if epo >= 50 and self.early_stopping.check([test_results['AUC']], epo):
                    break

        # restore best model
        print('Loading {}th epoch'.format(self.early_stopping.best_epoch))
        self.model.load_state_dict(self.early_stopping.best_state)

        # test metrics
        test_users = torch.empty(0, dtype=torch.int64).cuda()
        test_items = torch.empty(0, dtype=torch.int64).cuda()
        test_pre_ratings = torch.empty(0).cuda()
        test_ratings = torch.empty(0).cuda()
        for _, (users, items, ratings) in enumerate(self.test_loader):
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
            pre_ratings = self.model.predict(users, items)
            test_users = torch.cat((test_users, users))
            test_items = torch.cat((test_items, items))
            test_pre_ratings = torch.cat((test_pre_ratings, pre_ratings))
            test_ratings = torch.cat((test_ratings, ratings))

        test_results = utils.metrics.evaluate(test_pre_ratings, test_ratings,
                                              ['MSE', 'NLL', 'AUC', 'Recall_Precision_NDCG@'], users=test_users,
                                              items=test_items)

        print('-' * 99)
        print('The performance of TEST: {}'.format(
            ' '.join([key + ':' + '%.7f' % test_results[key] for key in test_results])))
        print('-' * 99)





if __name__ == "__main__":
    """
    Main function entry point
    """
    # Parse command line arguments
    argv = argparse.ArgumentParser()
    argv.add_argument('--seed',type=int,default=0, help='Random seed')
    argv.add_argument('--device',type=int,default=1, help='GPU device ID')
    argv.add_argument('--data_name', type=str, default='yahooR3', help='Dataset name: yahooR3, coat, KuaiRand-1K')
    argv.add_argument('--type', type=str, default='implicit', help='Feedback type: implicit/explicit')
    argv.add_argument("--threshold", type=int, default=4, help='Rating binarization threshold')
    argv.add_argument("--debug",type=bool, default=False, help='Whether to enable debug mode')
    argv.add_argument("--epochs", type=int, default=200, help='Maximum training epochs')
    argv.add_argument("--patience", type=int, default=50, help='Early stopping patience')
    args = argv.parse_args()

    args = vars(args)

    # Automatically set threshold based on dataset
    if "coat" in args["data_name"] or "yahooR3" in args["data_name"]:
        args["threshold"] = 4
    elif "KuaiRand-1K" in  args["data_name"]:
        args["threshold"] = 1

    # Set random seed and device
    setup_seed(args["seed"])
    torch.cuda.set_device(args["device"])

    # Debug mode settings
    if args["debug"]:
        torch.backends.cudnn.benchmark = True
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        torch.autograd.set_detect_anomaly(True)

    # Load dataset
    train, unif_train, validation, test = load_dataset(data_name=args["data_name"],
                                                        type=args["type"],
                                                        unif_ratio=0.05,
                                                        seed=args["seed"],
                                                        threshold=args["threshold"],)

    # Initialize trainer and start training
    trainer = train_and_eval(train_data=train, unif_data=unif_train, test_data=test, args=args)
    trainer.run()

    """
    nohup python -u src/main.py --seed 0 --device 0 --data_name yahooR3 --type implicit --threshold 4 --debug True > logs/main.log 2>&1 &


No noise    Epoch:126/500, TRAIN:, TEST:MSE:0.1044823 NLL:-0.6820550 AUC:0.7801264 Precision@5:0.2847487 Recall@5:0.8096287 NDCG@5:0.6692963 Precision@10:0.1845321 Recall@10:1.0000000 NDCG@10:0.7435655

NOISE 0.1 The performance of TEST: MSE:0.0980258 NLL:-0.6837612 AUC:0.7781715 Precision@5:0.2855286 Recall@5:0.8108202 NDCG@5:0.6630555 Precision@10:0.1845321 Recall@10:1.0000000 NDCG@10:0.7367171

NOISE 0 Epoch:140/200, TRAIN:, TEST:MSE:0.0973824 NLL:-0.6836848 AUC:0.7799769 Precision@5:0.2863085 Recall@5:0.8118678 NDCG@5:0.6696260 Precision@10:0.1845321 Recall@10:1.0000000 NDCG@10:0.7426778

NOISE 1e-3 The performance of TEST: MSE:0.0969683 NLL:-0.6836986 AUC:0.7800108 Precision@5:0.2860485 Recall@5:0.8114521 NDCG@5:0.6693076 Precision@10:0.1845321 Recall@10:1.0000000 NDCG@10:0.7425280
    """


    # The performance of TEST: MSE:0.1518036 NLL:-0.6756849 AUC:0.7043791 Precision@5:0.3210300 Recall@5:0.5555065 NDCG@5:0.5215713 Precision@10:0.2583691 Recall@10:0.8133445 NDCG@10:0.6166809


"""
for dataset coat
nohup python -u src/main.py --seed 0 --device 0 --data_name coat --type implicit --threshold 4 --debug True > logs/main_coat.log 2>&1 &

for yahoo!R3
nohup python -u src/main.py --seed 0 --device 0 --data_name yahooR3 --type implicit --threshold 4 --debug True > logs/main_yahooR3.log 2>&1 &

"""