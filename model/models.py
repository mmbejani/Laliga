from model.metrics import compute_accuracy_batch, compute_accuracy, compute_loss_on_batch, compute_loss, \
    compute_sub_accuracy
from model.utils import ImpactWave
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from model.utils import print_progress_bar
from time import time
from numpy import linalg
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn
import numpy as np
from model.utils import linear_wav, gaussian_wav, exponential_wav


class PassiveAuxiliaryLoss(nn.Module):

    def __init__(self, net: nn.Module, loss_function):
        super().__init__()
        self.net = net
        self.loss_function = loss_function

    def forward(self, x, y):
        outputs = self.net(x)
        losses = list()
        for output in outputs:
            losses.append(self.loss_function(output, y))
        return losses


class AuxiliaryLossFunction(nn.Module):

    def __init__(self, gamma, net: nn.Module, loss_function):
        super().__init__()
        self.net = net
        self.gamma = gamma
        self.loss_function = loss_function
        self.im = ImpactWave(self.gamma, 0.1, 4.)
        self.opt = SGD(net.t_param, 0.1, momentum=0.9)
        self.step_lr_80 = StepLR(self.opt, step_size=80, gamma=0.1)
        self.step_lr_40 = StepLR(self.opt, step_size=40, gamma=0.1)
        self.step_lr_20 = StepLR(self.opt, step_size=20, gamma=0.1)
        self.list_w_star = list()
        self.v = list()
        self.best_accuracy = 0.
        self.writer = SummaryWriter('./run2/cifar10-linear-1-4/main_acc')
        self.writers = list()
        for g in gamma:
            self.writers.append(SummaryWriter('./run2/cifar10-linear-1-4/sub_acc_{0}'.format(g)))

        for r_param in self.net.r_weights:
            temp_list = list()
            for w in r_param:
                temp_list.append(w.detach().cpu().numpy())
            self.list_w_star.append(temp_list)

    def forward(self, x, y):
        outputs = self.net(x)
        alpha = torch.tensor(self.im.alpha, requires_grad=False).cuda()
        loss_val = alpha[-1] * self.loss_function()(outputs[-1], y)
        for i, g in enumerate(self.gamma):
            #aux_loss = alpha[i] * (self.loss_function()(outputs[g], y) + 0.0001 * self.create_reg_term(g))
            aux_loss = alpha[i] * (self.loss_function()(outputs[g], y))
            loss_val += aux_loss
        return loss_val

    def approximation_svd_matrix(self, w) -> np.ndarray:
        u, s, v = linalg.svd(w)
        d = self.optimal_d(s)
        s = np.diag(s)
        wa = np.dot(u[:, :d], np.dot(s[:d, :d], v[:d, :]))
        return wa

    def approximate_svd_tensor(self, w: np.ndarray) -> np.ndarray:
        w_shape = w.shape
        n1 = w_shape[0]
        n2 = w_shape[1]
        ds = []
        if w_shape[2] == 1 or w_shape[3] == 1:
            return w
        u, s, v = linalg.svd(w)
        for i in range(n1):
            for j in range(n2):
                ds.append(self.optimal_d(s[i, j]))
        d = int(np.mean(ds))
        w = np.matmul(u[..., 0:d], s[..., 0:d, None] * v[..., 0:d, :])
        return w

    @staticmethod
    def optimal_d(s):
        variance = np.std(s)
        mean = np.average(s)
        for i in range(s.shape[0] - 1):
            if s[i] < mean + variance:
                return i
        return s.shape[0] - 1

    def create_reg_term(self, i):
        list_w = self.net.r_weights[i]
        list_w_star = self.list_w_star[i]
        sub_w = list()
        for w, w_star in zip(list_w, list_w_star):
            w_star_approx = w_star
            sub_w.append(torch.norm(w - torch.tensor(w_star_approx, requires_grad=False).cuda()))
        reg = sum(sub_w)
        return reg

    def update_w_star(self, current_accuracy):
        if current_accuracy > self.best_accuracy:
            print('The W*s are going to be updated...')
            self.best_accuracy = current_accuracy
            for i, g in enumerate(self.gamma):
                for i, w in enumerate(self.net.r_weights[g]):
                    wt = w.detach().cpu().numpy()
                    if len(wt.shape) == 4:
                        wt = self.approximate_svd_tensor(wt)
                    elif len(wt.shape) == 2 and np.prod(wt.shape) < 300000:
                        wt = self.approximation_svd_matrix(wt)
                    self.list_w_star[g][i] = wt

    def fit(self, train_data: DataLoader, test_data: DataLoader, epochs: int, save_model=False):
        max_batch_len = len(train_data)
        total_acc = 0.
        total_time = 0
        for epoch in range(1, epochs + 1):
            tic = time()
            for i, batch in enumerate(train_data):
                self.opt.zero_grad()
                loss_val = self(batch[0].cuda(), batch[1].cuda())
                loss_val.backward()
                self.opt.step()
                acc = compute_accuracy_batch(self.net, batch[0].cuda(), batch[1].cuda(), True)
                total_acc += acc
                print_progress_bar(i, max_batch_len, epoch, metrics={
                    'Loss Value': loss_val.detach().cpu().numpy(),
                    'Accuracy': total_acc / (i + 1)
                })

            toc = time()
            if epoch < 85 or 130 < epoch:
                self.step_lr_80.step()
            if 85 <= epoch <= 130:
                self.step_lr_40.step()
            #self.step_lr_20.step()

            acc = compute_accuracy(self.net, test_data, 'Test', True, self.writer)
            # compute_sub_accuracy(self.net, test_data, gamma=self.gamma, writers=self.writers)
            print('Elapse Time : %f' % (toc - tic))
            total_time += (toc - tic)
            total_acc = total_acc / len(train_data)
            if total_acc >= 99 .0:
                print('Algorithm reach to 90% acc in {0}'.format(total_time))
                exit()

            compute_loss(self.net, self.loss_function(), test_data, 'Test', True)

            batch = iter(train_data).next()
            train_loss_val = compute_loss_on_batch(self.net, self.loss_function(), batch[0].cuda(), batch[1].cuda(),
                                                   True)
            batch = iter(test_data).next()
            test_loss_val = compute_loss_on_batch(self.net, self.loss_function(), batch[0].cuda(), batch[1].cuda(),
                                                  True)
            self.v.append(test_loss_val / train_loss_val)
            self.im.update_wave_and_get_alpha(self.v, gaussian_wav)
            self.update_w_star(acc)
            print('The ImpactWave change alpha values to ' + str(self.im.alpha))
            total_acc = 0.
        self.writer.close()
        for w in self.writers:
            w.close()
