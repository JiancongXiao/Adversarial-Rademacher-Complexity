import copy
import numpy as np
from collections import Iterable
# from scipy.stats import truncnorm

import torch
import torch.nn as nn
import time
from adversarialbox.utils import to_var

# --- White-box attacks ---
def idx2onehot(idx, n):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n).cuda()
    onehot.scatter_(1, idx, 1)

    return onehot


class FGSMAttack(object):
    def __init__(self, model=None, epsilon = 8/255):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, epsilon=8/255):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilon is not None:
            self.epsilon = epsilon
        size = X.size()
        X = np.copy(X_nat)

        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.LongTensor(y))

        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += self.epsilon * grad_sign
        if size[1] == 3:
            X = np.clip(X, -1, 1)
        else:
            X = np.clip(X, 0, 1)

        return X


class LinfPGDAttack(object):
    def __init__(self, model=None, epsilon=16/255, k=10, a=0.01, 
        random_start=True):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, epsilon = 8/255):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        X_nat = X_nat.numpy()
        if self.rand:
            X = X_nat + np.random.uniform(-self.epsilon, self.epsilon,
                X_nat.shape).astype('float32')
        else:
            X = np.copy(X_nat)

        for i in range(self.k):
            X_var = to_var(X, requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad.data.cpu().numpy()

            X += self.a * np.sign(grad)

            X = np.clip(X, X_nat - self.epsilon, X_nat + self.epsilon)
            X = np.clip(X, 0, 1) # ensure valid pixel range

        return X

class LtwoPGDattack(object):
    def __init__(self, model=None, epsilon=2, k=40, a=0.5, 
        random_start=True):

        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, X_nat, y, epsilon = 0.25):
        
        X = X_nat
        size = X_nat.size()
        batch_size = size[0]
        X_nat =  X_nat.view(batch_size,-1)

        if epsilon is not None:
            self.epsilon = epsilon

        for i in range(self.k):
            X_var = to_var(X, requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(X_var)
            loss = self.loss_fn(scores, y_var)
            loss.backward()
            grad = X_var.grad.data.cpu()
            grad = grad.view(batch_size,-1)
            norm_grad = torch.norm(grad, dim = 1).view(-1,1)
            grad = torch.div(grad,norm_grad)
            X = X.view(batch_size, -1)
            X_temp = X + self.a * grad
            dis = X_temp - X_nat
            eta = torch.norm(dis, dim = 1)
            eta = eta.view(-1,1)
            eta1 = torch.clamp(eta,0, self.epsilon)

            X = X_nat + eta1 * torch.div(dis,eta)
            X = X.view(size)
            if size[1] == 1:
                X = torch.clamp(X, 0, 1) # ensure valid pixel range
            else:
                X = torch.clamp(X, 0, 1)
        X_adv = X.numpy()
        return X_adv

