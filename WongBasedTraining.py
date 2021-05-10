from BaseModels import BaseNeuralNet
from Architectures import PreActResNet18, WideResNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import torch.cuda as cutorch
cuda = torch.device('cuda:0')
from datetime import datetime
import time
# import logging
from apex import amp
# from torch.cuda import amp
from utils import clamp
import copy

from utils import (cifar10_std_tup, cifar10_mu_tup, cifar10_std, cifar10_mu, cifar10_upper_limit, cifar10_lower_limit)
from utils import (cifar100_std_tup, cifar100_mu_tup, cifar100_std, cifar100_mu, cifar100_upper_limit, cifar100_lower_limit)


class WongBasedTrainingCIFAR10(BaseNeuralNet):
    def __init__(self, attack_eps = [], train_eps=8, model_base=PreActResNet18):
        super().__init__(model_base)
        self.train_eps = train_eps
        self.attack_eps = attack_eps
    
    def fit(self, train_loader, test_loader, C=None, lr_schedule="cyclic", lr_min=0, lr_max=0.2, weight_decay=5e-4, early_stop=True,
                  momentum=0.9, epsilon=8, alpha=10, delta_init="random", seed=0, opt_level="O2", loss_scale=1.0, out_dir="WongNNCifar10",
                  maxSample = None, adv_train=False, val_attacks = [], predictionWeights=None, attack_iters=20, restarts=1, dataset_name="cifar10"):
        train_loader.dataset
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
    
        if dataset_name == "cifar10":
            (std_tup, mu_tup, std, mu, upper_limit, lower_limit) = (cifar10_std_tup, cifar10_mu_tup, cifar10_std, cifar10_mu, cifar10_upper_limit, cifar10_lower_limit)
        elif dataset_name == "cifar100":
            (std_tup, mu_tup, std, mu, upper_limit, lower_limit) = (cifar100_std_tup, cifar100_mu_tup, cifar100_std, cifar100_mu, cifar100_upper_limit, cifar100_lower_limit)

#         train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
#          have our own loaders

        val_X = None
        val_y = None
        for data in test_loader:
            val_X = data[0].to(cuda)
            val_y = data[1].to(cuda)
            break

        epsilon = (epsilon / 255.) / std
        alpha = (alpha / 255.) / std
        pgd_alpha = (2 / 255.) / std

        model = self.model
        model.train()

        opt = torch.optim.SGD(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=weight_decay)
        amp_args = dict(opt_level=opt_level, loss_scale=loss_scale, verbosity=False)
        if opt_level == 'O2':
            amp_args['master_weights'] = True
#             unclear if True is what we want here
        model, opt = amp.initialize(model, opt, **amp_args)
        criterion = nn.CrossEntropyLoss()

        if delta_init == 'previous':
            delta = torch.zeros(train_loader.batch_size, 3, 32, 32).cuda()
        
        epoch_size = len(train_loader.dataset)
        num_epochs = max(1, maxSample // epoch_size)
        lr_steps = num_epochs * len(train_loader)
        
#         lr_steps = args.epochs * len(train_loader)
#         redid lr_steps to work with maxsample
        if lr_schedule == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=lr_min, max_lr=lr_max,
                step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        elif lr_schedule == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

        # Training
        prev_robust_acc = 0.
        start_train_time = time.time()
        print('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
        currSamples = 0 # added
        for epoch in range(num_epochs):
            start_epoch_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            print("train_loader:", train_loader) #delete
            for i, data in enumerate(train_loader):
                X, y = data[0], data[1]
                currSamples += train_loader.batch_size # added
                X, y = X.cuda(), y.cuda()
                if i == 0:
                    first_batch = (X, y)
                if delta_init != 'previous':
                    delta = torch.zeros_like(X).cuda()
                if delta_init == 'random':
                    for j in range(len(epsilon)):
                        delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                if i % 100 == 99: #Adding this part from our own code (for validation)
#                     print("about to record accs", val_attacks)
                    self.record_accuracies(currSamples, train_X = X, train_y = y, val_X=val_X, val_y=val_y , attack_iters=attack_iters, restarts=restarts, val_attacks=val_attacks, dataset_name = dataset_name)
                delta.requires_grad = True
                output = model(X + delta[:X.size(0)])
                loss = F.cross_entropy(output, y)
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                delta = delta.detach()
                output = model(X + delta[:X.size(0)])
                loss = criterion(output, y)
                opt.zero_grad()
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                opt.step()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
                scheduler.step()
#             if early_stop:
#                 # Check current PGD robustness of model using random minibatch
#                 X, y = first_batch
#                 pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
#                 with torch.no_grad():
#                     output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
#                 robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
#                 if robust_acc - prev_robust_acc < -0.2:
#                     break
#                 prev_robust_acc = robust_acc
#                 best_state_dict = copy.deepcopy(model.state_dict())
            epoch_time = time.time()
            lr = scheduler.get_last_lr()[0]
