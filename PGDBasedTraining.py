import torch
import torch.nn as nn
import numpy as np
import time
from utils import clamp
from BaseModels import BaseNeuralNet
from torchvision import datasets
cuda = torch.device('cuda:0')
from utils import (cifar10_std_tup, cifar10_mu_tup, cifar10_std, cifar10_mu, cifar10_upper_limit, cifar10_lower_limit)
from utils import (cifar100_std_tup, cifar100_mu_tup, cifar100_std, cifar100_mu, cifar100_upper_limit, cifar100_lower_limit)


class PGDBasedTraining(BaseNeuralNet):
    def __init__(self, attack_eps, model_base):
        super().__init__(model_base, attack_eps)
    
    # def fit(self, train_loader, test_loader, C=None, lr_schedule="cyclic", lr_min=0, lr_max=0.2, weight_decay=5e-4, early_stop=True,
    #               momentum=0.9, epsilon=8, alpha=10, delta_init="random", seed=0, opt_level="O2", loss_scale=1.0,
    #               maxSample = None, adv_train=False, val_attacks = [], predictionWeights=None, attack_iters=20, restarts=1, dataset_name="cifar10"):
    def fit(self, train_loader, test_loader, config):
        np.random.seed(config["seed_wl"])
        torch.manual_seed(config["seed_wl"])
        torch.cuda.manual_seed(config["seed_wl"])

        if config["dataset_name"] == "cifar10":
            (std_tup, mu_tup, std, mu, upper_limit, lower_limit) = (cifar10_std_tup, cifar10_mu_tup, cifar10_std, cifar10_mu, cifar10_upper_limit, cifar10_lower_limit)
        elif config["dataset_name"] == "cifar100":
            (std_tup, mu_tup, std, mu, upper_limit, lower_limit) = (cifar100_std_tup, cifar100_mu_tup, cifar100_std, cifar100_mu, cifar100_upper_limit, cifar100_lower_limit)

        val_X = None
        val_y = None
        for data in test_loader:
            val_X = data[0].to(cuda)
            val_y = data[1].to(cuda)
            break

        epsilon = (config["train_eps_wl"] / 255.) / std
        alpha = (config["train_alpha_wl"] / 255.) / std

        model = self.model
        model.train()

        opt = torch.optim.SGD(model.parameters(), lr=config["lr_max_wl"], momentum=config["momentum_wl"], weight_decay=config["weight_decay_wl"])
        # amp_args = dict(opt_level=config["opt_level_wl"], loss_scale=config["loss_scale_wl"], verbosity=False)
        # if config["opt_level_wl"] == 'O2':
        #     amp_args['master_weights'] = config["master_weights_wl"]
        # model, opt = amp.initialize(model, opt, **amp_args)
        criterion = nn.CrossEntropyLoss()

        epoch_size = len(train_loader.dataset)
        num_epochs = max(1, config["num_samples_wl"] // epoch_size)
        lr_steps = num_epochs * len(train_loader)

        if config["lr_schedule_wl"] == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=config["lr_min_wl"], max_lr=config["lr_max_wl"],
                step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        elif config["lr_schedule_wl"] == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

        # Training
        currSamples = 0
        start_train_time = time.time()
        print('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
        for epoch in range(num_epochs):
            start_epoch_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            for i, (X, y) in enumerate(train_loader):
                currSamples += train_loader.batch_size 
                X, y = X.cuda(), y.cuda()
                delta = torch.zeros_like(X).cuda()
                if config["delta_init_wl"] == 'random':
                    for i in range(len(epsilon)):
                        delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta.requires_grad = True
                if i % 100 == 99:
                    self.record_accuracies(currSamples, val_X=val_X, val_y=val_y, train_X=X, train_y=y, attack_iters=config["attack_iters_val_wl"], 
                                            restarts=config["training_valrestarts"], val_attacks=config["val_attacks"], dataset_name=config["dataset_name"])
                for _ in range(config["attack_iters_wl"]):
                    output = model(X + delta)
                    loss = criterion(output, y)
                    # with amp.scale_loss(loss, opt) as scaled_loss:
                    #     scaled_loss.backward()
                    loss.backward()
                    grad = delta.grad.detach()
                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta.grad.zero_()
                delta = delta.detach()
                output = model(X + delta)
                loss = criterion(output, y)
                opt.zero_grad()
                # with amp.scale_loss(loss, opt) as scaled_loss:
                #     scaled_loss.backward()
                loss.backward()
                opt.step()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
                scheduler.step()
            epoch_time = time.time()
            lr = scheduler.get_lr()[0]
            print('%d \t %.1f \t \t %.4f \t %.4f \t %.4f'%(epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n))
        train_time = time.time()
        # torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
        print('Total train time: %.4f minutes', (train_time - start_train_time)/60)