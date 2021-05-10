from BaseModels import BaseNeuralNet
from trades import trades_loss
import torch
from torchvision import datasets
import numpy as np
cuda = torch.device('cuda:0')
from utils import (cifar10_std_tup, cifar10_mu_tup, cifar10_std, cifar10_mu, cifar10_upper_limit, cifar10_lower_limit)
from utils import (cifar100_std_tup, cifar100_mu_tup, cifar100_std, cifar100_mu, cifar100_upper_limit, cifar100_lower_limit)


class TradesBasedTrainingCIFAR10(BaseNeuralNet):
    def __init__(self, attack_eps, model_base):
        super().__init__(model_base, attack_eps)
    
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

        model = self.model
        model.train()

        opt = torch.optim.SGD(model.parameters(), lr=config['lr_max_wl'], momentum=config['momentum_wl'], weight_decay=config['weight_decay_wl'])
        
        epoch_size = len(train_loader.dataset)
        num_epochs = max(1, config["num_samples_wl"] // epoch_size)
        # lr_steps = num_epochs * len(train_loader) For cyclic LR if we want

        # Training
        currSamples = 0 # added
        for epoch in range(num_epochs):
            print("Epoch: ", epoch)
            for i, data in enumerate(train_loader):
                X, y = data[0], data[1]
                currSamples += train_loader.batch_size # added
                X, y = X.cuda(), y.cuda()
                if i % 100 == 99:
                    self.record_accuracies(currSamples, val_X=val_X, val_y=val_y, train_X=X, train_y=y, attack_iters=config["attack_iters"], 
                                            restarts=config["restarts"], val_attacks=config["val_attacks"], dataset_name=config["dataset_name"])
                loss = trades_loss(model=model,
                    x_natural=X,
                    y=y,
                    optimizer=opt,
                    step_size=config["step_size_trades_wl"],
                    epsilon=config["epsilon_trades_wl"],
                    perturb_steps=["perturb_steps_trades_wl"],
                    beta=config["beta_trades_wl"],
			        distance=config["distance_trades_wl"])
                loss.backward()
                opt.step()
