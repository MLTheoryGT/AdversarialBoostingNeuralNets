from BaseModels import BaseNeuralNet
from trades import trades_loss
import torch
from torchvision import datasets
import numpy as np
cuda = torch.device('cuda:0')
from utils import (cifar10_std_tup, cifar10_mu_tup, cifar10_std, cifar10_mu, cifar10_upper_limit, cifar10_lower_limit)
from utils import (cifar100_std_tup, cifar100_mu_tup, cifar100_std, cifar100_mu, cifar100_upper_limit, cifar100_lower_limit)
from utils import SnapshotEnsembleScheduler

def adjust_learning_rate(optimizer, epoch, config):
    """decrease the learning rate"""
    lr = config["lr_wl"]
    if epoch >= 75:
        lr = config["lr_wl"] * 0.1
    if epoch >= 90:
        lr = config["lr_wl"] * 0.01
    if epoch >= 100:
        lr = config["lr_wl"] * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

        opt = torch.optim.SGD(model.parameters(), lr=config['lr_wl'], momentum=config['momentum_wl'], weight_decay=config['weight_decay_wl'])
        
        epoch_size = len(train_loader.dataset)
        num_epochs = max(1, config["num_samples_wl"] // epoch_size)
        lr_steps = num_epochs * len(train_loader) For cyclic LR if we want
        if config["lr_schedule_wl"] == "snapshot":
            scheduler = SnapshotEnsembleScheduler(opt, lr_steps, config["snapshot_cycles_wl"], config["snapshot_a0_wl"])
            
        # Training
        currSamples = 0 # added
        snapshots = 0
        for epoch in range(num_epochs):
            print("Epoch: ", epoch)
            if config["lr_schedule_wl"] != "snapshot":
                adjust_learning_rate(opt, epoch, config)
                
            for i, data in enumerate(train_loader):
                X, y = data[0], data[1]
                currSamples += train_loader.batch_size # added
                X, y = X.cuda(), y.cuda()
                if i % 100 == 99:
                    self.record_accuracies(currSamples, val_X=val_X, val_y=val_y, train_X=X, train_y=y, attack_iters=config["attack_iters_val_wl"], 
                                            restarts=config["training_valrestarts"], val_attacks=config["val_attacks"], dataset_name=config["dataset_name"])
                loss = trades_loss(model=model,
                    x_natural=X,
                    y=y,
                    optimizer=opt,
                    step_size=config["step_size_wl"],
                    epsilon=config["train_eps_wl"],
                    perturb_steps=config["num_steps_wl"],
                    beta=config["beta_wl"])
                loss.backward()
                opt.step()
            
                if config["lr_schedule_wl"] == "snapshot":
                    scheduler.step()
                
                if config["lr_schedule_wl"] == "snapshot" and scheduler.snapshot():
                    print("doing snapshot, lr = ", scheduler.get_last_lr()[0])
                    torch.save(model.state_dict(), config["save_dir"] + str(config["snapshot_cycles_wl"] - snapshots - 1) + '.pth')
                    snapshots+=1

