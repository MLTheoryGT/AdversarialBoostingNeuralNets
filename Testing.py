from Boosting import Ensemble
from WongBasedTraining import WongBasedTrainingCIFAR10
from AdversarialAttacks import attack_pgd
from torchvision import datasets, transforms
from utils import cifar10_mean, cifar10_std
from datetime import datetime
import torch
import numpy as np

"""
    path: should contain wl number of .pth models named cifar10_wl_i.pth
        should also contain a weights.csv of form "alpha_1, alpha_2, ..., alpha_n"
    attacks: pass in functions
    numWL: int
"""
def testEnsemble(path, attacks, numWL, dataset=datasets.CIFAR10, numsamples_train=1000, numsamples_val=1000, attack_eps_ensemble = [0.03], attack_iters=20, restarts=10, gradOptWeights=False):
    train_transforms = []
    test_transforms = []
    
    def dataset_with_indices(cls):
        """
        Modifies the given Dataset class to return a tuple data, target, index
        instead of just data, target.
        """

        def __getitem__(self, index):
            data, target = cls.__getitem__(self, index)
            return data, target, index

        return type(cls.__name__, (cls,), {
            '__getitem__': __getitem__,
        })
    dataset_index = dataset_with_indices(dataset)
    
    train_transforms = []
    test_transforms = []
    
    if dataset == datasets.CIFAR10:
        for elt in [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]:
            train_transforms.append(elt)
    
    tens = transforms.ToTensor()
    train_transforms.append(tens)
    test_transforms.append(tens)
    
    if dataset == datasets.CIFAR10:
        norm = transforms.Normalize(cifar10_mean, cifar10_std)
        train_transforms.append(norm)
        test_transforms.append(norm)
    
    train_ds_index = dataset_index('./data', train=True, download=True, transform=transforms.Compose(train_transforms))
    test_ds_index = dataset_index('./data', train=False, download=True, transform=transforms.Compose(test_transforms))
    train_ds_index.targets = torch.tensor(np.array(train_ds_index.targets))
    test_ds_index.targets = torch.tensor(np.array(test_ds_index.targets))
    
    
    #mini loaders for ensemble
    # @Arvind, I think you may be able to change the batch size here
    test_loader_mini = torch.utils.data.DataLoader(test_ds_index, batch_size=128, shuffle=True) #Note: change this to True when using a subset
    train_loader_mini = torch.utils.data.DataLoader(
        dataset('./data', train=True, download=True, transform=transforms.Compose(train_transforms)),
        batch_size=100, shuffle=True) #change to True?
    
    
    wl = []
    wlWeights = []
    with open(path + "/wl_weights.csv") as f:
        for line in f:
            wlWeights = line.split(",")
            break
    
    for i in range(len(wlWeights)):
        wlWeights[i] = float(wlWeights[i])
    for i in range(len(wlWeights)):
        wl.append(path + f'wl_{i}.pth')
    
    startTime = datetime.now()
    ensemble = Ensemble(weakLearners=[], weakLearnerWeights=[], weakLearnerType=WongBasedTrainingCIFAR10, attack_eps=attack_eps_ensemble)
    weights = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8]
    for i in range(numWL):
        print("Weak Learner ", i, ".  Time Elapsed (s): ", (datetime.now()-startTime).seconds)
        ensemble.addWeakLearner(wl[i], wlWeights[i])
#         ensemble.addWeakLearner(wl[i], 1.0)
#         ensemble.gradOptWeights(train_loader_mini)
#         ensemble.addWeakLearner(wl[i], 0.01)
#         print("before ens acc", ensemble.accuracies)
        
        ensemble.record_accuracies(i, train_loader_mini, test_loader_mini, numsamples_train, numsamples_val, val_attacks=attacks, attack_iters=attack_iters)
        print("ensemble accuracies:", ensemble.accuracies)

    return ensemble
        
    