from Boosting import Ensemble
from WongBasedTraining import WongBasedTrainingCIFAR10
from AdversarialAttacks import attack_pgd
from torchvision import datasets, transforms
from datetime import datetime
import torch
from utils import applyDSTrans
import numpy as np

"""
    path: should contain wl number of .pth models named cifar10_wl_i.pth
        should also contain a weights.csv of form "alpha_1, alpha_2, ..., alpha_n"
    attacks: pass in functions
    numWL: int
"""
def testEnsemble(path, attacks, numWL, dataset=datasets.CIFAR10, numsamples_train=1000, numsamples_val=1000, attack_eps_ensemble = [0.03], attack_iters=20, restarts=10, gradOptWeights=False):
    if dataset == datasets.CIFAR10:
        dataset_name = 'cifar10'
    elif dataset == dataset.CIFAR100:
        dataset_name = 'cifar100'
    train_ds, test_ds = applyDSTrans(dataset)
    train_ds.targets = torch.tensor(np.array(train_ds.targets))
    test_ds.targets = torch.tensor(np.array(test_ds.targets))
    
    
    #mini loaders for ensemble
    # @Arvind, I think you may be able to change the batch size here
    test_loader_mini = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=True) #Note: change this to True when using a subset
    train_loader_mini = torch.utils.data.DataLoader(train_ds,
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
    for i in range(numWL):
        print("Weak Learner ", i, ".  Time Elapsed (s): ", (datetime.now()-startTime).seconds)
        ensemble.addWeakLearner(wl[i], wlWeights[i])
#         ensemble.addWeakLearner(wl[i], 1.0)
#         ensemble.gradOptWeights(train_loader_mini)
#         ensemble.addWeakLearner(wl[i], 0.01)
#         print("before ens acc", ensemble.accuracies)
        
        ensemble.record_accuracies(i, train_loader_mini, test_loader_mini, numsamples_train, numsamples_val, val_attacks=attacks, attack_iters=attack_iters, dataset_name=dataset_name)
        print("ensemble accuracies:", ensemble.accuracies)

    return ensemble