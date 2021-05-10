from __future__ import print_function
import torch
from torchvision import datasets, transforms
import numpy as np
from WongBasedTraining import WongBasedTrainingCIFAR10
from Architectures import PreActResNet18, PreActResNet18_100, WideResNet
import matplotlib.pyplot as plt
import os
from datetime import datetime

from Boosting import Ensemble, runBoosting
from AdversarialAttacks import attack_fgsm, attack_pgd

import utils

import argparse

cuda = torch.device('cuda:0')

epsilons = [0.127]

# Training the ensemble
from AdversarialAttacks import attack_fgsm, attack_pgd

def train_ensemble(num_wl=15, maxSamples=750000, dataset=datasets.CIFAR10, weakLearnerType=WongBasedTrainingCIFAR10, val_attacks=[], attack_eps_nn=epsilons, attack_eps_ensemble=epsilons, train_eps_nn=8, batch_size=100, model_base=PreActResNet18,
                              attack_iters=20, restarts=10, lr_max=0.2):

    ensemble = runBoosting(num_wl, maxSamples, dataset=dataset, weakLearnerType=weakLearnerType, val_attacks=val_attacks, 
                            attack_eps_nn=attack_eps_nn, attack_eps_ensemble=attack_eps_ensemble, train_eps_nn=train_eps_nn, batch_size=batch_size,
                           model_base=model_base, attack_iters=attack_iters, val_restarts=restarts, lr_max=lr_max)

    # WARNING: training with multiple ensembles hasn't been tested, use with caution
    return ensemble

def test_ensemble(path, attacks, num_wl, maxSamples=750000, numsamples_train=200, numsamples_val=1500, attack_eps_ensemble=epsilons, gradOptWeights=False,
                  attack_iters=20, restarts=1, train_eps_nn=8):
    ensemble = testEnsemble(path, [attack], num_wl, numsamples_train=numsamples_train, numsamples_val=numsamples_val, 
                            attack_eps_ensemble=attack_eps_ensemble, gradOptWeights=gradOptWeights, attack_iters=attack_iters, restarts=restarts)
    attackStr = "attack_pgd"
    resultsPath = f'results/plots/cifar10/train_eps_{train_eps_nn}/{attackStr}/'
    acc_file = resultsPath + f'acc_maxSamples_{maxSamples}.png'
    adv_acc_file = resultsPath + f'adv_acc_maxSamples_{maxSamples}.png'
    loss_file = resultsPath + f'loss_maxSamples_{maxSamples}.png'
    ensemble.plot_accuracies(acc_file)
    ensemble.plot_loss(loss_file)
    ensemble.plot_adversarial_accuracies(adv_acc_file)
    return ensemble


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxSamples", default=30000, type=int)
    parser.add_argument("--num_wl", default=15, type=int) # CHANGE
    parser.add_argument("--train_eps_nn", default=8, type=int)
    parser.add_argument("--batch_size", default=128, type=int) # can try increasing this a lot
    parser.add_argument("--attack_iters", default=20, type=int)
    parser.add_argument("--training_valrestarts", default=1, type=int)
    parser.add_argument("--dataset", default='cifar10')



    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10
        model_base = PreActResNet18
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100
        model_base = PreActResNet18_100

    # TODO? change these to args?
    weakLearnerType = WongBasedTrainingCIFAR10
    val_attacks = [attack_pgd]
    
    
    path = f'./models/{args.dataset}/{args.maxSamples}Eps{args.train_eps_nn}'
    
    
#     config = {
#         'num_wl': args.num_wl,
#         'weakLearnerType': weakLearnerType,
#         'dataset': dataset,
        
#     }

    ensemble = train_ensemble(num_wl=args.num_wl, maxSamples=args.maxSamples, dataset=dataset,weakLearnerType=weakLearnerType,
    val_attacks=val_attacks, train_eps_nn=args.train_eps_nn, model_base=model_base, attack_iters=args.attack_iters, restarts=args.training_valrestarts)
    

    

