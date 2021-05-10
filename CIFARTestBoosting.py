from __future__ import print_function
import torch
from torchvision import datasets, transforms
import numpy as np
from WongBasedTraining import WongBasedTrainingCIFAR10
from Architectures import PreActResNet18, WideResNet
import matplotlib.pyplot as plt
import os
from datetime import datetime
from Testing import testEnsemble
from Boosting import Ensemble, runBoosting
from AdversarialAttacks import attack_fgsm, attack_pgd
import json

import utils

import argparse

cuda = torch.device('cuda:0')

epsilons = [0.127]

# Training the ensemble

def test_ensemble(path, attacks, num_wl, dataset, maxSamples=750000, numsamples_train=200, numsamples_val=1500, attack_eps_ensemble=epsilons, gradOptWeights=False,
                  attack_iters=20, restarts=1, train_eps_nn=8):
    ensemble = testEnsemble(path, attacks, num_wl, dataset=dataset, numsamples_train=numsamples_train, numsamples_val=numsamples_val, 
                            attack_eps_ensemble=attack_eps_ensemble, gradOptWeights=gradOptWeights, attack_iters=attack_iters, restarts=restarts)
    if dataset == datasets.CIFAR10:
        dataset_name = 'cifar10'
    elif dataset == datasets.CIFAR100:
        dataset_name = 'cifar100'
    attackStr = "attack_pgd"
    resultsPath = f'results/plots/{dataset_name}/train_eps_{train_eps_nn}/{attackStr}/'
    acc_file = resultsPath + f'acc_maxSamples_{maxSamples}.png'
    adv_acc_file = resultsPath + f'adv_acc_maxSamples_{maxSamples}.png'
    loss_file = resultsPath + f'loss_maxSamples_{maxSamples}.png'
    ensemble.plot_accuracies(acc_file)
    ensemble.plot_loss(loss_file)
    ensemble.plot_adversarial_accuracies(adv_acc_file)
    return ensemble


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--num_samples_wl", default=30000, type=int)
    # parser.add_argument("--num_samples_train", default=200, type=int)
    # parser.add_argument("--num_samples_val", default=1500, type=int)
    # parser.add_argument("--train_eps_wl", default=8, type=int)
    # parser.add_argument("--num_wl", default=15, type=int) # CHANGE
    # parser.add_argument("--batch_size", default=128, type=int) # can try increasing this a lot
    # parser.add_argument("--attack_iters", default=20, type=int)
    # parser.add_argument("--testing_restarts", default=10, type=int)
    # parser.add_argument("--dataset_name", default='cifar10')
    parser.add_argument("--config_file", default="Configs/wongCIFAR10Test.json")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    test_config = {}

    with open(args.config_file) as f:
        test_config = json.load(f)
    

    if args.dataset == 'cifar10':
        test_config['dataset'] = datasets.CIFAR10
    elif args.dataset == 'cifar100':
        test_config['dataset'] = datasets.CIFAR100

    # TODO? change these to args?
    test_config['weakLearnerType'] = WongBasedTrainingCIFAR10
    test_config['val_attacks'] = [attack_pgd]
    test_config['model_base'] = PreActResNet18
    test_config['path' = f"./models/{test_config['dataset_name']}/{test_config['num_samples_wl']}Eps{test_config['train_eps_wl']}/"
    test_config['attack_eps_ensemble'] = [0.3]

    ensemble = test_ensemble(path, val_attacks, args.num_wl, dataset, maxSamples=args.maxSamples,numsamples_train=args.numsamples_train,
                             numsamples_val=args.numsamples_val, attack_iters=args.attack_iters, restarts=args.testing_restarts,
                             train_eps_nn=args.train_eps_nn)

    ensemble = test_ensemble(test_config)