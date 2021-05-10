from __future__ import print_function
import torch
from torchvision import datasets, transforms
import numpy as np
from WongBasedTraining import WongBasedTrainingCIFAR10
from Architectures import PreActResNet18, PreActResNet18_100, WideResNet
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json

from Boosting import Ensemble, runBoosting
from AdversarialAttacks import attack_fgsm, attack_pgd

import utils

import argparse

cuda = torch.device('cuda:0')

# epsilons = [0.127]

# Training the ensemble
from AdversarialAttacks import attack_fgsm, attack_pgd

def train_ensemble(train_config)

#     ensemble = runBoosting(num_wl, maxSamples, dataset=dataset, weakLearnerType=weakLearnerType, val_attacks=val_attacks, 
#                             attack_eps_nn=attack_eps_nn, attack_eps_ensemble=attack_eps_ensemble, train_eps_nn=train_eps_nn, batch_size=batch_size,
#                            model_base=model_base, attack_iters=attack_iters, val_restarts=restarts, lr_max=lr_max)
    
    ensemble = runBoosting(train_config)

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
    parser.add_argument("--num_samples_wl", default=30000, type=int)
    parser.add_argument("--num_wl", default=15, type=int) # CHANGE
    parser.add_argument("--train_eps_wl", default=8, type=int)
    parser.add_argument("--training_valrestarts", default=1, type=int)
    parser.add_argument("--dataset_name", default='cifar10')
    parser.add_argument("--train_alpha_wl", default=10, type=int)
    parser.add_argument("--lr_min_wl", default=0, type=float)
    parser.add_argument("--lr_max_wl", default=0.2, type=float)
    parser.add_argument("--weight_decay_wl", default=5e-4, type=float)
    parser.add_argument("--early_stop_wl", action='store_true')
    parser.add_argument("--delta_init_wl", default='random')
    parser.add_argument("--momentum_wl", default=0.9, type=float)
    parser.add_argument("--seed_wl", default=0, type=int)
    parser.add_argument("--opt_level_wl", default="O2")
    parser.add_argument("--loss_scale_wl", default=1.0, type=float)
    parser.add_argument("--delta_init_wl", default='random')
    parser.add_argument("--momentum_wl", default=0.9, type=float)
    parser.add_argument("--attack_iters_wl", default=20, type=int)
    parser.add_argument("--batch_size_wl", default=128, type=int)
    parser.add_argument("--restarts_wl", default=1, type=int)
    parser.add_argument("--training_method", default='wong'),
    parser.add_argument("--adv_train_wl", action='store_true')
    parser.add_argument("--prediction_weights_wl", action='store_false')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    if args.dataset_name == 'cifar10':
        dataset = datasets.CIFAR10
        model_base = PreActResNet18
    elif args.dataset_name == 'cifar100':
        dataset = datasets.CIFAR100
        model_base = PreActResNet18_100

    # TODO? change these to args?
    weakLearnerType = WongBasedTrainingCIFAR10
    val_attacks = [attack_pgd]
    attack_eps_wl = [0.127]
    
    
    path = f'./models/{args.dataset}/{args.maxSamples}Eps{args.train_eps_nn}'

    train_config = {
        "weakLearnerType": weakLearnerType,
        "dataset": dataset,
        "model_base": model_base,
        "attack_eps_wl": attack_eps_wl,
        "val_attacks_wl": val_attacks,

        "num_samples_wl", args.num_samples_wl,
        "num_wl": args.num_wl,
        "train_eps_wl": args.train_eps_wl,
        "training_valrestarts": args.training_valrestarts,
        "dataset_name": args.dataset_name,
        "train_alpha_wl": args.train_alpha_wl,
        "lr_min_wl": args.lr_min_wl,
        "lr_max_wl": args.lr_max_wl,
        "weight_decay_wl": args.weight_decay_wl.
        "early_stop_wl": args.early_stop_wl,
        "delta_init_wl", args.delta_init_wl,
        "momentum_wl": args.momentum_wl,
        "seed_wl": args.seed_wl,
        "opt_level_wl": args.opt_level_wl,
        "loss_scale_wl": args.loss_scale_wl,
        "delta_init_wl": args.delta_init_wl,
        "momentum_wl": args.momentum_wl,
        "attack_iters_wl": args.attack_iters_wl,
        "batch_size_wl": args.batch_size_wl,
        "restarts_wl": args.restarts_wl,
        "lr_schedule_wl": args.lr_schedule_wl,
        "adv_train_wl": args.adv_train_wl,
        "prediction_weights_wl": args.prediction_weights_wl
    }

    params_file = args.training_method + '_params.json'
    with open(params_file) as f:
        data = json.load(f)
        for k, v in data.items():
            train_config[k] = v


    ensemble = train_ensemble(train_config)

#     ensemble = train_ensemble(num_wl=args.num_wl, maxSamples=args.maxSamples, dataset=dataset,weakLearnerType=weakLearnerType,
#     val_attacks=val_attacks, train_eps_nn=args.train_eps_nn, model_base=model_base, attack_iters=args.attack_iters, restarts=args.training_valrestarts)
    

    

