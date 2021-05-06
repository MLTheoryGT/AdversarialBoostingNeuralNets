from BaseModels import BaseNeuralNet
from Architectures import PreActResNet18, WideResNet
from torchvision import datasets, transforms
from utils import applyDSTrans
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from WongBasedTraining import WongBasedTrainingCIFAR10
from pytorch_memlab import LineProfiler 
from BaseModels import MetricPlotter, Validator
import torch.cuda as cutorch
import gc
import sys
from datetime import datetime
from Ensemble import Ensemble
from AdversarialAttacks import attack_fgsm, attack_pgd
import csv
import os

def SchapireWongMulticlassBoosting(weakLearnerType, numLearners, dataset, maxSamples, alphaTol=1e-5, attack_eps_nn=[], attack_eps_ensemble=[],train_eps_nn=0.127, val_attacks=[], predictionWeights=False, batch_size=200, model_base=PreActResNet18, attack_iters=20, val_restarts=1, lr_max=0.2):    
    print("attack_eps_nn: ", attack_eps_nn)
    print("attack_eps_ensemble: ", attack_eps_ensemble)
    print("train_eps_nn: ", train_eps_nn)
    
    if dataset==datasets.CIFAR10:
        dataset_name = "cifar10"
    elif dataset==datasets.CIFAR100:
        dataset_name = "cifar100"
    
    train_ds, test_ds = applyDSTrans(dataset)
    train_ds.targets = torch.tensor(np.array(train_ds.targets))
    test_ds.targets = torch.tensor(np.array(test_ds.targets))

    m = len(train_ds)
    k = len(train_ds.classes)
    
    # Regular loaders used for training
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    for data in test_loader:
        val_X = data[0].cuda()
        val_y = data[1].cuda()
        break

    train_loader_default = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader_default = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    for data in train_loader_default:
        train_X = data[0].cuda()
        train_y = data[1].cuda()
        break

    f = np.zeros((m, k))
    
    print("attack eps ens", attack_eps_ensemble)
    ensemble = Ensemble(weakLearners=[], weakLearnerType=weakLearnerType, attack_eps=attack_eps_ensemble, model_base=model_base)
    
    start = datetime.now()
    
    path_head = f'./models/{dataset_name}/{maxSamples}Eps{train_eps_nn}/'
    print("path_head:", path_head)
    if os.path.exists(path_head):
        print("Already exists, exiting")
        return
    else:
        os.mkdir(path_head)
    
    def gcLoop():
        print("-"*100)
        print("Training weak learner {} of {}".format(t, numLearners))
        # Boosting matrix update
 
        C_t = np.zeros((m, k))
        fcorrect = f[np.arange(m), train_ds.targets]
        fexp = np.exp(f - fcorrect[:,None])
        C_t = fexp.copy()
        fexp[np.arange(m), train_ds.targets] = 0
        C_t[np.arange(m), train_ds.targets] = -np.sum(fexp, axis=1)
        C_tp = np.abs(C_t)
        
        # Set up boosting samplers
        train_sampler = BoostingSampler(train_ds, C_tp)
        train_loader = torch.utils.data.DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size)
        
        # Fit WL on weighted subset of data
        h_i = weakLearnerType(attack_eps=attack_eps_nn, train_eps=train_eps_nn)
        
        h_i.fit(train_loader, test_loader, C_t, adv_train=True, val_attacks=val_attacks, maxSample=maxSamples, predictionWeights=predictionWeights, attack_iters=attack_iters, restarts=val_restarts, lr_max=lr_max, dataset_name=dataset_name)

        # Get training and test acuracy of WL
        _, predictions, _ = pytorch_predict(h_i.model, test_loader_default, torch.device('cuda')) #y_true, y_pred, y_pred_prob
        wl_test_acc = (predictions == test_ds.targets.numpy()).astype(int).sum()/len(predictions)
        
        ensemble.accuracies['wl_val'].append(wl_test_acc)
        print("Test accuracy of weak learner: ", wl_test_acc)
        
        _, predictions, _ = pytorch_predict(h_i.model, train_loader_default, torch.device('cuda')) #y_true, y_pred, y_pred_prob
        wl_train_acc = (predictions == train_ds.targets.numpy()).astype(int).sum()/len(predictions)
        
        ensemble.accuracies['wl_train'].append(wl_train_acc)
        print("Training accuracy of weak learner: ", wl_train_acc)
        

        # update f one batch at a time via f[np.arange(batchsize)]
        advBatchSize = train_loader_default.batch_size
        a = 0
        allIndices = np.zeros(m)
        for advCounter, data in enumerate(train_loader_default):
            X = data[0].cuda()
            y = data[1].cuda()
#             print("train_eps_nn: ", train_eps_nn)
#             delta = attack_fgsm(X, y, attack_eps_nn[0], h_i.predict)
            delta = attack_pgd(X, y, attack_eps_nn[0], h_i.predict, restarts=1)
            predictions = h_i.predict(X + delta).argmax(axis=1)
#             print("RegularAcc: ", (regPred==y).int().sum()/y.shape[0])
#             print("AdvAcc: ", (predictions==y).int().sum()/y.shape[0])
            indices = predictions.detach().int().cpu().numpy()
            allIndices[np.arange(advCounter*advBatchSize, (advCounter + 1)*advBatchSize)] = indices
        
        print("After allindices: ", datetime.now()-start)
        print("Predictions: ", allIndices[:10])
        
        # Get alpha for this weak learners
        a = -C_t[np.arange(m), allIndices.astype(int)].sum()
        b = fexp.sum()
        delta_t = a / b
        alpha = 1/2*np.log((1+delta_t)/(1-delta_t))
        print("Alpha: ", alpha)
        print("before pessimistic update: ", datetime.now()-start)
        
        y_train = train_ds.targets
        correctIndices = (allIndices == y_train.numpy())
        incorrectIndices = (allIndices != y_train.numpy())
        f[np.arange(m), y_train] += alpha * correctIndices
        f[incorrectIndices, :] += alpha
        f[np.arange(m), y_train] -= alpha * incorrectIndices
    
        # save WL model and alpha
        model_path = f'{path_head}wl_{t}.pth'
        torch.save(h_i.model.state_dict(), model_path)
        
        h_iRef = h_i
        h_iModelRef = h_i.model
        del h_i.model
        del h_i
        del predictions
        
        torch.cuda.empty_cache()
        ensemble.addWeakLearner(model_path, alpha)
        print("t: ", t, "memory allocated:", cutorch.memory_allocated(0))   
        print("After WL ", t, " time elapsed(s): ", (datetime.now() - start).seconds)
    
        
    #end of gc loop
    for t in range(numLearners):
        gcLoop()
        gc.collect()
        
    weight_path = f'{path_head}wl_weights.csv'
    print("weights:", ensemble.weakLearnerWeights)
    
    with open(weight_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(ensemble.weakLearnerWeights)
        
    return ensemble


def pytorch_predict(model, test_loader, device):
    '''
    Make prediction from a pytorch model 
    '''
    # set model to evaluate model
    model.eval()
    
    y_true = torch.tensor([], dtype=torch.long, device=device)
    all_outputs = torch.tensor([], device=device)
    
    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        for data in test_loader:
            inputs = [i.to(device) for i in data[:-1]]
            labels = data[-1].to(device)
            
            outputs = model(*inputs)
            y_true = torch.cat((y_true, labels), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)
    
    y_true = y_true.cpu().numpy()  
    _, y_pred = torch.max(all_outputs, 1)
    y_pred = y_pred.cpu().numpy()
    y_pred_prob = F.softmax(all_outputs, dim=1).cpu().numpy()
    
    return y_true, y_pred, y_pred_prob


from torch.utils.data.sampler import Sampler
class BoostingSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, C):
        self.indices = indices
        self.C = C

    def __iter__(self):
        sampleWeights = self.C.sum(axis = 1)
        probs = sampleWeights/sampleWeights.sum()
        maxNum = len(probs)
        chosen = np.random.choice(maxNum, size=maxNum, replace=True, p=probs)
        # print("Sampler chosen shape: ", chosen.shape)
        return np.nditer(chosen)

    def __len__(self):
        return len(self.indices)
      
    def setC(self, C):
        self.C = C

def runBoosting(numWL, maxSamples, dataset=datasets.CIFAR10, weakLearnerType=WongBasedTrainingCIFAR10, val_attacks=[],
               attack_eps_nn=[], attack_eps_ensemble=[], train_eps_nn=0.3, batch_size=200, model_base=PreActResNet18, attack_iters=20, val_restarts=1, lr_max=0.2):

    from datetime import datetime

    t0 = datetime.now()

    ensemble = SchapireWongMulticlassBoosting(weakLearnerType, numWL, dataset, alphaTol=1e-10, val_attacks=val_attacks, maxSamples = maxSamples, predictionWeights=False, attack_eps_nn=attack_eps_nn, attack_eps_ensemble=attack_eps_ensemble,train_eps_nn=train_eps_nn, batch_size=batch_size, model_base=model_base, attack_iters=attack_iters, val_restarts=val_restarts, lr_max=lr_max)

    print("Finished in", (datetime.now()-t0).total_seconds(), " s")
    
    return ensemble
