from BaseModels import BaseNeuralNet, PreActResNet18
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from WeakLearners import WongNeuralNetCIFAR10, Net
from pytorch_memlab import LineProfiler 
from BaseModels import MetricPlotter, Validator
import torch.cuda as cutorch
import gc
import sys
from utils import cifar10_mean, cifar10_std
from datetime import datetime
from Ensemble import Ensemble
from AdversarialAttacks import attack_fgsm, attack_pgd
import csv
import os

def SchapireWongMulticlassBoosting(weakLearnerType, numLearners, dataset, alphaTol=1e-5, attack_eps_nn=[], attack_eps_ensemble=[],train_eps_nn=0.127, adv_train_prefix=0, val_attacks=[], maxSamples=None, predictionWeights=False, batch_size=200, val_flag=True):
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
    
    print("attack_eps_nn: ", attack_eps_nn)
    print("attack_eps_ensemble: ", attack_eps_ensemble)
    print("train_eps_nn: ", train_eps_nn)
    if maxSamples is None:
        maxSamples = [500 for i in range(numLearners)]
    else:
        maxSamples = [maxSamples for i in range(numLearners)]
    assert(numLearners == len(maxSamples))
    
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
                
    assert(len(train_transforms) == 4)
    assert(len(test_transforms) == 2)

    train_ds_index = dataset_index('./data', train=True, download=True, transform=transforms.Compose(train_transforms))
    test_ds_index = dataset_index('./data', train=False, download=True, transform=transforms.Compose(test_transforms))
    train_ds_index.targets = torch.tensor(np.array(train_ds_index.targets))
    test_ds_index.targets = torch.tensor(np.array(test_ds_index.targets))

    m = len(train_ds_index)
    k = len(train_ds_index.classes)
    
    # Regular loaders used for training
    test_loader = torch.utils.data.DataLoader(test_ds_index, batch_size=batch_size, shuffle=True)
    for data in test_loader:
        val_X = data[0].cuda()
        val_y = data[1].cuda()
        break

    train_loader_default = torch.utils.data.DataLoader(
        dataset('./data', train=True, download=True, transform=transforms.Compose(train_transforms)),
        batch_size=batch_size, shuffle=False)
    test_loader_default = torch.utils.data.DataLoader(
        dataset('./data', train=False, download=True, transform=transforms.Compose(train_transforms)),
        batch_size=batch_size, shuffle=False)
    for data in train_loader_default:
        train_X = data[0].cuda()
        train_y = data[1].cuda()
        break
       
    #mini loaders for ensemble
    test_loader_mini = torch.utils.data.DataLoader(test_ds_index, batch_size=10, shuffle=True)

    train_loader_mini = torch.utils.data.DataLoader(
        dataset('./data', train=True, download=True, transform=transforms.Compose(train_transforms)),
        batch_size=10, shuffle=True) #change to True?

    f = np.zeros((m, k))
    
    print("attack eps ens", attack_eps_ensemble)
    ensemble = Ensemble(weakLearners=[], weakLearnerType=weakLearnerType, attack_eps=attack_eps_ensemble)
    
    start = datetime.now()
    
    dir_name = maxSamples[0]
    path_head = f'./models/{dir_name}Eps{train_eps_nn}/'
    print("path_head:", path_head)
    if not os.path.exists(path_head):
        os.mkdir(path_head)
    
    
    def gcLoop():
        print("-"*100)
        print("Training weak learner {} of {}".format(t, numLearners))
        # Boosting matrix update
 
        C_t = np.zeros((m, k))
        fcorrect = f[np.arange(m), train_ds_index.targets]
        fexp = np.exp(f - fcorrect[:,None])
        C_t = fexp.copy()
        fexp[np.arange(m), train_ds_index.targets] = 0
        C_t[np.arange(m), train_ds_index.targets] = -np.sum(fexp, axis=1)
        C_tp = np.abs(C_t)
        print("C_t: ", C_t[:10])
        print("targets: ", train_ds_index.targets[:10])
        print("f: ", f[:10])
        print("fexp: ", fexp[:10])
        
        # Set up boosting samplers
        train_sampler = BoostingSampler(train_ds_index, C_tp, batch_size)
        train_loader = torch.utils.data.DataLoader(train_ds_index, sampler=train_sampler, batch_size=batch_size)
        
        # Fit WL on weighted subset of data
        h_i = weakLearnerType(attack_eps=attack_eps_nn, train_eps=train_eps_nn)
        
        if t < adv_train_prefix:
            h_i.fit(train_loader, test_loader, C_t, adv_train=True, val_attacks=val_attacks, maxSample=maxSample, predictionWeights=predictionWeights)
        else:
            h_i.fit(train_loader, test_loader, C_t, adv_train=False, val_attacks=val_attacks, maxSample=maxSample, predictionWeights=predictionWeights)
        
        print("After fit function: ", datetime.now()-start)
        # Get training and test acuracy of WL
        _, predictions, _ = pytorch_predict(h_i.model, test_loader_default, torch.device('cuda')) #y_true, y_pred, y_pred_prob
        wl_test_acc = (predictions == test_ds_index.targets.numpy()).astype(int).sum()/len(predictions)
        
        ensemble.accuracies['wl_val'].append(wl_test_acc)
        print("Test accuracy of weak learner: ", wl_test_acc)
        
        _, predictions, _ = pytorch_predict(h_i.model, train_loader_default, torch.device('cuda')) #y_true, y_pred, y_pred_prob
        wl_train_acc = (predictions == train_ds_index.targets.numpy()).astype(int).sum()/len(predictions)
        
        ensemble.accuracies['wl_train'].append(wl_train_acc)
        print("Training accuracy of weak learner: ", wl_train_acc)
        
        print("After train/test acc: ", datetime.now()-start)
        
    
        # enumerate dataloader
        # delta = fgsm(X, y, model)
        # update f one match at a time via f[np.arange(batchsize)]
        advBatchSize = train_loader_default.batch_size
        a = 0
        allIndices = np.zeros(m)
        for advCounter, data in enumerate(train_loader_default):
            X = data[0].cuda()
            y = data[1].cuda()
#             print("train_eps_nn: ", train_eps_nn)
#             delta = attack_fgsm(X, y, attack_eps_nn[0], h_i.predict)
            delta = attack_pgd(X, y, attack_eps_nn[0], h_i.predict, restarts=1)
        
#             print("delta max: ", delta.max())
#             print("delta min: ", delta.min())
#             flatDelta = torch.flatten(max_delta).squeeze().cpu().numpy()
            # uncomment if i want delta to be printed out
#             print(flatDelta)
#             plt.hist(flatDelta)
#             plt.show()
#             regPred = h_i.predict(X).argmax(axis=1)
#             print("regPred shape: ", regPred.shape)
#             print("y shape: ", y.shape)
            predictions = h_i.predict(X + delta).argmax(axis=1)
#             print("RegularAcc: ", (regPred==y).int().sum()/y.shape[0])
#             print("AdvAcc: ", (predictions==y).int().sum()/y.shape[0])
#             print(predictions)
#             print(h_i.predict(X))
            indices = predictions.detach().int().cpu().numpy()
            allIndices[np.arange(advCounter*advBatchSize, (advCounter + 1)*advBatchSize)] = indices
        
        print("After allindices: ", datetime.now()-start)
        print("Predictions: ", allIndices[:10])
        
        # Get alpha for this weak learners
#         a = -C_t[np.arange(m), predictions].sum()
        a = -C_t[np.arange(m), allIndices.astype(int)].sum()
        b = fexp.sum()
        delta_t = a / b
        alpha = 1/2*np.log((1+delta_t)/(1-delta_t))
#         alpha /= 2 #(Seeing if val accuracy improves if I decay the alpha parameter)
        print("Alpha: ", alpha)
        
        # targetIndices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
        # correctIndices
        # incorrectIndices
        # labels are: train_loader_default.targets
        print("before pessimistic update: ", datetime.now()-start)
        
        y_train = train_ds_index.targets
        correctIndices = (allIndices == y_train.numpy())
        print("correct Indices Mask: ", correctIndices)
        incorrectIndices = (allIndices != y_train.numpy())
        print("incorrect Indices Mask: ", incorrectIndices)
        f[np.arange(m), y_train] += alpha * correctIndices
        f[incorrectIndices, :] += alpha
        f[np.arange(m), y_train] -= alpha * incorrectIndices
        print("after pessimistic update: ", datetime.now()-start)
        
#         f[np.arange(m), allIndices.astype(int)] += alpha < -- old version
        

    
        # save WL model and alpha
        path_name = 'mnist' if dataset==datasets.MNIST else 'cifar10'
        
        
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
#         ensemble.record_accuracies(t, val_X=val_X, val_y=val_y, train_X=train_X, train_y=train_y, val_attacks=val_attacks)
        if val_flag:
            ensemble.record_accuracies(t, train_loader_mini, test_loader_mini, 1000, 1000, val_attacks=val_attacks)
    

    
        
    #end of gc loop
    """
        Args:
            maxSamples: either None or a list for each wl
    """

    for t, maxSample in enumerate(maxSamples):
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

    def __init__(self, indices, C, batch_size):
        self.indices = indices
        self.C = C
        self.batch_size = batch_size

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

def runBoosting(numWL, maxSamples, dataset=datasets.CIFAR10, weakLearnerType=WongNeuralNetCIFAR10, adv_train_prefix=0, val_attacks=[],
               attack_eps_nn=[], attack_eps_ensemble=[], train_eps_nn=0.3, batch_size=200, val_flag=True):
    
#     train_dataset = dataset('./data', train=True, download=True, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 ]))

#     test_dataset = dataset('./data', train=False, download=True, transform=transforms.Compose([
#                 transforms.ToTensor(),
#                 ]))

    from datetime import datetime
#     test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size, shuffle=False)
#     for data in test_loader:
#         val_X = data[0]
#         val_y = data[1]
#         break

    t0 = datetime.now()

    ensemble = SchapireWongMulticlassBoosting(weakLearnerType, numWL, dataset, alphaTol=1e-10, adv_train_prefix=adv_train_prefix, val_attacks=val_attacks, maxSamples = maxSamples, predictionWeights=False, attack_eps_nn=attack_eps_nn, attack_eps_ensemble=attack_eps_ensemble,train_eps_nn=train_eps_nn, batch_size=batch_size, val_flag=val_flag)

    print("Finished in", (datetime.now()-t0).total_seconds(), " s")
    
    return ensemble
