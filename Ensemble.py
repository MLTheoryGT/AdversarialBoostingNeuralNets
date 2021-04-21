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

class Ensemble(MetricPlotter, Validator):
    def __init__(self, weakLearners=[], weakLearnerWeights=[], weakLearnerType=WongNeuralNetCIFAR10, attack_eps=[]):
        """
        """
#         print("weakLearners before super call:", weakLearners)
        MetricPlotter.__init__(self, 'Number of weak learners')
        Validator.__init__(self)
#         print("weakLearners after super call:", weakLearners)
        self.weakLearners = weakLearners
        self.weakLearnerWeights = weakLearnerWeights
        self.weakLearnerType = weakLearnerType
        self.accuracies['wl_train'] = []
        self.accuracies['wl_val'] = []
        self.attack_eps = attack_eps
        assert len(self.weakLearners) == 0

    def plot_wl_acc(self, path=None):
        plt.subplots()
        plt.plot(self.train_checkpoints, self.accuracies['wl_train'])
        plt.plot(self.train_checkpoints, self.accuracies['wl_val'])
        plt.legend(['Train accuracy', 'Val accuracy'])
        plt.xlabel(self.xlabel)
        plt.ylabel('Weak learner accuracy')
        plt.title('Weak learner accuracy')
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
    
    def addWeakLearner(self, weakLearner, weakLearnerWeight):
        self.weakLearners.append(weakLearner)
        self.weakLearnerWeights.append(weakLearnerWeight)
    
    def getSingleWLPrediction(self, i, wLPredictions, X):
#         print("memory allocated:", cutorch.memory_allocated(0), "for WL num ", i)
        if not isinstance(self.weakLearners[i], str):
            learner = self.weakLearners[i]
        else:
            learner = self.weakLearnerType()
            learner.model.load_state_dict(torch.load(self.weakLearners[i]))
            learner.model = learner.model.to(torch.device('cuda:0'))
            learner.model.eval()
        prediction = learner.model(X)
        wLPredictions[:,:,i] = prediction
        return wLPredictions
            
            
    def getWLPredictions(self, X, k):
        T = len(self.weakLearners)
#         print("total num weak learners:", T)
#         print("Start WL Pred memory allocated:", cutorch.memory_allocated(0)) 
        wLPredictions = torch.zeros((X.shape[0], k, T)).cuda()

                    
        for i in range(T):
            wLPredictions = self.getSingleWLPrediction(i, wLPredictions, X)
        
        torch.cuda.empty_cache()
        
#         print("Finishing WL pred memory allocated:", cutorch.memory_allocated(0))
        return wLPredictions

    def predict(self, X):
        # We hope to deprecate schapirePredict since we shouldn't need anymore
#         learner = self.weakLearnerType()
#         learner.model.load_state_dict(torch.load(self.weakLearners[0]))
#         learner.model = learner.model.to(torch.device('cuda:0'))
#         learner.model.eval()
#         prediction = learner.model(X)
#         print("calling modded predict")
#         return prediction
    
        return self.schapireContinuousPredict(X, 10)

    def schapirePredict(self, X, k):
        print("shouldn't be here")
        wLPredictions = None

        predictions = np.zeros(len(X))
        T = len(self.weakLearners)
        
        wLPredictions = self.getWLPredictions(X, k).argmax(axis=1).transpose(1, 0)

        for i in range(len(X)):
            F_Tx =[]
            for l in range(k):
                F_Tx.append(sum([self.weakLearnerWeights[t] * (1 if wLPredictions[t][i] == l else 0) for t in range(T)]))
        
            predictions[i] = np.argmax(np.array(F_Tx))
        return predictions
    
    def schapireContinuousPredict(self, X, k):
        wLPredictions = None

        T = len(self.weakLearners)
        
        wLPredictions = self.getWLPredictions(X, k)
#         print(wlPre)
        weights = torch.tensor(self.weakLearnerWeights).unsqueeze(1).float().cuda()
        assert(wLPredictions.shape == (len(X), k, T))
#         print("weights shape:", weights.shape)
#         print("T:", T)
        assert(weights.shape == (T, 1))
        output = torch.matmul(wLPredictions, weights).squeeze(2)
        del wLPredictions
        assert(output.shape == (len(X), k))
        
        return output
    
    def gradOptWeights(self, X):
        # gradient optimize weights
        # get ~100 adv examples
        # calculate cross-entropy loss
        # calc gradient with respect to weights
    
    def toggleGrad(self):
        
    
    
#     def calc_adv_accuracy(self, dataset, num_batches = 15, train = False, attack_names = []):
#         accuracies = {}
#         loader = torch.utils.data.DataLoader(
#             dataset('./data', train=train, download=True, transform=transforms.Compose([
#             transforms.ToTensor(),
#             ])),
#             batch_size=100, shuffle=True)
        
    
#     def calc_accuracy(self, dataset, num_batches = 15, train=True):
#         totalIts = 0
#         loader = torch.utils.data.DataLoader(
#             dataset('./data', train=train, download=True, transform=transforms.Compose([
#             transforms.ToTensor(),
#             ])),
#             batch_size=100, shuffle=True)
        
#         accuracy = 0
#         for data in loader:
#             train_X_default = data[0]
#             train_Y_default = data[1]
            
# #             predictions = self.schapirePredict(train_X_default.to(torch.device('cuda:0')), 10)
#             predictions = self.schapireContinuousPredict(train_X_default.to(torch.device('cuda:0')), 10).argmax(axis=1).numpy()
            
#             accuracy += (predictions == train_Y_default.numpy()).astype(int).sum()/len(predictions)
#             totalIts+=1
#             if totalIts > num_batches:
#                 break
#         return accuracy / totalIts
    
#     def record_accuracies(self, dataset, wl_idx):
#         train_acc = self.calc_accuracy(dataset, train=True)
#         val_acc = self.calc_accuracy(dataset, train=False)
#         print("After newest WL, the ensemble's validation score is: ", train_acc)
#         print("After newest WL, the ensemble's training score is: ", val_acc)
#         self.accuracies['train'].append(train_acc)
#         self.accuracies['val'].append(val_acc)
#         self.train_checkpoints.append(wl_idx)
#         self.val_checkpoints.append(wl_idx)
    def get_sum(self, dicts):
        ans = {}
        for d in dicts:
            for k in d:
                if isinstance(k, str) and k == 'train' or k == 'val':
                    if k not in ans:
                        ans[k] = d[k]
                    else:
                        ans[k] += d[k]
                else:
                    if k not in ans:
                        ans[k] = np.array(d[k])
                    else:
                        ans[k] += np.array(d[k])
        return ans
    
    def get_mean(self, dicts):
        ans = {}
        for d in dicts:
            for k in d:
                if isinstance(k, str) and k == 'train' or k == 'val':
                    if k not in ans:
                        ans[k] = d[k]
                    else:
                        ans[k] += d[k]
                else:
                    if k not in ans:
                        ans[k] = np.array(d[k])
                    else:
                        ans[k] += np.array(d[k])
        for k in ans:
            ans[k] = ans[k] / len(dicts)
        return ans

    def record_accuracies(self, progress, train_loader, test_loader, numsamples_train, numsamples_val, val_attacks=[], attack_iters=20):
        
        # record train
        if numsamples_train >0:
            train_batch_size = train_loader.batch_size
            self.train_checkpoints.append(progress)
            # sum losses
            # average accuracies
            curSample = 0
            train_loss_dicts = []
            train_acc_dicts = []
            for i, data in enumerate(train_loader):
                curSample += train_batch_size
                if curSample >= numsamples_train: break
                losses, accuracies = self.calc_accuracies(data[0].cuda(), data[1].cuda(), data_type='train')
                train_loss_dicts.append(losses)
                train_acc_dicts.append(accuracies)
            self.losses['train'].append(self.get_sum(train_loss_dicts)['train'])
            self.accuracies['train'].append(self.get_mean(train_acc_dicts)['train'])
        
        # record val / adversarial
        
        val_batch_size = test_loader.batch_size
        self.val_checkpoints.append(progress)
        val_loss = 0
        val_acc = 0
        curSample = 0
        val_loss_dicts = []
        val_acc_dicts = []
        for i, data in enumerate(test_loader):
            curSample += val_batch_size
            if curSample >= numsamples_val: break
            losses, accuracies = self.calc_accuracies(data[0].cuda(), data[1].cuda(), data_type='val', val_attacks=val_attacks, attack_iters=attack_iters)
            val_loss_dicts.append(losses)
            val_acc_dicts.append(accuracies)
            print(accuracies)
        val_loss_dict = self.get_sum(val_loss_dicts)
        val_acc_dict = self.get_mean(val_acc_dicts)
        self.losses['val'].append(val_loss_dict['val'])
        self.accuracies['val'].append(val_acc_dict['val'])
        for k in val_loss_dict:
#             print("k:", k)
            if isinstance(k, str): continue
            attack = k
#             print("attack:", attack) #should include fgsm
            
            if len(self.losses[attack.__name__]) == 0:
                self.losses[attack.__name__] = [[] for i in range(len(self.attack_eps))]
                self.accuracies[attack.__name__] = [[] for i in range(len(self.attack_eps))]
            for i in range(len(self.attack_eps)):
                self.losses[attack.__name__][i].append(val_loss_dict[attack][i])
                self.accuracies[attack.__name__][i].append(val_acc_dict[attack][i])
        