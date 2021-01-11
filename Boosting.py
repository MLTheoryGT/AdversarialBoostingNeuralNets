from BaseModels import BaseNeuralNet, PreActResNet18
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from WeakLearners import WongNeuralNetCIFAR10, Net
from pytorch_memlab import LineProfiler 
from BaseModels import MetricPlotter

def SchapireWongMulticlassBoosting(weakLearner, numLearners, dataset, advDelta=0, alphaTol=1e-5, adv_train=False, val_attacks=[], maxSamples=None, predictionWeights=False, weakLearnerType=WongNeuralNetCIFAR10):
    """
        Args:
            maxSamples: either None or a list for each wl
    """
    if maxSamples is None:
        maxSamples = [500 for i in range(numLearners)]
    else:
        maxSamples = [maxSamples for i in range(numLearners)]
    assert(numLearners == len(maxSamples))
    
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

    train_ds_index = dataset_index('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
    test_ds_index = dataset_index('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
    train_ds_index.targets = torch.tensor(np.array(train_ds_index.targets))
    test_ds_index.targets = torch.tensor(np.array(test_ds_index.targets))

    m = len(train_ds_index)
    k = len(train_ds_index.classes)

    batch_size = 100
    test_loader = torch.utils.data.DataLoader(test_ds_index, batch_size=200, shuffle=False)
    for data in test_loader:
        test_X = data[0]
        test_y = data[1]

    train_loader_default = torch.utils.data.DataLoader(
        dataset('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=100, shuffle=False)

    f = np.zeros((m, k))    
    
    ensemble = Ensemble(weakLearnerType=weakLearnerType)

    for t, maxSample in enumerate(maxSamples):
        print("-"*100)
        print("Training weak learner {}".format(t))
        # Boosting matrix update
        C_t = np.zeros((m, k))
        fcorrect = f[np.arange(m), train_ds_index.targets]
        fexp = np.exp(f - fcorrect[:,None])
        C_t = fexp.copy()
        fexp[np.arange(m), train_ds_index.targets] = 0
        C_t[np.arange(m), train_ds_index.targets] = -np.sum(fexp, axis=1)
        C_tp = np.abs(C_t)
        
        # Set up boosting samplers
        train_sampler = BoostingSampler(train_ds_index, C_tp, batch_size)
        train_loader = torch.utils.data.DataLoader(train_ds_index, sampler=train_sampler, batch_size=batch_size)
        
        # Fit WL on weighted subset of data
        h_i = weakLearner()
        h_i.fit(train_loader, test_loader, C_t, adv_train=adv_train, val_attacks=val_attacks, maxSample=maxSample, predictionWeights=predictionWeights)
        
        # Get training acuracy of WL
        _, predictions, _ = pytorch_predict(h_i.model, train_loader_default, torch.device('cuda')) #y_true, y_pred, y_pred_prob
        wl_train_acc = (predictions == train_ds_index.targets.numpy()).astype(int).sum()/len(predictions)
        ensemble.accuracies['wl_train'].append(wl_train_acc)
        print("Training accuracy of weak learner: ", wl_train_acc)
        
        # Get alpha for this weak learners
        a = -C_t[np.arange(m), predictions].sum()
        b = fexp.sum()
        delta_t = a / b
        alpha = 1/2*np.log((1+delta_t)/(1-delta_t))
        print("Alpha: ", alpha)
        f[np.arange(m), predictions] += alpha
        
        # save WL model and alpha
        path_name = 'mnist' if dataset==datasets.MNIST else 'cifar10'
        model_path = f'./models/{path_name}_wl_{t}.pth'
        torch.save(h_i.model.state_dict(), model_path)
        del h_i
        del predictions
        torch.cuda.empty_cache()
        ensemble.addWeakLearner(model_path, alpha)
        
        # grab accuracy of full ensemble
        divider = 1
        if t % divider == 0:
            ensemble.record_accuracies(dataset, t)
        
        
    return ensemble

class Ensemble(MetricPlotter):
    def __init__(self, weakLearners=[], weakLearnerWeights=[], weakLearnerType=WongNeuralNetCIFAR10):
        """
        """
        super().__init__('Number of weak learners')
        self.weakLearners = weakLearners
        self.weakLearnerWeights = weakLearnerWeights
        self.weakLearnerType = weakLearnerType
        self.accuracies['wl_train'] = []
    
    def addWeakLearner(self, weakLearner, weakLearnerWeight):
        self.weakLearners.append(weakLearner)
        self.weakLearnerWeights.append(weakLearnerWeight)

    def getWLPredictionsString(self, X, k, argmax=True):
        T = len(self.weakLearners)
        wLPredictions = []
        for i in range(T):
            learner = self.weakLearnerType()
            learner.model.load_state_dict(torch.load(self.weakLearners[i]))
            learner.model = learner.model.to(torch.device('cuda:0'))
            learner.model.eval()
            if argmax:
                prediction = learner.model(X).argmax(axis = 1)
            else:
                prediction = learner.model(X)
            wLPredictions.append(prediction)
        return wLPredictions
    
    def getWLPredictions(self, X, k):
        T = len(self.weakLearners)
        wLPredictions = torch.tensor((k, T))
        for i in range(T):
            if not isinstance(self.weakLearners[i], str):
                learner = self.weakLearners[i]
            else:
                learner = self.weakLearnerType()
                learner.model.load_state_dict(torch.load(self.weakLearners[i]))
                learner.model = learner.model.to(torch.device('cuda:0'))
                learner.model.eval()
            prediction = learner.model(X)
            wLPredictions[:,i] = prediction
        return WLPredictions


    def schapirePredict(self, X, k):
        wLPredictions = None

        predictions = np.zeros(len(X))
        T = len(self.weakLearners)
        
#         if isinstance(self.weakLearners[0], str):
#             wLPredictions = self.getWLPredictionsString(X, k)
#         else:
#             wLPredictions = [self.weakLearners[i].predict(X).argmax(axis=1) for i in range(T)]
        
        wLPredictions = self.getWLPredictions(X, k)
        
        for i in range(len(X)):
            F_Tx =[]
            for l in range(k):
                F_Tx.append(sum([self.weakLearnerWeights[t] * (1 if wLPredictions[t][i] == l else 0) for t in range(T)]))
        
            predictions[i] = np.argmax(np.array(F_Tx))
        return predictions
    
    def schapireContinuousPredict(self, X, k):
        wLPredictions = None

        predictions = np.zeros(len(X))
        T = len(self.weakLearners)
        
        if isinstance(self.weakLearners[0], str):
            wLPredictions = self.getWLPredictionsString(X, k)
        else:
            wLPredictions = [self.weakLearners[i].predict(X).argmax(axis=1) for i in range(T)]
        
        nn_outputs = torch.tensor((k, T))
        for i in range(T):
            nn_outputs[:,i] = 
    
    
    def calc_adv_accuracy(self):
        # TODO
        pass
    
    def calc_accuracy(self, dataset, num_batches = 15, train=True):
        totalIts = 0
        loader = torch.utils.data.DataLoader(
            dataset('./data', train=train, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
            batch_size=100, shuffle=True)
        
        accuracy = 0
        for data in loader:
            train_X_default = data[0]
            train_Y_default = data[1]
            
            predictions = self.schapirePredict(train_X_default.to(torch.device('cuda:0')), 10)
            accuracy += (predictions == train_Y_default.numpy()).astype(int).sum()/len(predictions)
            totalIts+=1
            if totalIts > num_batches:
                break
        return accuracy / totalIts
    
    def record_accuracies(self, dataset, wl_idx):
        train_acc = self.calc_accuracy(dataset, train=True)
        val_acc = self.calc_accuracy(dataset, train=False)
        print("After newest WL, the ensemble's validation score is: ", train_acc)
        print("After newest WL, the ensemble's training score is: ", val_acc)
        self.accuracies['train'].append(train_acc)
        self.accuracies['val'].append(val_acc)
        self.train_checkpoints.append(wl_idx)
        self.val_checkpoints.append(wl_idx)


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

def runBoosting(numWL, maxSamples, dataset=datasets.CIFAR10, weakLearnerType=WongNeuralNetCIFAR10, adv_train=False, val_attacks=[]):
    train_dataset = dataset('./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

    test_dataset = dataset('./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

    from datetime import datetime
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=200, shuffle=False)
    for data in test_loader:
        val_X = data[0]
        val_y = data[1]
        break

    t0 = datetime.now()

    ensemble = SchapireWongMulticlassBoosting(weakLearnerType, numWL, dataset, advDelta=0, alphaTol=1e-10, adv_train=adv_train, val_attacks=val_attacks, maxSamples = maxSamples, predictionWeights=False, weakLearnerType=weakLearnerType)

    print("Finished in", (datetime.now()-t0).total_seconds(), " s")
    
    return ensemble
