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

def SchapireWongMulticlassBoosting(weakLearner, numLearners, dataset, alphaTol=1e-5, attack_eps_nn=[], attack_eps_ensemble=[],train_eps_nn=0.3, adv_train=False, val_attacks=[], maxSamples=None, predictionWeights=False, weakLearnerType=WongNeuralNetCIFAR10, batch_size=200):
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
    
    

    test_loader = torch.utils.data.DataLoader(test_ds_index, batch_size=batch_size, shuffle=False)
    for data in test_loader:
        val_X = data[0].cuda()
        val_y = data[1].cuda()
        break

    train_loader_default = torch.utils.data.DataLoader(
        dataset('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=False)
    for data in train_loader_default:
        train_X = data[0].cuda()
        train_y = data[1].cuda()
        break

    f = np.zeros((m, k))    
    
    print("attack eps ens", attack_eps_ensemble)
    ensemble = Ensemble(weakLearnerType=weakLearnerType, attack_eps=attack_eps_ensemble)

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
        h_i = weakLearner(attack_eps=attack_eps_nn, train_eps=train_eps_nn)
        
        h_i.fit(train_loader, test_loader, C_t, adv_train=adv_train, val_attacks=val_attacks, maxSample=maxSample, predictionWeights=predictionWeights)
        a = 0 # for memory debugging purposes
        
        # Get training acuracy of WL
        a, predictions, b = pytorch_predict(h_i.model, train_loader_default, torch.device('cuda')) #y_true, y_pred, y_pred_prob
        wl_train_acc = (predictions == train_ds_index.targets.numpy()).astype(int).sum()/len(predictions)
        del a
        del b
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

        del h_i.model
        del h_i
        del predictions
        
        torch.cuda.empty_cache()
        ensemble.addWeakLearner(model_path, alpha)
        
        # grab accuracy of full ensemble
#         divider = 1
#         if t % divider == 0:
#             print("VAL ATTACKS", val_attacks)

        # 2 NN params on it 2

                
        print("t: ", t, "memory allocated:", cutorch.memory_allocated(0))    
        ensemble.record_accuracies(t, val_X=val_X, val_y=val_y, train_X=train_X, train_y=train_y, val_attacks=val_attacks)
        a = 0 # for memory-debugging
        
        
        if t == 0:
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass

        
    return ensemble

class Ensemble(MetricPlotter, Validator):
    def __init__(self, weakLearners=[], weakLearnerWeights=[], weakLearnerType=WongNeuralNetCIFAR10, attack_eps=[]):
        """
        """
        MetricPlotter.__init__(self, 'Number of weak learners')
        Validator.__init__(self)
        self.weakLearners = weakLearners
        self.weakLearnerWeights = weakLearnerWeights
        self.weakLearnerType = weakLearnerType
        self.accuracies['wl_train'] = []
        self.attack_eps = attack_eps

    def plot_wl_train_acc(self, path=None):
        plt.subplots()
        plt.plot(self.train_checkpoints, self.accuracies['wl_train'])
        plt.xlabel(self.xlabel)
        plt.ylabel('Weak learner train accuracy')
        plt.title('Weak learner train accuracy')
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
    
    def addWeakLearner(self, weakLearner, weakLearnerWeight):
        self.weakLearners.append(weakLearner)
        self.weakLearnerWeights.append(weakLearnerWeight)
    
    def getWLPredictions(self, X, k):
        T = len(self.weakLearners)
        wLPredictions = torch.zeros((X.shape[0], k, T)).cuda()
                    
        for i in range(T):
            if not isinstance(self.weakLearners[i], str):
                learner = self.weakLearners[i]
            else:
                learner = self.weakLearnerType()
                learner.model.load_state_dict(torch.load(self.weakLearners[i]))
                learner.model = learner.model.to(torch.device('cuda:0'))
                learner.model.eval()
            prediction = learner.model(X)
            wLPredictions[:,:,i] = prediction
            del learner.model
            del learner
        return wLPredictions

    def predict(self, X):
        # We hope to deprecate schapirePredict since we shouldn't need anymore
        return self.schapireContinuousPredict(X, 10)

    def schapirePredict(self, X, k):
                
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
        weights = torch.tensor(self.weakLearnerWeights).unsqueeze(1).float().cuda()
        assert(wLPredictions.shape == (len(X), k, T))
        assert(weights.shape == (T, 1))
        output = torch.matmul(wLPredictions, weights).squeeze(2)
        del wLPredictions
        assert(output.shape == (len(X), k))
        
        return output
    
    
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

def runBoosting(numWL, maxSamples, dataset=datasets.CIFAR10, weakLearnerType=WongNeuralNetCIFAR10, adv_train=False, val_attacks=[],
               attack_eps_nn=[], attack_eps_ensemble=[], train_eps_nn=0.3, batch_size=200):
    
    train_dataset = dataset('./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

    test_dataset = dataset('./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

    from datetime import datetime
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size, shuffle=False)
    for data in test_loader:
        val_X = data[0]
        val_y = data[1]
        break

    t0 = datetime.now()

    ensemble = SchapireWongMulticlassBoosting(weakLearnerType, numWL, dataset, alphaTol=1e-10, adv_train=adv_train, val_attacks=val_attacks, maxSamples = maxSamples, predictionWeights=False, weakLearnerType=weakLearnerType, attack_eps_nn=attack_eps_nn, attack_eps_ensemble=attack_eps_ensemble,train_eps_nn=train_eps_nn, batch_size=batch_size)

    print("Finished in", (datetime.now()-t0).total_seconds(), " s")
    
    return ensemble
