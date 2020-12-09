from BaseModels import BaseNeuralNet, PreActResNet18
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from WeakLearners import WongNeuralNetCIFAR10
from pytorch_memlab import LineProfiler


class BoostedWongNeuralNet(BaseNeuralNet):

    def __init__(self):
        super().__init__(Net)

    def fit(self, train_loader, test_loader, C, alpha = 0.375, epochs = 1, lr_max = 5e-3, adv=True, maxIt = float("inf"), predictionWeights=False):
        cnt = 0
        val_X = None
        val_y = None
        for data in test_loader:
            if cnt > 1: break
            val_X = data[0].to(cuda)
            val_y = data[1].to(cuda)
            cnt += 1

        # optimizer here
        self.optimizer = torch.optim.Adam(self.model.parameters())
        lr_schedule = lambda t: np.interp([t], [0, epochs * 2//5, epochs], [0, lr_max, 0])[0]


        # test_set, val_set = torch.utils.data.random_split(test_loader.dataset, [9000, 1000])
        for epoch in range(epochs):
            print("Epoch:", epoch)
            for i, data in enumerate(train_loader):
                # print("In epoch: ", i)
                lr = lr_schedule(epoch + (i+1)/len(train_loader))
                self.optimizer.param_groups[0].update(lr=lr)
                # if cnt % 1000 == 1:
                # if i == 0:
                # print("Start with: ", self.validation(val_X, val_y))
                
                if cnt % 10 == 1:
                    # print("Iteration: ", cnt)
                    # print("memory usage:", cutorch.memory_allocated(0))
                    self.memory_usage.append(cutorch.memory_allocated(0))
                    self.iters.append(cnt)
                    val_loss, val_accuracy = self.validation(val_X, val_y)
                    self.val_losses.append(val_loss)
                    self.val_accuracies.append(val_accuracy)
                    self.losses.append(self.loss.item())
                    cnt += 1
                    X = data[0].cuda()
                    y = data[1].cuda()
                    indices = data[2].cuda()
                if i > maxIt:
                    del X
                    del y
                    torch.cuda.empty_cache()
                    print("WL has validation accuracy", self.validation(val_X, val_y))
                    return
                if adv:
                    self.batchUpdate(X, y, C, alpha = alpha)
                else:
                    # print("MB(%d), "%(i), end="")
                    self.batchUpdateNonAdv(X, y, indices, C, alpha=alpha, predictionWeights=predictionWeights)
                    del X
                    del y
        # print("Escaped epoch")
        print("WL has validation accuracy", self.val_accuracies[-1])
        print("WL has loss", self.losses[-1])

        torch.cuda.empty_cache()

    def batchUpdate(self, X, y, C, epochs = 1, epsilon = 0.3, alpha = 0.375):
        def cross_entropy(pred, soft_targets):
            logsoftmax = nn.LogSoftmax()
            return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))
        self.model.train()
        N = X.size()[0]
        
        optimizer = self.optimizer

        # Prints
        # print("In Batch Update X.shape: ", X.shape)
        # print("In Batch Update Y.shape: ", Y.shape)


        # compute delta (perturbation parameter)
        optimizer.zero_grad()
        # compute k_i's that need work (per example)
        worst_k = np.argmax(C, axis = 1)
        print("In Batch Update worst_k.shape: ", worst_k.shape)
        # compute L(x_i, k_i) for all examples
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).to(cuda)
        delta.requires_grad = True
        output = self.model(X + delta)
        loss = F.cross_entropy(output, worst_k)
        self.loss = loss
        # compute delta_k's in batch
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta - alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
        delta = delta.detach()

        # update network parameters
        output = self.model(torch.clamp(X + delta, 0, 1))
        loss = F.cross_entropy(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
    def batchUpdateNonAdv(self, X, y, indices, C, epochs = 1, epsilon = 0.3, alpha = 0.375, predictionWeights=False):
        self.model.train()
        N = X.size()[0]
        optimizer = self.optimizer

        # Prints
        # print("In Batch Update Y.shape: ", y.shape)

        # compute delta (perturbation parameter)
        optimizer.zero_grad()

        # update network parameters
        output = self.model(X)

        if predictionWeights:
            loss = F.cross_entropy(output, y, reduction="none")
            Ccopy = C.copy()
            # Ccopy = Ccopy + Ccopy.min()
            # Ccopy = Ccopy/Ccopy.max()
            Ccopy = Ccopy + Ccopy.min(axis=1)[:,None]
            Ccopy = Ccopy / Ccopy.max(axis=1)[:,None]

            c_tensor = torch.from_numpy(Ccopy).cuda()
            # print("output.shape", output.shape)
            # print("output.argmax().shape", output.argmax().shape)
            output_predicted = output.argmax(dim=1)
            c_predicted = c_tensor[indices, output_predicted]
            # print(loss)
            # print(c_predicted)


            loss = loss * (c_predicted)
            # print(loss)
            loss = loss.mean()
        else:
            loss = F.cross_entropy(output, y)

        self.loss = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def predict(self, X):
        return self.model(X)

def SchapireWongMulticlassBoosting(weakLearner, numLearners, dataset, advDelta=0, alphaTol=1e-5, adv=True, maxIt=float("inf"), predictionWeights=False):
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
    # print("m, k: ", m, k)

    batch_size = 100
    test_loader = torch.utils.data.DataLoader(test_ds_index, batch_size=200, shuffle=False)

    train_loader_default = torch.utils.data.DataLoader(
      datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
          transforms.ToTensor(),
          ])),
      batch_size=100, shuffle=False)

    weakLearners = []
    weakLearnerWeights = []
    f = np.zeros((m, k))

    for t in range(numLearners):
        print("-"*100)
        print("Training {}th weak learning".format(t))
        C_t = np.zeros((m, k))
        fcorrect = f[np.arange(m), train_ds_index.targets]
        fexp = np.exp(f - fcorrect[:,None])
        C_t = fexp.copy()
        fexp[np.arange(m), train_ds_index.targets] = 0
        C_t[np.arange(m), train_ds_index.targets] = -np.sum(fexp, axis=1)

        C_tp = np.abs(C_t)
        train_sampler = BoostingSampler(train_ds_index, C_tp, batch_size)
        train_loader = torch.utils.data.DataLoader(train_ds_index, sampler=train_sampler, batch_size=batch_size)

        # TODO: This hopefully commenting this out fixes memory problem between weak learners.
        train_loader_default = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=100, shuffle=False)
        
        h_i = weakLearner()
        h_i.fit(train_loader, test_loader, C_t, adv=adv, maxIt=maxIt, predictionWeights=predictionWeights)

        _, predictions, _ = pytorch_predict(h_i.model, train_loader_default, torch.device('cuda')) #y_true, y_pred, y_pred_prob
        print("targets:", train_ds_index.targets)
        print("Full Accuracy: ", (predictions == train_ds_index.targets.numpy()).astype(int).sum()/len(predictions))


        a = -C_t[np.arange(m), predictions].sum()
        b = fexp.sum()
        
        delta_t = a / b
        alpha = 1/2*np.log((1+delta_t)/(1-delta_t))
        print("Alpha: ", alpha)
        
        f[np.arange(m), predictions] += alpha
        
        weakLearners.append(h_i)
        weakLearnerWeights.append(alpha)
        
    return weakLearners, weakLearnerWeights


def SchapireWongMulticlassBoostingMemoryLess(weakLearner, numLearners, dataset, advDelta=0, alphaTol=1e-5, adv=True, maxIt=float("inf"), predictionWeights=False, epochs=1):
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
    # print("m, k: ", m, k)

    batch_size = 100
    test_loader = torch.utils.data.DataLoader(test_ds_index, batch_size=200, shuffle=False)
    for data in test_loader:
        test_X = data[0]
        test_y = data[1]

    train_loader_default = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=100, shuffle=False)

    weakLearners = []
    weakLearnerWeights = []

    f = np.zeros((m, k))
    
    val_accuracies = []

    for t in range(numLearners):
        print("-"*100)
        print("Training {}th weak learning".format(t))
        C_t = np.zeros((m, k))
        fcorrect = f[np.arange(m), train_ds_index.targets]
        fexp = np.exp(f - fcorrect[:,None])
        C_t = fexp.copy()
        fexp[np.arange(m), train_ds_index.targets] = 0
        C_t[np.arange(m), train_ds_index.targets] = -np.sum(fexp, axis=1)

        C_tp = np.abs(C_t)
        train_sampler = BoostingSampler(train_ds_index, C_tp, batch_size)
        train_loader = torch.utils.data.DataLoader(train_ds_index, sampler=train_sampler, batch_size=batch_size)
        # Trying out the old train_loader to see if dis works

        # TODO: This hopefully commenting this out fixes memory problem between weak learners.
        train_loader_default = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=100, shuffle=False)
        
        h_i = weakLearner()
        h_i.fit(train_loader, test_loader, C_t, adv=adv, maxIt=maxIt, predictionWeights=predictionWeights, epochs=epochs)

        _, predictions, _ = pytorch_predict(h_i.model, train_loader_default, torch.device('cuda')) #y_true, y_pred, y_pred_prob
        print("targets:", train_ds_index.targets)
        print("Full Accuracy: ", (predictions == train_ds_index.targets.numpy()).astype(int).sum()/len(predictions))

        a = -C_t[np.arange(m), predictions].sum()
        b = fexp.sum()
        
        delta_t = a / b
        alpha = 1/2*np.log((1+delta_t)/(1-delta_t))
        print("Alpha: ", alpha)
        
        f[np.arange(m), predictions] += alpha

        model_path = f'./models/cifar10_wl_{t}.pth'
        torch.save(h_i.model.state_dict(), model_path)
        weakLearners.append(model_path)
        del h_i
        del predictions
        torch.cuda.empty_cache()
        weakLearnerWeights.append(alpha)
        
        
        ensemble = Ensemble(weakLearners, weakLearnerWeights)
        test_predictions = ensemble.schapirePredict(test_X.to(torch.device('cuda:0')), 10)
        new_val_accuracy = (test_predictions == test_y.numpy()).astype(int).sum()/len(test_predictions)
        print("After newest WL score is: ", new_val_accuracy)
        val_accuracies.append(new_val_accuracy)
        
    return weakLearners, weakLearnerWeights, val_accuracies

class Ensemble:
    def __init__(self, weakLearners, weakLearnerWeights):
        """
        """
        self.weakLearners = weakLearners
        self.weakLearnerWeights = weakLearnerWeights

    def getWLPredictionsString(self, X, k):
        T = len(self.weakLearners)
        wLPredictions = []
        for i in range(T):
            model = PreActResNet18()
            model.load_state_dict(torch.load(self.weakLearners[i]))
            model = model.to(torch.device('cuda:0'))
            model.eval()
            prediction = model(X).argmax(axis = 1)
            wLPredictions.append(prediction)
        return wLPredictions


    def schapirePredict(self, X, k):
        wLPredictions = None

        predictions = np.zeros(len(X))
        T = len(self.weakLearners)
        
        if isinstance(self.weakLearners[0], str):
            wLPredictions = self.getWLPredictionsString(X, k)
        else:
            wLPredictions = [self.weakLearners[i].predict(X).argmax(axis=1) for i in range(T)]

        
        for i in range(len(X)):
            F_Tx =[]
            for l in range(k):
                F_Tx.append(sum([self.weakLearnerWeights[t] * (1 if wLPredictions[t][i] == l else 0) for t in range(T)]))
        
            predictions[i] = np.argmax(np.array(F_Tx))
        return predictions


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

def runMemlessCifarBoosting(numWL, maxIt, epochs):
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ]))

    from datetime import datetime
    # train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=200, shuffle=False)
    for data in test_loader:
        val_X = data[0]
        val_y = data[1]
        break

    t0 = datetime.now()
    smallerdataset = torch.utils.data.Subset(datasets.CIFAR10, range(5000))
    # instance.fit
    # class.fit
    # from pytorch_memlab import LineProfiler

    wl, wlweights, val_accuracies = SchapireWongMulticlassBoostingMemoryLess(WongNeuralNetCIFAR10, numWL, datasets.CIFAR10, advDelta=0, alphaTol=1e-10, adv=False, maxIt=maxIt, predictionWeights=False, epochs=1)


    # for data in test_loader:
    #   val_X = data[0]
    #   val_y = data[1]
    #   break


    ensemble = Ensemble(wl, wlweights)

    predictions = ensemble.schapirePredict(val_X.to(torch.device('cuda:0')), 10)
    print("Finished With: ", (predictions == val_y.numpy()).astype(int).sum()/len(predictions))
    print("In ", (datetime.now()-t0).total_seconds(), " s")
    return wl, wlweights, val_accuracies
        
# def plot_accuracies(val_accuracies):
#     num_weak_learners = len(val_accuracies)
#     plt, ax = plt.subplots()
    