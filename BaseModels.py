import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
cuda = torch.device('cuda:0')
import torch.cuda as cutorch
import gc

from AdversarialAttacks import attack_fgsm, attack_pgd

class MetricPlotter():
    
    def __init__(self, xlabel):
        self.val_checkpoints = [] # this is the x value for plotting
        self.train_checkpoints = []
        self.xlabel = xlabel

        self.losses = {'train': [], 'val': [], 'attack_fgsm': [], 'attack_pgd': []}
        self.accuracies = {'train': [], 'val': [], 'attack_fgsm': [], 'attack_pgd': []}
        
#         for i in range(len(self.attack_eps)):
#             self.memory_usage = []
#             self.train_memory = []
#             self.val_memory = []

    def plot_train_loss(self, path=None):
        plt.subplots()
        plt.plot(self.train_checkpoints, self.losses['train'])
        plt.xlabel(self.xlabel)
        plt.ylabel('Training loss')
        plt.title('Training loss')
        plt.grid()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
  
    def plot_val_loss(self, path=None):
        plt.subplots()
        plt.plot(self.val_checkpoints, self.losses['val'])
        plt.xlabel(self.xlabel)
        plt.ylabel('Validation loss')
        plt.title('Validation loss')
        plt.grid()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
    
    def plot_val_accuracies(self, path=None):
        plt.subplots()
        plt.plot(self.val_checkpoints, self.accuracies['val'])
        plt.xlabel(self.xlabel)
        plt.ylabel('Validation accuracy')
        plt.title('Validation accuracy')
        plt.grid()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
    
    def plot_train_accuracies(self, path=None):
        plt.subplots()
        plt.plot(self.train_checkpoints, self.accuracies['train'])
        plt.xlabel(self.xlabel)
        plt.ylabel('Training accuracy')
        plt.title('Training accuracy')
        plt.grid()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
        
    def plot_accuracies(self, path=None):
        plt.subplots()
        plt.plot(self.train_checkpoints, self.accuracies['train'])
        plt.plot(self.val_checkpoints, self.accuracies['val'])
        plt.xlabel(self.xlabel)
        plt.ylabel("Accuracy")
        plt.legend(["Training accuracy", "Validation accuracy"])
        plt.title("Accuracy")
        plt.grid()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
    
    def plot_loss(self, path=None):
        plt.subplots()
        plt.plot(self.train_checkpoints, self.losses['train'])
        plt.plot(self.val_checkpoints, self.losses['val'])
        plt.xlabel(self.xlabel)
        plt.ylabel("Loss")
        plt.legend(["Training loss", "Validation loss"])
        plt.title("Loss")
        plt.grid()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
  
    def plot_adversarial_accuracies(self, path=None):
        # f, ax = plt.subplots(12)
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        plt.subplots()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        for attack_name in self.losses:
            if attack_name not in ['val', 'train']:
                if len(self.accuracies[attack_name]) == 0: continue
                for i in range(len(self.attack_eps)):
                    plt.plot(self.val_checkpoints, self.accuracies[attack_name][i], color = colors[i], label = 'Epsilon = {}'.format(self.attack_eps[i]))

                plt.legend()
                plt.xlabel(self.xlabel)
                plt.ylabel("Accuracy")
                plt.title(f"Adversarial accuracy ({attack_name})")
        plt.grid()
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
    
    def plot_memory_usage(self):
        f, ax = plt.subplots()
        plt.plot(self.iters, self.memory_usage)
        plt.xlabel(self.xlabel)
        plt.ylabel("Total memory usage")
        plt.grid()
        plt.title("Memory usage over number of iterations")

class Validator():
    
    def predict(self, X):
        # Must be implemented by classes that inherit Validator
        pass
    
    def calc_accuracies(self, X, y, data_type='val', val_attacks=[], alpha=1e-2, attack_iters=20, restarts=1, y_pred=None):
#         print("in validation")
        
        losses = {} # (non_adv, adv)
        accuracies = {}
                
                

        # accuracy on clean examples
        if y_pred is None:
            y_pred = self.predict(X).detach() # did this to debug memory issues (1 / 18)
            y_pred.requires_grad=False
        
        accuracy = (y_pred.max(1)[1] == y).sum().item() / X.shape[0]
        loss = F.cross_entropy(y_pred, y)
        
        losses[data_type] = loss.item()
        del loss
        accuracies[data_type] = accuracy
        
        # accuracy on adversarial examples
        epsilons = self.attack_eps
        
#         print("self.attack_eps", self.attack_eps)
        # TODO: modify the below block when I want to also test PGD
        for attack in val_attacks:
#             print("about to attack",attack)
            losses[attack] = []
            accuracies[attack] = []

            for i in range(len(self.attack_eps)):
#                 print("epsilon:", epsilons[i])
                epsilon = epsilons[i]
                delta = None
                if attack == attack_fgsm:
                    delta = attack_fgsm(X, y, epsilon, self.predict)
                else:
                    # assuming attack == attack_pgd
                    delta = attack_pgd(X, y, epsilon, self.predict, attack_iters=attack_iters, restarts=restarts)
                X_adv = X + delta
                y_pred = self.predict(X_adv).detach()
                accuracy = (y_pred.max(1)[1] == y).sum().item() / X_adv.shape[0]
#                 print("acc: ", accuracy)
                loss = F.cross_entropy(y_pred, y)
                losses[attack].append(loss.item())
                accuracies[attack].append(accuracy)
                del delta
                del X_adv
                del loss
                del y_pred
        torch.cuda.empty_cache()
        return losses, accuracies
    
    def record_accuracies(self, progress, val_X = None, val_y = None, train_X=None, train_y=None, attack_iters=20, restarts=1, val_attacks=[]):
#         self.memory_usage.append(cutorch.memory_allocated(0))
#         print("memory:", cutorch.)

        # seems that at this point there are already NN params, but right before calling this function there are no NN params???? (1/18)
#         if len(self.weakLearners) == 2 and self.xlabel == 'Number of weak learners':
#             for obj in gc.get_objects():
#                 try:
#                     if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#                         print(type(obj), obj.size())
#                 except:
#                     pass
            
        if train_X is not None and train_y is not None:
            self.train_checkpoints.append(progress)
            losses, accuracies = self.calc_accuracies(train_X, train_y, data_type='train')
            self.losses['train'].append(losses['train'])
            self.accuracies['train'].append(accuracies['train'])

        if val_X is not None and val_y is not None:
            self.val_checkpoints.append(progress)
            losses, accuracies = self.calc_accuracies(val_X, val_y, data_type='val', val_attacks = val_attacks, attack_iters=attack_iters, restarts=restarts)
#             print("losses, accs", losses, accuracies)
            self.losses['val'].append(losses['val'])
            self.accuracies['val'].append(accuracies['val'])
#             print("Val accuracy:", accuracies['val'])
#             print("losses", losses)
#             print("self.losses", self.losses)
            for attack in losses:
#                 print("in loop attack: ", attack)
                
                if type(attack) != str:
                    if len(self.losses[attack.__name__]) == 0:
                        self.losses[attack.__name__] = [[] for i in range(len(self.attack_eps))]
                        self.accuracies[attack.__name__] = [[] for i in range(len(self.attack_eps))]
                    for i in range(len(self.attack_eps)):
                        self.losses[attack.__name__][i].append(losses[attack][i])
                        self.accuracies[attack.__name__][i].append(accuracies[attack][i])
            print("Progress: %d,  val accuracy: %.4f" %(progress, self.accuracies['val'][-1]))
            print("PGD accuracy:", self.accuracies['attack_pgd'])
    
    

class BaseNeuralNet(MetricPlotter, Validator):
    
    def __init__(self, netOfChoice):
        MetricPlotter.__init__(self, 'Total samples')
        Validator.__init__(self)
        self.model = netOfChoice().to(cuda)

   # overrides the predict in Validator
    def predict(self, X):
        return self.model(X)
