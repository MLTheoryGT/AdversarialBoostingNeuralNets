import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
cuda = torch.device('cuda:0')
import torch.cuda as cutorch
import gc


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.linear1 = nn.Linear(32*7*7,100)
        self.linear2 = nn.Linear(100, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return x

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


from AdversarialAttacks import attack_fgsm, attack_pgd
    
    
class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

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
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
  
    def plot_val_loss(self, path=None):
        plt.subplots()
        plt.plot(self.val_checkpoints, self.losses['val'])
        plt.xlabel(self.xlabel)
        plt.ylabel('Validation loss')
        plt.title('Validation loss')
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
    
    def plot_val_accuracies(self, path=None):
        plt.subplots()
        plt.plot(self.val_checkpoints, self.accuracies['val'])
        plt.xlabel(self.xlabel)
        plt.ylabel('Validation accuracy')
        plt.title('Validation accuracy')
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
    
    def plot_train_accuracies(self, path=None):
        plt.subplots()
        plt.plot(self.train_checkpoints, self.accuracies['train'])
        plt.xlabel(self.xlabel)
        plt.ylabel('Training accuracy')
        plt.title('Training accuracy')
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
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
  
    def plot_adversarial_accuracies(self, path=None):
        # f, ax = plt.subplots(12)
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        plt.subplots()
        for attack_name in self.losses:
            if attack_name not in ['val', 'train']:
                if len(self.accuracies[attack_name]) == 0: continue
                for i in range(len(self.attack_eps)):
                    plt.plot(self.val_checkpoints, self.accuracies[attack_name][i], color = colors[i], label = 'Epsilon = {}'.format(self.attack_eps[i]))

                plt.legend()
                plt.xlabel(self.xlabel)
                plt.ylabel("Accuracy")
                plt.title(f"Adversarial accuracy ({attack_name})")
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
    
    def plot_memory_usage(self):
        f, ax = plt.subplots()
        plt.plot(self.iters, self.memory_usage)
        plt.xlabel(self.xlabel)
        plt.ylabel("Total memory usage")
        plt.title("Memory usage over number of iterations")

class Validator():
    
    def predict(self, X):
        # Must be implemented by classes that inherit Validator
        pass
    
    def calc_accuracies(self, X, y, data_type='val', val_attacks=[], alpha = 1e-2, attack_iters = 50, restarts = 10, y_pred=None):
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
#             print("about to attack")
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
                    delta = attack_pgd(X, y, epsilon, self.predict, alpha, attack_iters, restarts)
                X_adv = X + delta
                y_pred = self.predict(X_adv).detach()
                accuracy = (y_pred.max(1)[1] == y).sum().item() / X_adv.shape[0]
                loss = F.cross_entropy(y_pred, y)
                losses[attack].append(loss.item())
                accuracies[attack].append(accuracy)
                del delta
                del X_adv
                del loss
                del y_pred
        torch.cuda.empty_cache()
        return losses, accuracies
    
    def record_accuracies(self, progress, val_X = None, val_y = None, train_X=None, train_y=None, val_attacks=[]):
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
            losses, accuracies = self.calc_accuracies(val_X, val_y, data_type='val', val_attacks = val_attacks)
            self.losses['val'].append(losses['val'])
            self.accuracies['val'].append(accuracies['val'])
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
    
    

class BaseNeuralNet(MetricPlotter, Validator):
    
    def __init__(self, netOfChoice):
        MetricPlotter.__init__(self, 'Total samples')
        Validator.__init__(self)
        self.model = netOfChoice().to(cuda)

   # overrides the predict in Validator
    def predict(self, X):
        return self.model(X)
