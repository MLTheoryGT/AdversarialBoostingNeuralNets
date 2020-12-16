import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
cuda = torch.device('cuda:0')


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


class BaseNeuralNet():
    
    def __init__(self, netOfChoice):
        self.model = netOfChoice().to(cuda)
        self.losses = []
        self.iters = []
        self.val_losses = []
        self.val_accuracies = []
        self.adversarial_losses = []
        self.adversarial_accuracies = []
        self.fgsm_losses = []
        self.fgsm_accuracies = []
        self.pgd_losses = []
        self.pgd_accuracies = []
        self.attack_epsilons=[0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
        for i in range(len(self.attack_epsilons)):
            self.adversarial_losses.append([])
            self.adversarial_accuracies.append([])
            self.fgsm_losses.append([])
            self.fgsm_accuracies.append([])
            self.pgd_losses.append([])
            self.pgd_accuracies.append([])
            self.memory_usage = []
            self.train_memory = []
            self.val_memory = []

    def plot_train(self, batchSize):
        plt.plot(np.arange(0, len(self.losses*batchSize), batchSize), self.losses)
        plt.xlabel('Iterations')
        plt.ylabel('Training loss')
        plt.title('Training loss vs iterations')
        plt.show()
  
    def plot_val(self, batchSize):
        plt.plot(np.arange(0, len(self.val_losses*batchSize), batchSize), self.val_losses)
        plt.xlabel('Iterations')
        plt.ylabel('Validation loss')
        plt.title('Validation loss vs iterations')
        plt.show()
    
    def plot_val_accuracies(self, batchSize):
        plt.plot(np.arange(0, len(self.val_accuracies*batchSize), batchSize), self.val_accuracies)
        plt.xlabel('Iterations')
        plt.ylabel('Validation accuracy')
        plt.title('Validation accuracy vs iterations')
        plt.show()
  
    def plot_adversarial_accuracies(self):
        # f, ax = plt.subplots(12)
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        for i in range(len(self.attack_epsilons)):
            plt.plot(self.iters, self.adversarial_accuracies[i], color = colors[i], label = 'Epsilon = {}'.format(self.attack_epsilons[i]))
            # print("epsilon:", self.attack_epsilons[i])
            # print("iters")
            # print(self.iters)
            # print("iters length:", len(self.iters))
            # print("adversarial accuracies")
            # print(self.adversarial_accuracies[i])
            # print("accuracies length:", len(self.adversarial_accuracies[i]))

        plt.legend()
        plt.xlabel("Number of training iterations")
        plt.ylabel("Accuracy")
        plt.title("Adversarial accuracy over number of iterations")
    
    def plot_memory_usage(self):
        f, ax = plt.subplots()
        plt.plot(self.iters, self.memory_usage)
        plt.xlabel("Number of training iterations")
        plt.ylabel("Total memory usage")
        plt.title("Memory usage over number of iterations")
  
    def validation(self, X, y, attack=attack_fgsm, alpha = 1e-2, attack_iters = 50, restarts = 10, y_pred=None):
#         print("in validation")
        total_loss = 0
        cnt = 0
        val_accuracy = None


        # accuracy on clean examples
        if y_pred is None:
            y_pred = self.model(X)
        val_accuracy = (y_pred.max(1)[1] == y).sum().item() / X.shape[0]
        loss = F.cross_entropy(y_pred, y).item()
        cnt += 1
        total_loss += loss
        
        # accuracy on adversarial examples
        epsilons = self.attack_epsilons

        # TODO: modify the below block when I want to also test PGD
        adversarial_losses = None
        adversarial_accuracies = None
        if attack == attack_fgsm:
            adversarial_losses = self.fgsm_losses
            adversarial_accuracies = self.fgsm_accuracies
        else:
            adversarial_losses = self.pgd_losses
            adversarial_accuracies = self.pgd_accuracies
    
        for i in range(len(epsilons)):
            epsilon = epsilons[i]
            delta = None
            if attack == attack_fgsm:
                delta = attack_fgsm(X, y, epsilon, self.model)
            else:
                # assuming attack == attack_pgd
                delta = attack_pgd(X, y, epsilon, self.model, alpha, attack_iters, restarts)
            X_adv = X + delta
            y_pred = self.model(X_adv).detach()
            accuracy = (y_pred.max(1)[1] == y).sum().item() / X_adv.shape[0]
            loss = F.cross_entropy(y_pred, y)
            self.adversarial_losses[i].append(loss)
            self.adversarial_accuracies[i].append(accuracy)
    
        return total_loss, val_accuracy

    def predict(self, X):
        return self.model(X)
