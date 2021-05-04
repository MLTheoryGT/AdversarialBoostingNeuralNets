# import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar10_mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
cifar10_std = torch.tensor(cifar10_std).view(3,1,1).cuda()
cifar10_upper_limit = ((1 - cifar10_mu)/ cifar10_std)
cifar10_lower_limit = ((0 - cifar10_mu)/ cifar10_std)

cifar100_mean = (0.507, 0.487, 0.441) 
cifar100_std = (0.267, 0.256, 0.276)
cifar100_mu = torch.tensor(cifar100_mean).view(3,1,1).cuda()
cifar100_std = torch.tensor(cifar100_std).view(3,1,1).cuda()
cifar100_upper_limit = ((1 - cifar100_mu)/ cifar100_std)
cifar100_lower_limit = ((0 - cifar100_mu)/ cifar100_std)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

"""
    Takes in a pytorch dataset object and returns train/test datasets after transformations
"""
def applyDSTrans(dataset):
    train_transforms = []
    test_transforms = []
    
    if dataset == datasets.CIFAR10 or dataset==datasets.CIFAR100:
        for elt in [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]:
            train_transforms.append(elt)
    
    tens = transforms.ToTensor()
    train_transforms.append(tens)
    test_transforms.append(tens)
    
    if dataset == datasets.CIFAR10:
        norm = transforms.Normalize(cifar10_mean, cifar10_std)
        train_transforms.append(norm)
        test_transforms.append(norm)

    if dataset == datasets.CIFAR100:
        norm = transforms.Normalize(cifar100_mean, cifar100_std)
        train_transforms.append(norm)
        test_transforms.append(norm)
                
    assert(len(train_transforms) == 4)
    assert(len(test_transforms) == 2)

    train_ds_index = dataset_index('./data', train=True, download=True, transform=transforms.Compose(train_transforms))
    test_ds_index = dataset_index('./data', train=False, download=True, transform=transforms.Compose(test_transforms))

    return train_ds_index, test_ds_index

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

def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    scaler = torch.cuda.amp.GradScaler()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            with torch.cuda.amp.autocast():
                output = model(X + delta)
                index = torch.where(output.max(1)[1] == y)
                if len(index[0]) == 0:
                    break
                loss = F.cross_entropy(output, y)
            if opt is not None:
#                 with torch.cuda.amp.scale_loss(loss, opt) as scaled_loss:
#                     scaled_loss.backward()
                scaler.scale(loss).backward()
                
            else:
                scaler.scale(loss).backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n
