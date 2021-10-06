from torchvision import datasets, transforms
from Architectures import PreActResNet18
import torch
from autoattack import AutoAttack
import numpy as np

nn = PreActResNet18()
nn.load_state_dict(torch.load("./models/wong/cifar10/baseline/cifar_model_weights_15_epochs.pth"))
nn.eval()
nn.cuda()

test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False)
test_ds.targets = torch.tensor(np.array(test_ds.targets))

xys = [(x, y) for (x, y) in test_loader]
l = [x for (x, y) in xys]
x_test = torch.cat(l, 0)
#     x_test = torch.cat(l[:num_batches], 0) for doing less than the whole test set
l = [y for (x, y) in xys]
y_test = torch.cat(l, 0)
    
adversary = AutoAttack(nn.predictUnnormalizedDataCIFAR10, norm='Linf', eps=0.031, version='standard', log_path="./results/AutoAttack/Wong1NNBaseline/1NNBaseline.txt")
x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=512)