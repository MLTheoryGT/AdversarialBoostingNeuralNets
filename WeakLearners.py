from BaseModels import BaseNeuralNet, PreActResNet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.cuda as cutorch
cuda = torch.device('cuda:0')
from datetime import datetime

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
    

class WongNeuralNetMNIST(BaseNeuralNet):

    def __init__(self, attack_eps=[], train_eps=0.3):
        super().__init__(Net)
        self.attack_eps = attack_eps
        self.train_eps = train_eps

    def fit(self, train_loader, test_loader, C, alpha = 0.375, epochs = 1, lr_max = 5e-3, adv_train=True, val_attacks = [], maxSample=None, predictionWeights=False):
        val_X = None
        val_y = None
        for data in test_loader:
            val_X = data[0].cuda()
            val_y = data[1].cuda()
            break

        # optimizer here
        self.optimizer = torch.optim.Adam(self.model.parameters())
        lr_schedule = lambda t: np.interp([t], [0, epochs * 2//5, epochs], [0, lr_max, 0])[0]
        currSamples = 0
        
#         print(f"adv_train: {adv_train}")

        # test_set, val_set = torch.utils.data.random_split(test_loader.dataset, [9000, 1000])
        for epoch in range(epochs):
            print("Epoch:", epoch)
            for i, data in enumerate(train_loader):
                currSamples += train_loader.batch_size
                lr = lr_schedule(epoch + (i+1)/len(train_loader))
                self.optimizer.param_groups[0].update(lr=lr)
                
                if i % 10 == 1:
                    self.record_accuracies(currSamples, val_X=val_X, val_y=val_y, val_attacks=val_attacks)

                X = data[0].cuda()
                y = data[1].cuda()
                indices = data[2].cuda()
                if currSamples > maxSample:
                    del X
                    del y
                    self.record_accuracies(currSamples, val_X=val_X, val_y=val_y, val_attacks=val_attacks)
                    torch.cuda.empty_cache()
                    return
                if adv_train:
                    loss = self.batchUpdate(X, y, alpha = alpha)
                    self.losses['train'].append(loss.item())
                else:
                    # print("MB(%d), "%(i), end="")
                    loss = self.batchUpdateNonAdv(X, y, indices, C, alpha=alpha, predictionWeights=predictionWeights)
                    self.losses['train'].append(loss.item())
                    del X
                    del y
                self.train_checkpoints.append(currSamples)
                
        # print("Escaped epoch")
        print("WL has validation accuracy", self.accuracies['val'][-1])
        print("WL has loss", self.losses['train'][-1])

        torch.cuda.empty_cache()
    
    def batchUpdate(self, X, y, alpha = 0.375):
        epsilon = self.train_eps
#         print("doing adversarial batchUpdate")
        self.model.train()
        N = X.size()[0]

        optimizer = self.optimizer

        # compute delta (perturbation parameter)
        # taken from https://github.com/locuslab/fast_adversarial/blob/master/MNIST/train_mnist.py
        optimizer.zero_grad()
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).to(cuda)
        delta.requires_grad = True
        output = self.model(X + delta)
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data = torch.max(torch.min(1-X, delta.data), 0-X)
        delta = delta.detach()


        # update network parameters
        output = self.model(torch.clamp(X + delta, 0, 1))
        loss = F.cross_entropy(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def batchUpdate_phase3(self, X, y, C, alpha = 0.375):
#         print("starting batchUpdate")
        epsilon = self.train_eps
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
#         print("In Batch Update worst_k.shape: ", worst_k.shape)
        # compute L(x_i, k_i) for all examples
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
        delta.requires_grad = True
        output = self.model(X + delta)
        loss = F.cross_entropy(output, worst_k)
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
        return loss
  
    def batchUpdateNonAdv(self, X, y, indices, C, epochs = 1, alpha = 0.375, predictionWeights=False):
        epsilon = self.train_eps
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, X):
        return self.model(X)
    

    
    
class WongNeuralNetCIFAR10(BaseNeuralNet):
    def __init__(self, attack_eps = [], train_eps=8):
        super().__init__(PreActResNet18)
        self.train_eps = train_eps
        self.attack_eps = attack_eps
    
    def fit(self, train_loader, test_loader, C=None, epochs=100, lr_schedule="cyclic", lr_min=0, lr_max=0.2, weight_decay=5e-4, early_stop=True,
                  momentum=0.9, epsilon=8, alpha=10, delta_init="random", seed=0, opt_level="O2", loss_scale=1.0, out_dir="WongNNCifar10",
                  maxSample = None, adv_train=False, val_attacks = [], predictionWeights=None):
      
        from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
        attack_pgd, evaluate_pgd, evaluate_standard)
        import time
        
        scaler = torch.cuda.amp.GradScaler()

        val_X = None
        val_y = None
        for data in test_loader:
            val_X = data[0].to(cuda)
            val_y = data[1].to(cuda)
            break


        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        epsilon = (self.train_eps / 255.) / std
        alpha = (alpha / 255.) / std
        pgd_alpha = (2 / 255.) / std
        
        if delta_init == 'previous':
            delta = torch.zeros(train_loader.batch_size, 3, 32, 32).cuda()

        model = self.model
        # model = PreActResNet18().cuda()
        model.train()
        # print("memory usage after init model:", cutorch.memory_allocated(0))

        opt = torch.optim.SGD(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=weight_decay)
        amp_args = dict(opt_level=opt_level, loss_scale=loss_scale, verbosity=False)
        # if opt_level == 'O2':
        #     amp_args['master_weights'] = master_weights
#         model, opt = torch.cuda.amp.initialize(model, opt, **amp_args)
        criterion = nn.CrossEntropyLoss()

        if delta_init == 'previous':
            delta = torch.zeros(train_loader.batch_size, 3, 32, 32).cuda()

        lr_steps = epochs * len(train_loader)
        if lr_schedule == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=lr_min, max_lr=lr_max,
                step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        elif lr_schedule == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

#         print("adv_train:", adv_train)
        # Training
        prev_robust_acc = 0.
        start_train_time = time.time()
#         logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
        currSamples = 0
        done = False
        delta = None
        for epoch in range(epochs):
            print("Epoch %d"%(epoch))
            start_epoch_time = time.time()
            for i, data in enumerate(train_loader):
                currSamples += train_loader.batch_size
                if maxSample and currSamples >= maxSample:
                    done = True
                    break
                X, y = data[0].cuda(), data[1].cuda()
                if i % 100 == 99:
#                     print("about to record accs", val_attacks)
                    self.record_accuracies(currSamples, train_X = X, train_y = y, val_X=val_X, val_y=val_y , val_attacks=val_attacks)
                    
                if i == 0:
                    first_batch = (X, y)
                
                if adv_train:
                    loss = self.batchUpdate(X, y, epsilon, delta, delta_init=delta_init, alpha=alpha)
#                     self.losses['train'].append(loss.item())
                else:
                    loss = self.batchUpdateNonAdv(X, y)
#                     self.losses['train'].append(loss.item())
                
#                 self.train_checkpoints.append(currSamples)

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                del loss
                del X
                del y
                del data
                torch.cuda.empty_cache()

            if early_stop:
                a = 0 #memory-debugging
                # Check current PGD robustness of model using random minibatch
                X, y = first_batch
                pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
                with torch.no_grad():
                    output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
                robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
                if robust_acc - prev_robust_acc < -0.2:
                    break
                prev_robust_acc = robust_acc
                # best_state_dict = copy.deepcopy(model.state_dict())
            epoch_time = time.time()
            lr = scheduler.get_lr()[0]
            
            if done:
                break
        del val_X
        del val_y
        del delta
        torch.cuda.empty_cache()
        end_train_time = time.time()
        print("Total training time{}".format(end_train_time - start_train_time))
        
    def batchUpdate(self, X, y, epsilon, delta, delta_init='random', alpha=0):
        a = 0 # memory-debugging
        from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
        attack_pgd, evaluate_pgd, evaluate_standard) # delete whichever ones are unnecessary
        if delta_init != 'previous':
            delta = torch.zeros_like(X).cuda()

        if delta_init == 'random':
            for j in range(len(epsilon)):
                delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)

        delta.requires_grad = True
        with torch.cuda.amp.autocast():
            output = self.model(X + delta[:X.size(0)])
            loss = F.cross_entropy(output, y)
        scaler = torch.cuda.amp.GradScaler()
        scaler.scale(loss).backward()

        grad = delta.grad.detach()
        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
        delta = delta.detach()
        output = self.model(X + delta[:X.size(0)])
        del delta
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        del output
        del grad
        return loss
        
    def batchUpdateNonAdv(self, X, y, indices, C, epsilon = 0.3, alpha = 0.375, predictionWeights=False):
        output = model(X)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        return loss

    def predict(self, X):
        return self.model(X)
    
# class FreeNeuralNetCIFAR10(BaseNeuralNet):
#     def __init__(self, attack_eps = [], train_eps=8):
#         super().__init__(PreActResNet18)
#         self.train_eps = train_eps
#         self.attack_eps = attack_eps
    
#     '''
#     Arguments for fit:
#     parser.add_argument('--batch-size', default=128, type=int)
#     parser.add_argument('--data-dir', default='../../cifar-data', type=str)
#     parser.add_argument('--epochs', default=10, type=int, help='Total number of epochs will be this argument * number of minibatch replays.')
#     parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
#     parser.add_argument('--lr-min', default=0., type=float)
#     parser.add_argument('--lr-max', default=0.04, type=float)
#     parser.add_argument('--weight-decay', default=5e-4, type=float)
#     parser.add_argument('--momentum', default=0.9, type=float)
#     parser.add_argument('--epsilon', default=8, type=int)
#     parser.add_argument('--minibatch-replays', default=8, type=int)
#     parser.add_argument('--out-dir', default='train_free_output', type=str, help='Output directory')
#     parser.add_argument('--seed', default=0, type=int)
#     parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
#         help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
#     parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
#         help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
#     parser.add_argument('--master-weights', action='store_true',
#         help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
#     '''
#     def fit(self, train_loader, test_loader, epochs=100, lr_schedule=cyclic, lr_min=0, lr_max=0.04,
#             weight_decay=5e-4, momentum=0.9, minibatch_replays=8, seed=0, opt_level='O2', loss_scale=1.0,
#             master_weights=True,
#             maxSample = None, adv_train=False, val_attacks = [], predictionWeights=None):
        
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
        
        
#         epsilon = (self.train_eps / 255.) / std
        
#         model = self.model
#         model.train()
        
#         opt = torch.optim.SGD(model.parameters(), lr=lr_max, momentum=momentum, weight_decay=weight_decay)
#         criterion = nn.CrossEntropyLoss()
        
#         delta = torch.zeros(train_loader.batch_size, 3, 32, 32).cuda()
#         delta.requires_grad = True
        
#         lr_steps = epochs * len(train_loader) * minibatch_replays
#         if lr_schedule == 'cyclic':
#             scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=lr_min, max_lr=lr_max,
#                 step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
#         elif lr_schedule == 'multistep':
#             scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
            
#         # Training
#         start_train_time = time.time()
# #         logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
#         for epoch in range(epochs):
#             start_epoch_time = time.time()
#             train_loss = 0
#             train_acc = 0
#             train_n = 0
#             for i, (X, y) in enumerate(train_loader):
#                 X, y = X.cuda(), y.cuda()
#                 for _ in range(minibatch_replays):
#                     if adv_train:
#                         loss = self.batchUpdate(X, y)
#                     else:
#                         loss = self.batchUpdateNonAdv(X, y)
#                 train_loss += loss.item() * y.size(0)
#                 train_acc += (output.max(1)[1] == y).sum().item()
#                 train_n += y.size(0)
#             epoch_time = time.time()
#             lr = scheduler.get_lr()[0]
#             logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
#                 epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
#         train_time = time.time()
# #         torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))
# #         logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)
            
#         def batchUpdate(self, X, y):
#             # TODO
#             with torch.cuda.amp.autocast():
#                 output = model(X + delta[:X.size(0)])
#                 loss = criterion(output, y)
#             scaler = torch.cuda.amp.GradScaler()
#             opt.zero_grad()
#             scaler.scale(loss).backward()
#             with amp.scale_loss(loss, opt) as scaled_loss:
#                 scaled_loss.backward()
#             grad = delta.grad.detach()
#             delta.data = clamp(delta + epsilon * torch.sign(grad), -epsilon, epsilon)
#             delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
#             opt.step()
#             delta.grad.zero_()
#             scheduler.step()
            
#         def batchUpdateNonAdv(self, X, y):
#             output = model(X)
#             criterion = nn.CrossEntropyLoss()
#             loss = criterion(output, y)
#             return loss
#         def predict(self, X):
#             return self.model(X)
        