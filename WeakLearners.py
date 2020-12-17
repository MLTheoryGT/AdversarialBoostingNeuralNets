from BaseModels import BaseNeuralNet, PreActResNet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.cuda as cutorch
cuda = torch.device('cuda:0')

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
    
    
class WongNeuralNet(BaseNeuralNet):
    """
        Note: does not avide by the same maxSample that the other weak learners do since this isn't actually a boosting class
    """

    def __init__(self):
        super().__init__(Net)
  
    def fit(self, train_loader, test_loader, alpha = 0.375, epochs = 10, lr_max = 5e-3):
        print("Starting fit")
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
            # change the below to include indices:
            # https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
            for i, data in enumerate(train_loader):
                # print("minibatch: ", i)
                lr = lr_schedule(epoch + (i+1)/len(train_loader))
                self.optimizer.param_groups[0].update(lr=lr)
                # if cnt % 1000 == 1:
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
                # print("regular x shape", X.shape)
                y = data[1].cuda()
                self.batchUpdate(X, y, alpha = alpha)
                del X
                del y
                torch.cuda.empty_cache()

    def batchUpdate(self, X, y, epochs = 1, epsilon = 0.3, alpha = 0.375):
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
        self.loss = loss
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

class BoostedWongNeuralNet(BaseNeuralNet):

    def __init__(self):
        super().__init__(Net)

    def fit(self, train_loader, test_loader, C, alpha = 0.375, epochs = 1, lr_max = 5e-3, adv=True, maxSample=None, predictionWeights=False):
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

        # test_set, val_set = torch.utils.data.random_split(test_loader.dataset, [9000, 1000])
        for epoch in range(epochs):
            print("Epoch:", epoch)
            for i, data in enumerate(train_loader):
                currSamples += train_loader.batch_size
                # print("In epoch: ", i)
                lr = lr_schedule(epoch + (i+1)/len(train_loader))
                self.optimizer.param_groups[0].update(lr=lr)

                # if i == 0:
                # print("Start with: ", self.validation(val_X, val_y))
                
                if i % 10 == 1:
                    # print("memory usage:", cutorch.memory_allocated(0))
                    self.memory_usage.append(cutorch.memory_allocated(0))
                    self.iters.append(i)
                    val_loss, val_accuracy = self.validation(val_X, val_y)
                    self.val_losses.append(val_loss)
                    self.val_accuracies.append(val_accuracy)
                    self.losses.append(self.loss.item())

                X = data[0].cuda()
                y = data[1].cuda()
                indices = data[2].cuda()
                if currSamples > maxSample:
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
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
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
    

    
    
class WongNeuralNetCIFAR10(BaseNeuralNet):
    def __init__(self):
        super().__init__(PreActResNet18)
    
    # parser.add_argument('--batch-size', default=128, type=int)
    # parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    # parser.add_argument('--epochs', default=15, type=int)
    # parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    # parser.add_argument('--lr-min', default=0., type=float)
    # parser.add_argument('--lr-max', default=0.2, type=float)
    # parser.add_argument('--weight-decay', default=5e-4, type=float)
    # parser.add_argument('--momentum', default=0.9, type=float)
    # parser.add_argument('--epsilon', default=8, type=int)
    # parser.add_argument('--alpha', default=10, type=float, help='Step size')
    # parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
    #     help='Perturbation initialization method')
    # parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    # parser.add_argument('--seed', default=0, type=int, help='Random seed')
    # parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    # parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
    #     help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    # parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
    #     help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    # parser.add_argument('--master-weights', action='store_true',
    #     help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    
    def fit(self, train_loader, test_loader, C=None, epochs=100, lr_schedule="cyclic", lr_min=0, lr_max=0.2, weight_decay=5e-4, early_stop=True,
                  momentum=0.9, epsilon=8, alpha=10, delta_init="random", seed=0, opt_level="O2", loss_scale=1.0, out_dir="WongNNCifar10",
                  maxSample = None, adv=False, predictionWeights=None):
      
        from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
        attack_pgd, evaluate_pgd, evaluate_standard)
        import time
        # args = get_args()
        # if not os.path.exists(args.out_dir):
        #     os.mkdir(args.out_dir)
        # logfile = os.path.join(args.out_dir, 'output.log')
        # if os.path.exists(logfile):
        #     os.remove(logfile)
        
        scaler = torch.cuda.amp.GradScaler()

        # logging.basicConfig(
        #     format='[%(asctime)s] - %(message)s',
        #     datefmt='%Y/%m/%d %H:%M:%S',
        #     level=logging.INFO,
        #     filename=os.path.join(args.out_dir, 'output.log'))
        # logger.info(args)
        # print("memory usage start:", cutorch.memory_allocated(0))
        val_X = None
        val_y = None
        for data in test_loader:
            val_X = data[0].to(cuda)
            val_y = data[1].to(cuda)
            break


        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        epsilon = (epsilon / 255.) / std
        alpha = (alpha / 255.) / std
        pgd_alpha = (2 / 255.) / std

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

        print("adv:", adv)
        # Training
        prev_robust_acc = 0.
        start_train_time = time.time()
#         logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
        cur_iteration = 0
        done = False
        for epoch in range(epochs):
            print("Epoch %d"%(epoch))
            start_epoch_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            for i, data in enumerate(train_loader):
                cur_iteration += train_loader.batch_size
                if maxSample and cur_iteration >= maxSample:
                    done = True
                    break
                X, y = data[0].cuda(), data[1].cuda()
                if i % 100 == 99:
#                     print("Iteration: ", i)
#                     print("memory usage:", cutorch.memory_allocated(0))
                    self.memory_usage.append(cutorch.memory_allocated(0))
                    # print("memory usage:", cutorch.memory_allocated(0))
                    self.iters.append(i)
                    
                    val_loss, val_accuracy = self.validation(val_X, val_y, y_pred=model(val_X))
                    # print("memory usage:", cutorch.memory_allocated(0))
                    self.val_losses.append(val_loss)
                    self.val_accuracies.append(val_accuracy)
                    self.losses.append(self.loss.item())
                    print("it %d val accuracy: %.4f" %(i, self.val_accuracies[-1]))
                if i == 0:
                    first_batch = (X, y)

                output = None

                # print(X.shape)

                if adv:
                    if delta_init != 'previous':
                        delta = torch.zeros_like(X).cuda()

                    if delta_init == 'random':
                        for j in range(len(epsilon)):
                            delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                        delta.data = clamp(delta, lower_limit - X, upper_limit - X)

                    delta.requires_grad = True
                    with torch.cuda.amp.autocast():
                        output = model(X + delta[:X.size(0)])
                        loss = F.cross_entropy(output, y)
#                     with amp.scale_loss(loss, opt) as scaled_loss:
#                         scaled_loss.backward()
                    scaler.scale(loss).backward()

                    grad = delta.grad.detach()
                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                    delta = delta.detach()
                    # print("memory usage:", cutorch.memory_allocated(0))
                    output = model(X + delta[:X.size(0)])
                else:
                    output = model(X)
                # print("memory usage:", cutorch.memory_allocated(0))
                loss = criterion(output, y)
                self.loss = loss
                opt.zero_grad()
#                 with torch.cuda.amp.scale_loss(loss, opt) as scaled_loss:
#                     scaled_loss.backward()
                scaler.scale(loss).backward()
#                 opt.step()
                scaler.step(opt)
                scaler.update()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
                scheduler.step()
                del X
                del y
                torch.cuda.empty_cache()

            if early_stop:
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
        torch.cuda.empty_cache()
        # logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
        # epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
        train_time = time.time()
        # if not early_stop:
            # best_state_dict = model.state_dict()
        # torch.save(best_state_dict, os.path.join(out_dir, 'model.pth'))
        # logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)


        # # Evaluation
        # model_test = PreActResNet18().cuda()
        # model_test.load_state_dict(best_state_dict)
        # model_test.float()
        # model_test.eval()


        # pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
        # test_loss, test_acc = evaluate_standard(test_loader, model_test)


        # logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        # logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
    
    def batchUpdate(X, y, delta):
        grad = delta.grad.detach()
        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
        delta = delta.detach()
        # print("memory usage:", cutorch.memory_allocated(0))
        output = model(X + delta[:X.size(0)])
        # print("memory usage:", cutorch.memory_allocated(0))
        loss = criterion(output, y)
        self.loss = loss
        opt.zero_grad()
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
        opt.step()
        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        scheduler.step()
        del X
        del y
        torch.cuda.empty_cache()

    def batchUpdateNonAdv(X, y):
        self.model.train()
        N = X.size()[0]
        
        optimizer = self.optimizer

        # Prints
        # print("In Batch Update Y.shape: ", y.shape)

        # compute delta (perturbation parameter)
        optimizer.zero_grad()

        # update network parameters
        output = self.model(X)
        loss = F.cross_entropy(output, y)
        self.loss = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def predict(self, X):
        return self.model(X)