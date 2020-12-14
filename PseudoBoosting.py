import torch
import numpy as np
from math import ceil
from torchvision import transforms, datasets
from Boosting import BoostingSampler


def FakeSchapireMulticlassBoosting(weakLearner, numLearners, dataset, advDelta=0, alphaTol=1e-5, adv=True, maxIt=float("inf")):
    def dataset_with_indices(cls):
        """
        Modifies the given Dataset class to return a tuple data, target, index
        instead of just data, target.
        """

        def __getitem__(self, index):
            data, target = cls.__getitem__(self, index)
            return data, target, index.astype(int)

        return type(cls.__name__, (cls,), {
            '__getitem__': __getitem__,
        })
    dataset_index = dataset_with_indices(dataset)

    train_ds_index = dataset_index('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
    test_ds_index = dataset_index('./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))

    m = len(train_ds_index)
    k = len(train_ds_index.classes)
    # print("m, k: ", m, k)

    batch_size = 100
    test_loader = torch.utils.data.DataLoader(test_ds_index, batch_size=2000, shuffle=False)

    weakLearners = []
    weakLearnerWeights = []

    f = np.zeros((m, k))

    for t in range(numLearners):
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
        train_loader_default = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])),
            batch_size=100, shuffle=False)

        h_i = weakLearner()
        h_i.fit(train_loader, test_loader, C_t, adv=adv, maxIt=maxIt)

        predictions = h_i.predict(train_loader_default)
        a = -C_t[np.arange(m), predictions].sum()
        b = fexp.sum()

        delta_t = a / b
        alpha = 1/2*np.log((1+delta_t)/(1-delta_t))

        f[np.arange(m), predictions] += alpha

        weakLearners.append(h_i)
        weakLearnerWeights.append(alpha)
    return weakLearners, weakLearnerWeights


class FakeWeakLearner:

    def __init__(self, percent_correct = 0.20, dataset_size = 50000, num_labels = 10):
        self.percent_correct = percent_correct
        self.num_labels = 10
        self.dataset_size = dataset_size
        self.predictions = np.random.randint(self.num_labels, size = (self.dataset_size))
        self.correct_labels = np.random.randint(self.num_labels, size = (self.dataset_size))

    def fit(self, train_loader, test_loader, C, adv=True, maxIt = float("inf")):
        percent_correct = self.percent_correct
        all_indices = np.array([]).astype(int)
        j = 0
        for i, data in enumerate(train_loader):
            indices = data[2]
            self.correct_labels[indices] = data[1]
            # if j < 1:
            #   print("indices:", data[2], "labels:", data[1])
            #   j += 1
            all_indices = np.concatenate((all_indices, indices), axis=None)

        unique_indices = np.unique(all_indices)
        num_unique_indices = unique_indices.shape[0]
        # print("num unique:", num_unique_indices)
        correct_indices = np.random.choice(unique_indices, ceil(percent_correct * num_unique_indices))
        incorrect_indices = np.setdiff1d(unique_indices, correct_indices)

        # force some of the images observed during training to be predicted correctly

        self.predictions[correct_indices] = self.correct_labels[correct_indices]

        # force the remaining images observed during training to be predicted incorrectly (randomly assigned to an incorrect label)

        num_incorrect_indices = incorrect_indices.shape[0]
        incorrect_predictions = self.correct_labels[incorrect_indices]
        add_to_incorrect = np.random.randint(1, self.num_labels, size = (num_incorrect_indices))
        incorrect_predictions += add_to_incorrect
        incorrect_predictions = np.remainder(incorrect_predictions, self.num_labels)

        self.predictions[incorrect_indices] = incorrect_predictions

        correct_accuracy = (self.predictions[correct_indices] == self.correct_labels[correct_indices]).astype(int).sum()/len(correct_indices)
        incorrect_accuracy = (self.predictions[incorrect_indices] == self.correct_labels[incorrect_indices]).astype(int).sum()/len(incorrect_indices)
        current_accuracy = (self.predictions[unique_indices] == self.correct_labels[unique_indices]).astype(int).sum()/len(unique_indices)
        print("current accuracy:", current_accuracy)
        # print("correct accuracy:", correct_accuracy)
        # print("incorrect accuracy:", incorrect_accuracy)
        # print("correct size:", len(correct_indices))
        # print("incorrect size:", len(incorrect_indices))
        # print("unique size:", len(unique_indices))
    
    def predict(self, X):
        return self.predictions


class FakeEnsemble:
    def __init__(self, weakLearners, weakLearnerWeights):
        """
        """
        self.weakLearners = weakLearners
        self.weakLearnerWeights = np.array(weakLearnerWeights)
        self.numWeakLearners = len(weakLearners)

    def schapirePredict(self, X, k):
        T = len(self.weakLearners)
        wLPredictions = np.array([self.weakLearners[i].predict(X) for i in range(T)])
        # print("Prediciton shape: ", wLPredictions[0].shape)

        all_predictions = []

        all_predictions = np.zeros((T, len(X)))
        for learner in range(T):
            predictions = np.zeros(len(X))
            F_Tx = np.zeros((k, len(X)))
            # for l in range(k):
            #   print("equal: {}".format(wLPredictions[:learner + 1][:] == l))
            # print("total sum over class {}: {}".format(l, (wLPredictions[:learner + 1][:] == l).astype(int).sum()))
            for l in range(k):
                a = (wLPredictions[:learner + 1][:] == l)
                b = self.weakLearnerWeights[:learner + 1]
                # print(a.shape, b.shape)
                # print(wLPredictions.shape, self.weakLearnerWeights.shape)
                F_Tx[l] = np.sum((wLPredictions[:learner + 1][:] == l) * self.weakLearnerWeights[:learner + 1].reshape(learner + 1, 1), axis=0)
            predictions = np.argmax(F_Tx, axis = 0)
            # print("F_Tx", F_Tx)
            # print("predictions:", predictions)
            all_predictions[learner] = predictions

        return all_predictions, all_predictions[-1]

def plot_fake_accuracies(wl, wlweights, train_y):
    ensemble = FakeEnsemble(wl, wlweights)
    predictions, last_prediction = ensemble.schapirePredict(np.zeros((50000)), 10)
    numWeakLearners = len(predictions)
    wl_range = list(range(1, numWeakLearners + 1))
    wl_accuracies = []
    for i in range(numWeakLearners):
        print("numWeakLearners:", numWeakLearners)
        cur_prediction = predictions[i]
        print("cur_prediction:", cur_prediction)
        cur_accuracy = (cur_prediction == train_y).astype(int).sum()/len(cur_prediction)
        wl_accuracies.append(cur_accuracy)
    
    f, ax = plt.subplots()
    ax.plot(wl_range, wl_accuracies)
    plt.xlabel('Number of weak learners')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. number of weak learners')
    plt.show()
    return predictions, last_prediction
