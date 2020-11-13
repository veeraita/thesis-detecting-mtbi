import os
import time
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

import tools
import meg_dataset
from net import Net
from inspection import plot_learning_curves

DATA_DIR = '/scratch/nbe/tbi-meg/veera/fooof_data/'
DATA_FILENAME = 'meg_{}_features_{}_window_v2.h5'
SUBJECTS_DATA_FPATH = 'subject_demographics.csv'
RANDOM_SEED = 42
CV = 10
BATCH_SIZE = 64
TEST_SPLIT = 0.2
NORMALIZE = True
N_EPOCHS = 20
SKIP_TRAINING = False
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001


def get_normalization_params(tensor):
    m = tensor.mean(0, keepdim=True)
    s = tensor.std(0, unbiased=False, keepdim=True)
    return m, s


def normalize(tensor, m, s):
    normalized_tensor = (tensor - m) / s
    return normalized_tensor


def compute_accuracy(net, testloader, m, s, device='cpu'):
    net.eval()
    correct = 0
    total = 0
    f1_scores = []
    with torch.no_grad():
        for inputs, labels, subjects in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            inputs = normalize(inputs, m, s)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            f1_scores.append(f1_score(labels, predicted, average='binary', zero_division=0))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    f1 = np.mean(f1_scores)
    return accuracy, f1


def fit(model, dataloader, optimizer, criterion, m, s, device='cpu', verbose=False):
    if verbose:
        print('Training')
    model.train()
    running_loss = 0.0

    for (inputs, labels, subjects) in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = normalize(inputs, m, s)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    if verbose:
        print(f"Train Loss: {epoch_loss:.3f}")

    return epoch_loss


def validate(model, dataloader, criterion, m, s, device='cpu', verbose=False):
    if verbose:
        print('Validating')
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (inputs, labels, subjects) in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            inputs = normalize(inputs, m, s)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

    epoch_accuracy, epoch_f1 = compute_accuracy(model, dataloader, m, s, device)
    epoch_loss = running_loss / len(dataloader)
    if verbose:
        print(f"Val Loss: {epoch_loss:.3f}")
        print(f"Accuracy: {epoch_accuracy:.3f}")
        print(f"F1 Score: {epoch_f1:.3f}\n")

    return epoch_loss, epoch_accuracy, epoch_f1


# Training loop
def train(model, trainloader, testloader, optimizer, criterion, mean, std, device='cpu', verbose=False):
    train_loss = []
    val_loss = []
    accuracy = []
    f1_score = []

    start = time.time()
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1} of {N_EPOCHS}")

        train_epoch_loss = fit(model, trainloader, optimizer, criterion, mean, std, device, verbose=verbose)
        val_epoch_loss, epoch_accuracy, epoch_f1_score = validate(model, testloader, criterion, mean, std, device, verbose=verbose)

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        accuracy.append(epoch_accuracy)
        f1_score.append(epoch_f1_score)

    end = time.time()

    print(f"{(end - start) / 60:.3} minutes")
    print(f"Final F1 Score: {f1_score[-1]:.3f}\n")
    return model, train_loss, val_loss, accuracy, f1_score


def run_cv(model, cv, dataset, optimizer, criterion, device='cpu', verbose=False):
    f1_scores = []
    train_losses = []
    val_losses = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(dataset)):
        print('FOLD', fold + 1)
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        X_train = Subset(dataset, train_idx)
        X_test = Subset(dataset, test_idx)

        tl = DataLoader(X_train, batch_size=len(train_idx))
        trainsample, _, _ = iter(tl).next()
        mean, std = get_normalization_params(trainsample)

        trainloader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
        testloader = DataLoader(X_test, batch_size=BATCH_SIZE, shuffle=False)

        model, train_loss, val_loss, accuracy, f1 = train(model, trainloader, testloader, optimizer, criterion,
                                                          mean, std, device, verbose=verbose)
        f1_scores.append(f1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    return np.asarray(f1_scores), np.asarray(train_losses), np.asarray(val_losses)


def train_with_tuning(config, dataset=None, checkpoint_dir=None):
    dataset_size = len(dataset)
    test_size = int(TEST_SPLIT * dataset_size)
    train_size = dataset_size - test_size
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

    tl = DataLoader(trainset, batch_size=dataset_size)
    trainsample, _, _ = iter(tl).next()
    mean, std = get_normalization_params(trainsample)
    in_features = trainsample.shape[1]

    net = Net(in_features, 2, config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, config["weight"]]).float().to(device))
    optimizer = optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["wd"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8)

    for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
        train_epoch_loss = fit(net, trainloader, optimizer, criterion, mean, std, device)
        val_epoch_loss, epoch_accuracy, epoch_f1_score = validate(net, testloader, criterion, mean, std, device)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_epoch_loss, f1=epoch_f1_score)
    print("Finished Training")


def run_hyperparameter_tuning(dataset):
    config = {
        "l1": tune.choice(2 ** np.arange(2, 9)),
        "l2": tune.choice(2 ** np.arange(2, 9)),
        "lr": tune.loguniform(1e-5, 1e-1),
        "wd": tune.loguniform(1e-5, 1e-1),
        "weight": tune.choice([1, 5, 10, 15, 20])
    }
    num_samples = 30
    max_num_epochs = 10
    gpus_per_trial = 0

    scheduler = ASHAScheduler(
        metric="f1",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(metric_columns=["loss", "f1", "training_iteration"])

    result = tune.run(
        partial(train_with_tuning, dataset=dataset, checkpoint_dir='reports'),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("f1", "max", "last-5-avg")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation F1: {}".format(
        best_trial.last_result["f1"]))


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--parc', help='The cortical parcellation to use', choices=['aparc', 'aparc_sub'],
                        default='aparc_sub')
    parser.add_argument('-t', '--target', help='Whether to do the analysis subject-wise or parcel-wise',
                        choices=['subjects', 'parcels'], default='subjects')
    parser.add_argument('-c', '--cohorts', help='Fit classifier for each cohort separately', action='store_true',
                        default=False)
    parser.add_argument('--tune', help='Run hyperparameter tuning', action='store_true', default=False)

    args = parser.parse_args()

    device = tools.get_device()
    print('Setting device as', device)

    data_filename = DATA_FILENAME.format(args.parc, args.target)
    data_key = data_filename.split('.')[0]
    data_fpath = os.path.join(DATA_DIR, data_filename)

    dataset = meg_dataset.FOOOFDataset(data_fpath, data_key)
    print(f'The dataset contains {len(dataset)} samples')

    if args.tune:
        run_hyperparameter_tuning(dataset)
    else:
        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        test_size = int(TEST_SPLIT * dataset_size)
        train_size = dataset_size - test_size
        trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

        tl = DataLoader(trainset, batch_size=train_size)
        trainsample, _, _ = iter(tl).next()
        in_features = trainsample.shape[1]

        net = Net(in_features, 2, l1=256, l2=64)
        net.to(device)
        print(net)

        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 10]).float())
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
        f1_scores, train_losses, val_losses = run_cv(net, cv, trainset, optimizer, criterion)
        print('F1: %.2f, STD %.2f' % (np.mean(f1_scores[:, -1]), np.std(f1_scores[:, -1])))

        plot_learning_curves(train_losses, val_losses, f1_scores, 'fig/learning_curves.png')

if __name__ == "__main__":
    main()