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
from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from functools import partial

import nn_tools
from nn_tools import get_normalization_params, normalize
import meg_dataset
from net import Net
from inspection import plot_learning_curves

DATA_DIR = '/scratch/nbe/tbi-meg/veera/fooof_data/'
DATA_FILENAME = 'meg_{}_features_{}_window_v3.h5'
SUBJECTS_DATA_FPATH = 'subject_demographics.csv'
RANDOM_SEED = 42
CV = 7
TEST_SPLIT = 0.2
NORMALIZE = True
SKIP_TRAINING = False

L1 = 8
L2 = 16
L3 = 80
LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.1
WEIGHT = [2, 1]
BATCH_SIZE = 64
N_EPOCHS = 150
NOISE_FACTOR = 0.5

CASES = ['%03d' % n for n in range(28)]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.set_deterministic(True)


def performance_per_subject(y_test, y_pred, sample_names, subjects, average='binary'):
    y_test_subjects = []
    y_pred_subjects = []
    for s in subjects:
        sum = 0
        for i in range(len(sample_names)):
            if sample_names[i].split('_')[0] == s:
                sum += y_pred[i]
        if sum > (len(y_pred) / len(y_test_subjects)) / 2:
            y_pred_subjects.append(1)
        else:
            y_pred_subjects.append(0)

    for i in range(len(subjects)):
        print(y_test_subjects[i], y_pred_subjects[i], subjects[i])
    accuracy = accuracy_score(y_test_subjects, y_pred_subjects)
    precision = precision_score(y_test_subjects, y_pred_subjects, zero_division=1, average=average)
    recall = recall_score(y_test_subjects, y_pred_subjects, zero_division=1, average=average)
    f1 = f1_score(y_test_subjects, y_pred_subjects, zero_division=1, average=average)

    print(classification_report(y_test_subjects, y_pred_subjects, labels=[0, 1], target_names=['negative', 'positive'],
                                zero_division=1))
    return accuracy, precision, recall, f1


def compute_accuracy(net, testloader, m, s, device='cpu', average='binary', last_epoch=False):
    net.eval()
    correct = 0
    total = 0
    f1_scores = []
    precisions = []
    recalls = []
    with torch.no_grad():
        for inputs, labels, sample_names in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            inputs = normalize(inputs, m, s)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            if last_epoch:
                for i in range(len(predicted)):
                    print(labels[i].item(), predicted[i].item(), sample_names[i])

            # performance_per_subject(labels, predicted, sample_names, subjects, average='binary')

            f1_scores.append(f1_score(labels, predicted, average=average, zero_division=0))
            precisions.append(precision_score(labels, predicted, average=average, zero_division=0))
            recalls.append(recall_score(labels, predicted, average=average, zero_division=0))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    f1 = np.mean(f1_scores)
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    return accuracy, f1, precision, recall


def fit(model, dataloader, optimizer, criterion, m, s, device='cpu', verbose=False):
    if verbose:
        print('Training')
    model.train()
    running_loss = 0.0

    for (inputs, labels, subjects) in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        inputs = normalize(inputs, m, s)
        i_noisy = inputs + NOISE_FACTOR * torch.randn_like(inputs)

        optimizer.zero_grad()
        outputs = model(i_noisy)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    if verbose:
        print(f"Train Loss: {epoch_loss:.3f}")

    return epoch_loss


def validate(model, dataloader, criterion, m, s, device='cpu', verbose=False, last_epoch=False):
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

    epoch_accuracy, epoch_f1, epoch_precision, epoch_recall = compute_accuracy(model, dataloader, m, s, device,
                                                                               last_epoch=last_epoch)
    epoch_loss = running_loss / len(dataloader)
    if verbose:
        print(f"Val Loss: {epoch_loss:.3f}")
        print(f"Accuracy: {epoch_accuracy:.3f}")
        print(f"F1 Score: {epoch_f1:.3f}")
        print(f"Precision: {epoch_precision:.3f}")
        print(f"Recall: {epoch_recall:.3f}\n")

    return epoch_loss, epoch_accuracy, epoch_f1, epoch_precision, epoch_recall


# Training loop
def train(model, trainloader, testloader, optimizer, criterion, mean, std, device='cpu', verbose=False, tuning=False):
    train_loss = []
    val_loss = []
    accuracy = []
    f1_score = []
    precision = []
    recall = []
    last_epoch = False
    start = time.time()
    for epoch in range(N_EPOCHS):
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1} of {N_EPOCHS}")

        if (epoch + 1) == N_EPOCHS:
            last_epoch = True
        train_epoch_loss = fit(model, trainloader, optimizer, criterion, mean, std, device, verbose=verbose)
        val_epoch_loss, epoch_accuracy, epoch_f1_score, epoch_precision, epoch_recall = validate(model, testloader,
                                                                                                 criterion, mean, std,
                                                                                                 device,
                                                                                                 verbose=verbose,
                                                                                                 last_epoch=last_epoch)

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        accuracy.append(epoch_accuracy)
        f1_score.append(epoch_f1_score)
        precision.append(epoch_precision)
        recall.append(epoch_recall)

    end = time.time()

    print(f"----- Final results of the fold -----")
    print(f"{(end - start) / 60:.3} minutes")
    print(f"Train loss: {train_loss[-1]:.3f}")
    print(f"Val loss: {val_loss[-1]:.3f}\n")
    print(f"Accuracy: {accuracy[-1]:.3f}")
    print(f"F1 Score: {f1_score[-1]:.3f}")
    print(f"Precision: {precision[-1]:.3f}")
    print(f"Recall: {recall[-1]:.3f}\n")
    return model, train_loss, val_loss, accuracy, f1_score, precision, recall


def run_cv(model, cv, dataset, optimizer, criterion, subjects, device='cpu', batch_size=BATCH_SIZE, verbose=False,
           tuning=False):
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    train_losses = []
    val_losses = []

    subs_y = np.asarray([1 if s[:3] in CASES else 0 for s in list(subjects)])
    subjects = np.asarray(subjects)
    for fold, (train_i, test_i) in enumerate(cv.split(subjects, y=subs_y)):
        print('FOLD', fold + 1)
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        train_idx = np.where(dataset.df.index.str.split(pat='_').str[0].isin(subjects[train_i]))[0]
        test_idx = np.where(dataset.df.index.str.split(pat='_').str[0].isin(subjects[test_i]))[0]
        X_train = Subset(dataset, train_idx)
        X_test = Subset(dataset, test_idx)

        tl = DataLoader(X_train, batch_size=len(train_idx))
        trainsample, _, _ = iter(tl).next()
        mean, std = get_normalization_params(trainsample)

        sampler = RandomSampler(X_train, replacement=True, num_samples=5000)
        trainloader = DataLoader(X_train, batch_size=batch_size, sampler=sampler)

        # trainloader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(X_test, batch_size=batch_size, shuffle=False)

        model, train_loss, val_loss, accuracy, f1, precision, recall = train(model, trainloader, testloader, optimizer,
                                                                             criterion,
                                                                             mean, std, device, verbose=verbose,
                                                                             tuning=tuning)
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if tuning:
            tune.report(loss=val_loss[-1], f1=f1[-1], acc=accuracy[-1])
    return np.asarray(accuracies), np.asarray(f1_scores), np.asarray(precisions), np.asarray(recalls), \
           np.asarray(train_losses), np.asarray(val_losses)


def train_with_tuning(config, dataset=None, checkpoint_dir=None, cv=None):
    dataset_size = len(dataset)
    tl = DataLoader(dataset, batch_size=dataset_size)
    sample, labels, sample_names = iter(tl).next()
    subjects = list(set([s.split('_')[0] for s in sample_names]))
    subjects = np.asarray(subjects)
    in_features = sample.shape[1]
    print(f'The number of input features is {in_features}')

    net = Net(in_features, 2, config["l1"], config["l2"], config["l3"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, config["weight"]]).float().to(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["wd"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    accuracies, f1_scores, precisions, recalls, train_losses, val_losses = run_cv(net, cv, dataset, optimizer,
                                                                                  criterion, subjects,
                                                                                  batch_size=config["batch_size"],
                                                                                  verbose=False, tuning=True)

    print("Finished")


def run_hyperparameter_tuning(dataset, cv):
    config = {
        "l1": tune.choice(2 ** np.arange(2, 8)),
        "l2": tune.choice(2 ** np.arange(2, 8)),
        "l3": tune.choice(2 ** np.arange(2, 8)),
        "lr": tune.choice([0.00001, 0.0001, 0.001]),
        "wd": tune.choice([0.0001, 0.001, 0.01, 0.1]),
        "batch_size": tune.choice([32, 64, 128])
    }
    num_samples = 50
    max_num_epochs = 1
    gpus_per_trial = 0

    scheduler = ASHAScheduler(
        metric="acc",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(metric_columns=["loss", "f1", "acc", "training_iteration"])

    result = tune.run(
        partial(train_with_tuning, dataset=dataset, cv=cv),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("acc", "max", "avg")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["acc"]))
    print("Best trial final validation F1: {}".format(
        best_trial.last_result["f1"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--parc', help='The cortical parcellation to use', choices=['aparc', 'aparc_sub'],
                        default='aparc_sub')
    parser.add_argument('-t', '--target', help='Whether to do the analysis subject-wise or parcel-wise',
                        choices=['subjects', 'parcels'], default='subjects')
    parser.add_argument('-c', '--cohorts', help='Fit classifier for each cohort separately', action='store_true',
                        default=False)
    parser.add_argument('--tune', help='Run hyperparameter tuning', action='store_true', default=False)

    args = parser.parse_args()

    device = nn_tools.get_device()
    print('Setting device as', device)

    data_filename = DATA_FILENAME.format(args.parc, args.target)
    data_key = data_filename.split('.')[0]
    data_fpath = os.path.join(DATA_DIR, data_filename)

    dataset = meg_dataset.FOOOFDataset(data_fpath, data_key, index_filter='^0')
    print(f'The dataset contains {len(dataset)} samples')

    cv = StratifiedKFold(n_splits=CV, shuffle=True, random_state=RANDOM_SEED)

    if args.tune:
        run_hyperparameter_tuning(dataset, cv)
    else:
        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        test_size = TEST_SPLIT
        train_size = dataset_size - test_size
        # trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

        tl = DataLoader(dataset, batch_size=dataset_size)
        sample, labels, sample_names = iter(tl).next()
        subjects = list(set([s.split('_')[0] for s in sample_names]))
        subjects = np.asarray(subjects)
        in_features = sample.shape[1]
        print(f'The number of input features is {in_features}')

        splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
        for train_i, test_i in splitter.split(subjects):
            train_idx = np.where(dataset.df.index.str.split(pat='_').str[0].isin(subjects[train_i]))[0]
            test_idx = np.where(dataset.df.index.str.split(pat='_').str[0].isin(subjects[test_i]))[0]
        X_train = Subset(dataset, train_idx)
        X_test = Subset(dataset, test_idx)

        net = Net(in_features, 2, l1=L1, l2=L2, l3=L3)
        net.to(device)
        print(net)

        criterion = nn.CrossEntropyLoss(weight=torch.tensor(WEIGHT).float())
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        accuracies, f1_scores, precisions, recalls, train_losses, val_losses = run_cv(net, cv, dataset, optimizer,
                                                                                      criterion, subjects)

        print('Accuracy: %.2f, STD %.2f' % (np.mean(accuracies[:, -1]), np.std(accuracies[:, -1])))
        print('Precision: %.2f, STD %.2f' % (np.nanmean(precisions[:, -1]), np.nanstd(precisions[:, -1])))
        print('Recall: %.2f, STD %.2f' % (np.nanmean(recalls[:, -1]), np.nanstd(recalls[:, -1])))
        print('F1: %.2f, STD %.2f' % (np.mean(f1_scores[:, -1]), np.std(f1_scores[:, -1])))

        plot_learning_curves(train_losses, val_losses, f1_scores, accuracies, 'fig/learning_curves.png')

        if np.mean(accuracies[:, -1]) > 0.7:
            with open('reports/nn_results.txt', 'a') as f:
                f.write(f'l1 = {L1}, l2 = {L2}, lr = {LEARNING_RATE}, wd = {WEIGHT_DECAY}, weight = {WEIGHT}, '
                        f'batch size = {BATCH_SIZE}, noise = {NOISE_FACTOR}\n')
                f.write('Accuracy: %.2f, STD %.2f\n' % (np.mean(accuracies[:, -1]), np.std(accuracies[:, -1])))
                f.write('F1: %.2f, STD %.2f\n' % (np.mean(f1_scores[:, -1]), np.std(f1_scores[:, -1])))


if __name__ == "__main__":
    main()
