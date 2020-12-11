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

from nn_tools import *
import meg_dataset
from ae import Encoder, Decoder
from inspection import plot_learning_curves, plot_loss_dist

DATA_DIR = '/scratch/nbe/tbi-meg/veera/fooof_data/'
DATA_FILENAME = 'meg_{}_features_{}_window_v2.h5'
SUBJECTS_DATA_FPATH = 'subject_demographics.csv'
RANDOM_SEED = 42
CV = 10
BATCH_SIZE = 64
TEST_SPLIT = 0.2
NORMALIZE = True
N_EPOCHS = 200
SKIP_TRAINING = False
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.001

CASES = ['%03d' % n for n in range(28)]


def get_loss_dist(dataset, encoder, decoder, criterion):
    encoder.eval()
    decoder.eval()
    losses = []
    labels = []
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for i, (inputs, labs, names) in enumerate(dataloader):
            z = encoder(inputs)
            outputs = decoder(z)
            for j in range(len(inputs)):
                inp = inputs[j].unsqueeze(0)
                target = outputs[j].unsqueeze(0)
                loss = criterion(inp, target)
                losses.append(torch.mean(torch.max(loss, 1)[0]).item())
                labels.append(labs[j].item())
    return losses, labels


def print_results(loss_dist, labels, threshold):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    total_anom = 0

    for i in range(len(loss_dist)):
        label = labels[i]
        total_anom += label

        if loss_dist[i] >= threshold:
            if label == 1:
                tp += 1
            else:
                fp += 1
        else:
            if label == 1:
                fn += 1
            else:
                tn += 1
    print('[TP] {}\t\t[FP] {}\t\t[MISSED] {}'.format(tp, fp, total_anom - tp))
    print('[TN] {}\t\t[FN] {}'.format(tn, fn))


def fit(encoder, decoder, dataloader, optimizer, criterion, m, s, device='cpu', verbose=False):
    if verbose:
        print('Training')
    encoder.train()
    decoder.train()
    running_loss = 0.0

    for (inputs, labels, subjects) in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = normalize(inputs, m, s)

        optimizer.zero_grad()
        z = encoder(inputs)
        outputs = decoder(z)

        loss = criterion(inputs, outputs)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    if verbose:
        print(f"Train Loss: {epoch_loss:.3f}")

    return epoch_loss


def validate(encoder, decoder, dataloader, criterion, m, s, device='cpu', verbose=False):
    if verbose:
        print('Validating')
    encoder.eval()
    decoder.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (inputs, labels, subjects) in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            inputs = normalize(inputs, m, s)

            z = encoder(inputs)
            outputs = decoder(z)

            loss = criterion(inputs, outputs)
            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    if verbose:
        print(f"Val Loss: {epoch_loss:.3f}")
        print(f"Accuracy: {epoch_accuracy:.3f}")
        print(f"F1 Score: {epoch_f1:.3f}")
        print(f"Precision: {epoch_precision:.3f}")
        print(f"Recall: {epoch_recall:.3f}\n")

    return epoch_loss


# Training loop
def train(encoder, decoder, trainloader, testloader, optimizer, criterion, mean, std, device='cpu', verbose=False,
          tuning=False):
    train_loss = []
    val_loss = []

    start = time.time()
    for epoch in range(N_EPOCHS):
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1} of {N_EPOCHS}")

        train_epoch_loss = fit(encoder, decoder, trainloader, optimizer, criterion, mean, std, device, verbose=verbose)
        val_epoch_loss = validate(encoder, decoder, testloader, criterion, mean, std, device, verbose=verbose)

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)

    end = time.time()

    print(f"----- Final results of the fold -----")
    print(f"{(end - start) / 60:.3} minutes")
    print(f"Train loss: {train_loss[-1]:.3f}")
    print(f"Validation loss: {val_loss[-1]:.3f}")
    return encoder, decoder, train_loss, val_loss


def run_cv(encoder, decoder, cv, dataset, optimizer, criterion, subjects, sample_names, device='cpu', batch_size=BATCH_SIZE,
           verbose=False, tuning=False):
    train_losses = []
    val_losses = []

    subs_y = np.asarray([1 if s[:3] in CASES else 0 for s in list(subjects)])
    # for fold, (train_idx, test_idx) in enumerate(cv.split(dataset)):
    subjects = np.asarray(subjects)
    for fold, (train_i, test_i) in enumerate(cv.split(subjects, y=subs_y)):
        print('FOLD', fold + 1)
        for layer in encoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        _, train_idx, _ = np.intersect1d(sample_names, subjects[train_i], return_indices=True)
        _, test_idx, _ = np.intersect1d(sample_names, subjects[test_i], return_indices=True)
        X_train = Subset(dataset, train_idx)
        X_test = Subset(dataset, test_idx)

        tl = DataLoader(X_train, batch_size=len(train_idx))
        trainsample, _, _ = iter(tl).next()
        mean, std = get_normalization_params(trainsample)

        trainloader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(X_test, batch_size=batch_size, shuffle=False)

        encoder, decoder, train_loss, val_loss = train(encoder, decoder, trainloader, testloader, optimizer, criterion, mean, std,
                                            device, verbose=verbose, tuning=tuning)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        #if tuning:
        #    tune.report(loss=val_loss[-1], f1=f1[-1], acc=accuracy[-1])
    return np.asarray(train_losses), np.asarray(val_losses)


def train_with_tuning(config, dataset=None, checkpoint_dir=None, cv=None):
    dataset_size = len(dataset)
    tl = DataLoader(dataset, batch_size=dataset_size)
    sample, labels, sample_names = iter(tl).next()
    subjects = list(set([s.split('_')[0] for s in sample_names]))
    subjects = np.asarray(subjects)
    in_features = sample.shape[1]
    print(f'The number of input features is {in_features}')

    net = Net(in_features, 2, config["l1"], config["l2"])

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

    train_losses, val_losses = run_cv(net, cv, dataset, optimizer, criterion, subjects,
                                                             batch_size=config["batch_size"], verbose=False,
                                                             tuning=True)

    print("Finished")


def run_hyperparameter_tuning(dataset, cv):
    config = {
        "l1": tune.choice(2 ** np.arange(2, 9)),
        "l2": tune.choice(2 ** np.arange(2, 9)),
        "lr": tune.choice([0.00001, 0.0001, 0.001, 0.01]),
        "wd": tune.choice([0.0001, 0.001, 0.01, 0.1]),
        "batch_size": tune.choice([16, 32, 64, 128, 256])
        # "weight": tune.choice([1, 5, 10, 15, 20])
    }
    num_samples = 50
    max_num_epochs = 10
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

    device = get_device()
    print('Setting device as', device)

    data_filename = DATA_FILENAME.format(args.parc, args.target)
    data_key = data_filename.split('.')[0]
    data_fpath = os.path.join(DATA_DIR, data_filename)

    cases_set = meg_dataset.FOOOFDataset(data_fpath, data_key, index_filter='^00|^01|^02[0-7]')
    control_set = meg_dataset.FOOOFDataset(data_fpath, data_key, index_filter='^sub|^03|^04|^028|^029')
    names = control_set.df.index.str.split(pat='_').str[0].values

    #dataset = meg_dataset.FOOOFDataset(data_fpath, data_key)
    print(f'The training set contains {len(control_set)} samples')

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    if args.tune:
        run_hyperparameter_tuning(dataset, cv)
    else:
        # Creating data indices for training and validation splits:
        dataset_size = len(control_set)
        test_size = TEST_SPLIT
        train_size = dataset_size - test_size
        # trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

        tl = DataLoader(control_set, batch_size=dataset_size)
        sample, labels, sample_names = iter(tl).next()
        subjects = list(set([s.split('_')[0] for s in sample_names]))
        subjects = np.asarray(subjects)
        in_features = sample.shape[1]
        print(f'The number of input features is {in_features}')

        splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
        for train_i, test_i in splitter.split(subjects):
            train_idx = np.where(control_set.df.index.str.split(pat='_').str[0].isin(subjects[train_i]))[0]
            test_idx = np.where(control_set.df.index.str.split(pat='_').str[0].isin(subjects[test_i]))[0]
        trainset = Subset(control_set, train_idx)
        train_names = names[train_idx]
        testset = Subset(control_set, test_idx)
        test_names = names[test_idx]

        n_features = 50
        encoder = Encoder(in_features, n_features)
        encoder.to(device)

        decoder = Decoder(n_features, in_features)
        decoder.to(device)

        print(encoder)
        print(decoder)

        parameters = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()
        train_losses, val_losses = run_cv(encoder, decoder, cv, trainset, optimizer, criterion, subjects, train_names,
                                          verbose=False)
        print('Train loss: %.2f, STD %.2f' % (np.mean(train_losses[:, -1]), np.std(train_losses[:, -1])))
        print('Val loss: %.2f, STD %.2f' % (np.mean(val_losses[:, -1]), np.std(val_losses[:, -1])))

        plot_learning_curves(train_losses, val_losses, fig_fpath='fig/learning_curves_ae.png')

        criterion = nn.MSELoss(reduction='none')
        loss_dist_test, labels_test = get_loss_dist(testset, encoder, decoder, criterion)
        loss_dist_cases, labels_cases = get_loss_dist(cases_set, encoder, decoder, criterion)
        loss_dist = loss_dist_test + loss_dist_cases
        labels = labels_test + labels_cases
        threshold = 1100

        plot_loss_dist(loss_dist, 'fig/loss_dist.png', threshold=threshold)

        print_results(loss_dist, labels, threshold)


if __name__ == "__main__":
    main()
