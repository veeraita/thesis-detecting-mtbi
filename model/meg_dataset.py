import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from glob import glob


class FOOOFDataset(Dataset):
    def __init__(self, data_fpath, data_key, transform=None):
        self.transform = transform
        self.cases = ['%03d' % n for n in range(28)]

        df = pd.read_hdf(data_fpath, key=data_key)
        df = df.apply(lambda col: col.fillna(
            0) if '_amp' in col.name or '_width' in col.name or 'theta_freq' in col.name else col)
        # drop columns with more than 10% nans
        df = df.dropna(thresh=len(df) - int(len(df) / 10), axis=1)
        # change theta_freq to binary
        df = df.apply(lambda col: col.where(col == 0, other=1) if 'theta_freq' in col.name else col)

        subject_data = pd.read_csv('subject_demographics.csv').set_index('subject')
        df['cohort'] = df.apply(lambda row: subject_data.loc[row.name.split('_')[0], 'cohort'], axis=1)

        Xnum = df.filter(regex='alpha|exponent|_amp|_width|cohort').values
        Xcat = df.filter(regex='theta_freq').values
        col_mean = np.nanmean(Xnum, axis=0)
        inds = np.where(np.isnan(Xnum))
        Xnum[inds] = np.take(col_mean, inds[1])
        dataset = np.hstack((Xnum, Xcat))

        self.dataset = torch.from_numpy(dataset).float()

        self.subjects = df.index.values
        self.labels = np.array([1 if s[:3] in self.cases else 0 for s in self.subjects])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x = self.dataset[i]
        if self.transform is not None:
            x = self.transform(x)
        label = self.labels[i]
        subject = self.subjects[i]
        return x, label, subject