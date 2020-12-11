import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from glob import glob

from preprocessing import get_dataset, features_from_df

CASES = ['%03d' % n for n in range(28)]


class FOOOFDataset(Dataset):
    def __init__(self, data_fpath, data_key, transform=None, index_filter=None):
        self.transform = transform
        df, y, sample_names = get_dataset(data_fpath, data_key, CASES, 'subject_demographics.csv',
                                          column_filter='alpha_amp|beta_amp|alpha_width|beta_width|alpha_freq|beta_freq|exponent',
                                          #column_filter='_amp|alpha_freq|beta_freq',
                                          index_filter=index_filter, dropna=False)

        #subject_data = pd.read_csv('subject_demographics.csv').set_index('subject')
        #df['cohort'] = df.apply(lambda row: subject_data.loc[row.name.split('_')[0], 'cohort'], axis=1)
        X, _, feature_names = features_from_df(df)
        self.dataset = torch.from_numpy(X).float()
        self.labels = y
        self.subjects = sample_names
        self.df = df

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x = self.dataset[i]
        if self.transform is not None:
            x = self.transform(x)
        label = self.labels[i]
        subject = self.subjects[i]
        return x, label, subject