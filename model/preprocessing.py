import pandas as pd
import numpy as np
import random
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectorMixin

pd.set_option('use_inf_as_na', True)


def get_tmap_dataset(fpath, cases, freq_bands):
    dataset = pd.read_csv(fpath, header=None, index_col=0)
    data = dataset.values
    # bin the data into frequency bands
    binned_data = []
    for (lo, hi) in freq_bands.values():
        binned_data.append(np.mean(data[:, lo:hi], axis=1))
    data = np.asarray(binned_data).T

    # rearrange the data matrix into 'samples x features'
    dataset = pd.DataFrame(data, index=dataset.index, columns=freq_bands.keys())
    dataset['subject'] = dataset.index.str.split('-', 1).str[0]
    dataset['label'] = dataset.index.str.split('-', 1).str[1]
    dataset = dataset.set_index(['subject', 'label'])
    dataset = dataset.unstack()
    dataset.columns = ['-'.join(col).strip() for col in dataset.columns.values]
    dataset = dataset.loc[:, dataset.std() > 0.01]
    print('The shape of the data is', dataset.shape)
    print(dataset.head())

    sample_names = dataset.index
    y = np.array([1 if s[:3] in cases else 0 for s in sample_names])
    print('Dataset contains %d rows (%d positive)' % (len(y), sum(y)))
    return dataset, y


def features_from_df(X_df):
    X = X_df.values
    # replace nans with mean
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    feature_names = np.asarray(list(X_df.columns))
    return X, feature_names


def preprocess(X_train, y_train, X_test, y_test):
    x_scaler = RobustScaler().fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)

    print("Train set is %d documents (%d positive)" % (len(y_train), sum(y_train)))
    print("Test set is %d documents (%d positive)" % (len(y_test), sum(y_test)))
    return X_train, y_train, X_test, y_test


def feature_selection_by_clustering(X, thresh=1, random_state=1):
    print(f'Performing hierarchical clustering of features (threshold = {thresh})')
    corr = spearmanr(X).correlation
    corr_linkage = hierarchy.ward(corr)

    cluster_ids = hierarchy.fcluster(corr_linkage, thresh, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    # select a single feature from each cluster
    selected_feature_inds = [v[-1] for v in cluster_id_to_feature_ids.values()]
    print(f'Selected {len(selected_feature_inds)} features')

    return np.asarray(selected_feature_inds)


def feature_selection_by_mi(X, y, k=10, random_state=1):
    def mi_score(X, y):
        return mutual_info_classif(X, y, random_state=random_state)

    print(f'Selecting {k} best features based on mutual information')
    if k != 'all' and k > X.shape[1]:
        print(f'k > {X.shape[1]}, selecting all features')
        k = 'all'
    selector = SelectKBest(mi_score, k=k)
    selector.fit(X, y)
    selected_feature_inds = selector.get_support(indices=True)
    return np.asarray(selected_feature_inds)


class FeatureSelector(BaseEstimator, SelectorMixin):
    def __init__(self, clf=None, k=10, thresh=1, random_state=1):
        self.clf = clf
        self.k = k
        self.thresh = thresh
        self.random_state = random_state
        self.feature_inds_ = None
        self.support_mask_ = None

    def fit(self, X, y=None, **fit_params):
        clustered_features = feature_selection_by_clustering(X, thresh=self.thresh, random_state=self.random_state)
        X_new = X[:, clustered_features]
        if self.k != 'all':
            # Select best features based on mutual information
            selected_features = feature_selection_by_mi(X_new, y, k=self.k, random_state=self.random_state)
            self.feature_inds_ = clustered_features[selected_features]
        else:
            self.feature_inds_ = clustered_features
        self.support_mask_ = np.asarray([False] * X.shape[1])
        self.support_mask_[self.feature_inds_] = True
        return self

    def _get_support_mask(self):
        return self.support_mask_

