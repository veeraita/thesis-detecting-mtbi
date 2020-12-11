import pandas as pd
import numpy as np
import random
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectorMixin

from confounds import ConfoundRegressor

pd.set_option('use_inf_as_na', True)


def get_dataset(fpath, key, cases, subject_data_fpath, column_filter='_amp|_width|_freq|_exponent', index_filter=None,
                dropna=True):
    meg_data_orig = pd.read_hdf(fpath, key=key)
    meg_data = meg_data_orig.filter(regex=column_filter, axis='columns')
    if index_filter is not None:
        meg_data = meg_data.filter(regex=index_filter, axis='index')

    meg_data = meg_data.replace([np.inf, -np.inf], np.nan)
    meg_data[meg_data > 50] = np.nan
    # amps to zero if peak does not exist
    meg_data = meg_data.apply(
        lambda col: col if '_freq' in col.name or '_knee' in col.name else col.fillna(0))
    # change some columns to binary
    meg_data = meg_data.apply(
        lambda col: col.where(col == 0, other=1) if 'theta_freq' in col.name or 'delta_freq' in col.name or 'gamma_freq'
                                                    in col.name else col)
    #meg_data['theta_amp_global'] = meg_data_orig.filter(regex='theta_amp', axis='columns').mean(axis=1)
    #meg_data['delta_amp_global'] = meg_data_orig.filter(regex='delta_amp', axis='columns').mean(axis=1)
    # for the remaining columns, drop if contains more than 10% nans
    orig_ncol = meg_data.shape[1]
    if dropna:
        meg_data = meg_data.dropna(thresh=len(meg_data) - int(len(meg_data) / 10), axis=1)
        print(f'Dropped {orig_ncol - meg_data.shape[1]} columns with too many nan values')

    meg_data = meg_data.loc[:, meg_data.std() > 0]

    subject_data = pd.read_csv(subject_data_fpath).set_index('subject')
    meg_data['cohort'] = meg_data.apply(lambda row: subject_data.loc[row.name.split('_')[0], 'cohort'], axis=1)
    # meg_data['site'] = meg_data.apply(lambda row: 0 if row.name.startswith('sub-') else 1, axis=1)

    print('The shape of the data is', meg_data.shape)
    print(meg_data.head())

    sample_names = meg_data.index
    y = np.array([1 if s[:3] in cases else 0 for s in sample_names])
    print('Dataset contains %d rows (%d positive)' % (len(y), sum(y)))

    return meg_data, y, sample_names


def features_from_df(X_df):
    confounds = X_df[['cohort']].values
    X_df = X_df.drop(['cohort'], axis=1)
    X = X_df.values
    # replace nans with mean
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    feature_names = np.asarray(list(X_df.columns))
    return X, confounds, feature_names


def preprocess(X_train, y_train, X_test, y_test, confounds=None):
    if confounds is not None:
        print("Fitting confound regression on training data")
        cr = ConfoundRegressor(confounds, np.vstack((X_train, X_test)))
        cr.fit(X_train)
        X_train = cr.transform(X_train)
        X_test = cr.transform(X_test)

    x_scaler = StandardScaler().fit(X_train)
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
    selected_feature_inds = [v[0] for v in cluster_id_to_feature_ids.values()]
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


def feature_selection_by_rfe(X, y, clf, k=10, step=1):
    print(f'Selecting {k} best features based on RFE')
    selector = RFE(clf, n_features_to_select=k, step=step, verbose=0)
    selector.fit(X, y)
    selected_feature_inds = selector.get_support(indices=True)
    return np.asarray(selected_feature_inds)


def feature_selection(X_train_orig, X_test_orig, y_train, feature_names, clf=None, k=10, step=1, thresh=1,
                      random_state=1, linear=False):
    # Select features by hierarchical clustering to reduce multicollinearity
    clustered_features = feature_selection_by_clustering(X_train_orig, thresh=thresh, random_state=random_state)
    X_train = X_train_orig[:, clustered_features]
    X_test = X_test_orig[:, clustered_features]
    feature_names = feature_names[clustered_features]

    if linear:
        selected_features = feature_selection_by_rfe(X_train, y_train, clf, k=k, step=step)
    else:
        # Select best features based on mutual information
        selected_features = feature_selection_by_mi(X_train, y_train, k=k, random_state=random_state)
    X_train = X_train[:, selected_features]
    X_test = X_test[:, selected_features]
    feature_names = feature_names[selected_features]

    feature_inds = clustered_features[selected_features]

    return X_train, X_test, feature_names, feature_inds


class FeatureSelector(BaseEstimator, SelectorMixin):
    def __init__(self, clf=None, k=10, step=1, thresh=1, random_state=1, linear=False):
        self.clf = clf
        self.k = k
        self.step = step
        self.thresh = thresh
        self.random_state = random_state
        self.linear = linear
        self.feature_inds = None
        self.support_mask = None

    def fit(self, X, y=None, **fit_params):
        clustered_features = feature_selection_by_clustering(X, thresh=self.thresh, random_state=self.random_state)
        X_new = X[:, clustered_features]
        if self.linear:
            selected_features = feature_selection_by_rfe(X_new, y, self.clf, k=self.k, step=self.step)
        else:
            # Select best features based on mutual information
            selected_features = feature_selection_by_mi(X_new, y, k=self.k, random_state=self.random_state)
        self.feature_inds = clustered_features[selected_features]
        self.support_mask = np.asarray([False] * X.shape[1])
        self.support_mask[self.feature_inds] = True
        return self

    def _get_support_mask(self):
        return self.support_mask
