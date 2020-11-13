#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

import os
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from inspection import feature_selection_by_clustering, feature_selection_by_mi, feature_importance, \
    save_feature_importance, plot_correlation, plot_linear_coefs, plot_class_separation

DATA_DIR = '/scratch/nbe/tbi-meg/veera/fooof_data/'
DATA_FILENAME = 'meg_{}_features_{}_window_v2.h5'
SUBJECTS_DATA_FPATH = 'subject_demographics.csv'
RANDOM_SEED = 42
CV = 10
TEST_SPLIT = 0.2
CLUSTER_THRESH = 10
N_FEATURES = 50
N_PERM_REPEATS = 5

CASES = ['%03d' % n for n in range(28)]


def get_dataset(fpath, key, column_filter='_amp|_width|_freq|_exponent'):
    meg_data_orig = pd.read_hdf(fpath, key=key).filter(regex=column_filter)
    print('The shape of the data is', meg_data_orig.shape)

    # widths and amps to zero if peak does not exist
    meg_data = meg_data_orig.apply(lambda col: col.fillna(
        0) if '_amp' in col.name or '_width' in col.name or 'theta' in col.name or 'delta' in col.name else col)
    # change theta_freq to binary
    meg_data = meg_data.apply(
        lambda col: col.where(col == 0, other=1) if 'theta_freq' in col.name or 'delta_freq' in col.name else col)
    # for the remaining columns, drop if contains more than 10% nans
    meg_data = meg_data.dropna(thresh=len(meg_data) - int(len(meg_data) / 10), axis=1)
    print(f'Dropped {meg_data_orig.shape[1] - meg_data.shape[1]} columns with too many nan values')

    sample_names = meg_data.index
    y = np.array([1 if s[:3] in CASES else 0 for s in sample_names])
    print('Dataset contains %d rows (%d positive)' % (len(y), sum(y)))

    return meg_data, y, sample_names


def features_from_df(X_df, num_filter='alpha|exponent|_amp|_width', cat_filter='theta_freq'):
    # replace nans with mean
    Xnum = X_df.filter(regex=num_filter).values
    Xcat = X_df.filter(regex=cat_filter).values
    col_mean = np.nanmean(Xnum, axis=0)
    inds = np.where(np.isnan(Xnum))
    Xnum[inds] = np.take(col_mean, inds[1])
    feature_names = np.asarray(
        list(X_df.filter(regex=num_filter).columns) + list(X_df.filter(regex=cat_filter).columns))
    X = np.hstack((Xnum, Xcat))
    return X, feature_names


def preprocess(X_train, y_train, X_test, y_test):
    x_scaler = StandardScaler().fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)

    print("Train set is %d documents (%d positive)" % (len(y_train), sum(y_train)))
    print("Test set is %d documents (%d positive)" % (len(y_test), sum(y_test)))
    return X_train, y_train, X_test, y_test


def feature_selection(X_train_orig, X_test_orig, y_train, feature_names, k=10):
    # Select features by hierarchical clustering to reduce multicollinearity
    clustered_features = feature_selection_by_clustering(X_train_orig, thresh=CLUSTER_THRESH)
    X_train = X_train_orig[:, clustered_features]
    X_test = X_test_orig[:, clustered_features]
    feature_names = feature_names[clustered_features]

    # Select best features based on mutual information
    selected_features = feature_selection_by_mi(X_train, y_train, k)
    X_train = X_train[:, selected_features]
    X_test = X_test[:, selected_features]
    feature_names = feature_names[selected_features]

    feature_inds = clustered_features[selected_features]

    return X_train, X_test, feature_names, feature_inds


def plot_results(clf, X_train, X_test, y_train, y_test, feature_names, feature_importances, linear_coefs,
                 parc='aparc_sub'):
    plot_correlation(X_train, feature_names, 'fig/corr.png')

    plot_class_separation(clf, X_train, X_test, y_train, y_test, 'fig/class_separation.png')

    importance_fig_fpath = f'fig/feature_importance_{parc}_thresh_{CLUSTER_THRESH}.png'
    importance_txt_fpath = f'reports/feature_importance_{parc}_thresh_{CLUSTER_THRESH}.txt'
    save_feature_importance(feature_importances, feature_names, importance_fig_fpath, importance_txt_fpath)

    plot_linear_coefs(linear_coefs, feature_names, 'fig/coef.png')


@ignore_warnings(category=ConvergenceWarning)
def run_cv(clf, X_df, y, average='binary', nested=False):
    cv_outer = StratifiedKFold(n_splits=CV, shuffle=True, random_state=RANDOM_SEED)
    print(f'Starting {cv_outer.n_splits}-fold cross validation')
    print(clf)
    search_space = [
        {'kernel': ['rbf'],
         'gamma': [1e-3, 1e-4, 'scale'],
         'C': [10, 100, 1000],
         'class_weight': [None, 'balanced']}
    ]
    precision = []
    recall = []
    f1 = []
    roc_auc = []
    importances = np.zeros((cv_outer.n_splits, X_df.shape[1], N_PERM_REPEATS))
    coefs = np.zeros((cv_outer.n_splits, X_df.shape[1]))
    for ii, (train, test) in enumerate(cv_outer.split(X_df, y=y)):
        print('Fold', ii + 1)
        X, feature_names = features_from_df(X_df)

        X_train, y_train, X_test, y_test = preprocess(X[train], y[train], X[test], y[test])

        X_train, X_test, feature_names, feature_inds = feature_selection(X_train, X_test, y_train, feature_names,
                                                                         k=N_FEATURES)

        if nested:
            cv_inner = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
            grid = GridSearchCV(clf, search_space, scoring='f1', cv=cv_inner, n_jobs=4, verbose=1)
            grid.fit(X_train, y_train)

            print('Best params:', grid.best_params_)
            print('Estimated F1 score: %.2f' % grid.best_score_)
            best_clf = grid.best_estimator_
        else:
            clf.fit(X_train, y_train)
            best_clf = clf

        importances[ii, feature_inds] = feature_importance(best_clf, X_test, y_test, n_repeats=N_PERM_REPEATS,
                                                           random_state=RANDOM_SEED)
        # Find linear coefficients
        linear_params = best_clf.get_params()
        linear_params['kernel'] = 'linear'
        classifier_linear = SVC(**linear_params)
        classifier_linear.fit(X_train, y_train)
        coef = classifier_linear.coef_[0]
        coefs[ii, feature_inds] = coef
        y_pred = best_clf.predict(X_test)
        try:
            y_scores = best_clf.decision_function(X_test)
        except:
            y_scores = np.max(best_clf.predict_proba(X_test), axis=1)

        cv_precision = precision_score(y_test, y_pred, zero_division=1, average=average)
        cv_recall = recall_score(y_test, y_pred, zero_division=1, average=average)
        cv_f1 = f1_score(y_test, y_pred, zero_division=1, average=average)
        try:
            cv_roc_auc = roc_auc_score(y_test, y_scores)
        except ValueError:
            cv_roc_auc = np.nan

        print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['negative', 'positive'],
                                    zero_division=1))

        precision.append(cv_precision)
        recall.append(cv_recall)
        f1.append(cv_f1)
        roc_auc.append(cv_roc_auc)

    return np.asarray(precision), np.asarray(recall), np.asarray(f1), np.asarray(roc_auc), \
           np.mean(importances, axis=0), np.mean(coefs, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--parc', help='The cortical parcellation to use', choices=['aparc', 'aparc_sub'],
                        default='aparc_sub')
    parser.add_argument('-t', '--target', help='Whether to do the analysis subject-wise or parcel-wise',
                        choices=['subjects', 'parcels'], default='subjects')
    parser.add_argument('-f', '--fit-params', help='Parameters to pass to the classifier (JSON string)',
                        type=json.loads, default='{}')
    parser.add_argument('-c', '--cohorts', help='Fit classifier for each cohort separately', action='store_true',
                        default=False)
    parser.add_argument('-n', '--nested-cv', help='Use nested cross-validation for model selection and validation',
                        action='store_true', default=False)

    args = parser.parse_args()

    data_filename = DATA_FILENAME.format(args.parc, args.target)
    data_key = data_filename.split('.')[0]
    data_fpath = os.path.join(DATA_DIR, data_filename)

    X_df, y, names = get_dataset(data_fpath, data_key, column_filter='_amp|_width|alpha_freq|_exponent')

    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED,
                                                              stratify=y)
    X_train, feature_names = features_from_df(X_train_df)
    X_test, _ = features_from_df(X_test_df)

    X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test)

    classifier = SVC(random_state=RANDOM_SEED, max_iter=300, cache_size=1000, **args.fit_params)

    if args.cohorts:
        precision = []
        recall = []
        f1 = []
        roc_auc = []
        importances = []
        coefs = []

        subject_data = pd.read_csv(SUBJECTS_DATA_FPATH).set_index('subject')
        X_df['cohort'] = X_df.apply(lambda row: subject_data.loc[row.name.split('_')[0], 'cohort'], axis=1)

        for cohort in range(1, 6):
            print('Starting cohort', cohort)
            X_df_cohort = X_df[X_df['cohort'] == cohort].drop('cohort', axis=1)
            y_cohort = y[X_df['cohort'] == cohort]
            c_precision, c_recall, c_f1, c_roc_auc, c_importances, c_coefs = run_cv(classifier, X_df_cohort, y_cohort,
                                                                                    nested=args.nested_cv)
            precision.append(c_precision)
            recall.append(c_recall)
            f1.append(c_f1)
            roc_auc.append(c_roc_auc)
            importances.append(importances, axis=0)
            coefs.append(c_coefs)
    else:
        precision, recall, f1, roc_auc, importances, coefs = run_cv(classifier, X_df, y, nested=args.nested_cv)

    print('--- Total cross-validation performance ---')
    print('Precision: %.2f, STD %.2f' % (np.nanmean(precision), np.nanstd(precision)))
    print('Recall: %.2f, STD %.2f' % (np.nanmean(recall), np.nanstd(recall)))
    print('F1 score: %.2f, STD %.2f' % (np.nanmean(f1), np.nanstd(f1)))
    print('ROC AUC: %.2f, STD %.2f' % (np.nanmean(roc_auc), np.nanstd(roc_auc)))

    if not args.nested_cv:
        classifier.fit(X_train, y_train)
        plot_results(classifier, X_train, X_test, y_train, y_test, feature_names, importances, coefs, parc=args.parc)


if __name__ == "__main__":
    main()
