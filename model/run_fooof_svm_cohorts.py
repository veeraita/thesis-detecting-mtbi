#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

import os
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.random import sample_without_replacement
from sklearn.utils import shuffle

from preprocessing import get_dataset, features_from_df, preprocess, feature_selection
from inspection import *

DATA_DIR = '/scratch/nbe/tbi-meg/veera/fooof_data/'
DATA_FILENAME = 'meg_{}_features_{}_window_v2.h5'
SUBJECTS_DATA_FPATH = 'subject_demographics.csv'
RANDOM_SEED = 42
CV = 5
TEST_SPLIT = 0.2
CLUSTER_THRESH = 10
N_FEATURES = 100
N_PCA_COMP = 50
N_PERM_REPEATS = 5
SELECT_FEATURES = True

CASES = ['%03d' % n for n in range(28)]


def plot_results(feature_names, feature_importances, cohort):
    importance_fig_fpath = f'fig/feature_importance_cohort_{cohort}_thresh_{CLUSTER_THRESH}.png'
    importance_txt_fpath = f'reports/feature_importance_cohort_{cohort}_thresh_{CLUSTER_THRESH}.txt'
    save_feature_importance(feature_importances, feature_names, importance_fig_fpath, importance_txt_fpath)


@ignore_warnings(category=ConvergenceWarning)
def run_cv_cohort(clf, X_df, y, cohort, average='binary', select_features=True, pca=False, match=False):
    X_df_s, y_s = shuffle(X_df, y, random_state=RANDOM_SEED)
    X_s, feature_names_orig = features_from_df(X_df_s)

    mask_cohort = (X_df_s['cohort'] == cohort)
    X_cohort = X_s[mask_cohort]
    X_df_cohort = X_df_s[mask_cohort]
    y_cohort = y_s[mask_cohort]

    mask_other = (X_df_s['cohort'] != cohort) & (X_df_s['cohort'] < 6)
    # mask_other = (X_df_s['cohort'] < 6)
    # mask_other = (X_df_s['cohort'] != cohort)
    X_other = X_s[mask_other]
    X_df_other = X_df_s[mask_other]
    y_other = y_s[mask_other]
    # X_other = X_s
    # X_df_other = X_df_s
    # y_other = y_s

    cv = StratifiedKFold(n_splits=CV, shuffle=True, random_state=RANDOM_SEED)

    print(f'Starting {cv.n_splits}-fold cross validation')
    print(clf)
    precision = []
    recall = []
    f1 = []
    roc_auc = []
    importances = np.zeros((cv.n_splits, X_s.shape[1], N_PERM_REPEATS))
    coefs = np.zeros((cv.n_splits, X_s.shape[1]))

    for ii, (train, test) in enumerate(cv.split(X_cohort, y=y_cohort)):
        print('Fold', ii + 1)

        if match:
            # Check that the train and test sets are completely separate
            assert len(pd.merge(X_df_cohort.iloc[train], X_df_cohort.iloc[test], how='inner', on=['subject'])) == 0
            X_train, y_train, X_test, y_test = preprocess(X_cohort[train], y_cohort[train], X_cohort[test],
                                                          y_cohort[test])
        else:
            assert len(pd.merge(X_df_other.iloc[train], X_df_other.iloc[test], how='inner', on=['subject'])) == 0
            X_train, y_train, X_test, y_test = preprocess(X_other[train], y_other[train], X_cohort[test], y_cohort[test])
            # print(X_df_other.iloc[train][y_train == 1])
            # print(X_df_cohort.iloc[test][y_test == 1])

        X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
        if pca:
            pca = PCA(N_PCA_COMP, random_state=RANDOM_SEED)
            X_train = pca.fit_transform(X_train_orig)
            X_test = pca.transform(X_test_orig)
            var = np.around(np.cumsum(pca.explained_variance_ratio_), 3)[-1] * 100
            print(f'Selected {pca.n_components_} PCA components, explaining {var} % of variance')
        elif select_features:
            X_train, X_test, feature_names, feature_inds = feature_selection(X_train_orig, X_test_orig, y_train,
                                                                             feature_names_orig,
                                                                             k=N_FEATURES, thresh=CLUSTER_THRESH)

        clf.fit(X_train, y_train)
        print("Training score:", clf.score(X_train, y_train))
        print("Testing score:", clf.score(X_test, y_test))

        if select_features and match:
            importances[ii, feature_inds] = feature_importance(clf, X_test, y_test, n_repeats=N_PERM_REPEATS,
                                                               random_state=RANDOM_SEED)
        y_pred = clf.predict(X_test)
        try:
            y_scores = clf.decision_function(X_test)
        except:
            y_scores = np.max(clf.predict_proba(X_test), axis=1)

        # for i in range(len(y_pred)):
        #    print(y_test[i], y_pred[i], y_scores[i])
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
           np.mean(importances, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--parc', help='The cortical parcellation to use', choices=['aparc', 'aparc_sub'],
                        default='aparc_sub')
    parser.add_argument('-t', '--target', help='Whether to do the analysis subject-wise or parcel-wise',
                        choices=['subjects', 'parcels'], default='subjects')
    parser.add_argument('-f', '--fit-params', help='Parameters to pass to the classifier (JSON string)',
                        type=json.loads, default='{}')
    parser.add_argument('--pca', help='Use PCA for feature selection', action='store_true', default=False)
    parser.add_argument('-m', '--match', help='Run CV with age-matched samples', action='store_true', default=False)

    args = parser.parse_args()

    data_filename = DATA_FILENAME.format(args.parc, args.target)
    data_key = data_filename.split('.')[0]
    data_fpath = os.path.join(DATA_DIR, data_filename)

    X_df, y, names = get_dataset(data_fpath, data_key, CASES, column_filter='_amp|_width|alpha_freq|exponent')

    X, feature_names = features_from_df(X_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED,
                                                              stratify=y)

    classifier = SVC(random_state=RANDOM_SEED, max_iter=800, cache_size=1000, verbose=True, **args.fit_params)

    subject_data = pd.read_csv(SUBJECTS_DATA_FPATH).set_index('subject')
    X_df['cohort'] = X_df.apply(lambda row: subject_data.loc[row.name.split('_')[0], 'cohort'], axis=1)

    precision = []
    recall = []
    f1 = []
    roc_auc = []
    for cohort in range(1, 6):
        print('Starting cohort', cohort)
        c_precision, c_recall, c_f1, c_roc_auc, c_importances = run_cv_cohort(classifier, X_df, y, cohort,
                                                                              select_features=SELECT_FEATURES,
                                                                              pca=args.pca, match=args.match)

        print(f'--- Cross-validation performance for cohort {cohort} ---')
        print('Precision: %.2f, STD %.2f' % (np.nanmean(c_precision), np.nanstd(c_precision)))
        print('Recall: %.2f, STD %.2f' % (np.nanmean(c_recall), np.nanstd(c_recall)))
        print('F1 score: %.2f, STD %.2f' % (np.nanmean(c_f1), np.nanstd(c_f1)))
        print('ROC AUC: %.2f, STD %.2f' % (np.nanmean(c_roc_auc), np.nanstd(c_roc_auc)))

        precision.append(c_precision)
        recall.append(c_recall)
        f1.append(c_f1)
        roc_auc.append(c_roc_auc)

        if not args.pca and args.match and SELECT_FEATURES:
            plot_results(feature_names, c_importances, cohort)

    print('--- Total cross-validation performance ---')
    print('Precision: %.2f, STD %.2f' % (np.nanmean(precision), np.nanstd(precision)))
    print('Recall: %.2f, STD %.2f' % (np.nanmean(recall), np.nanstd(recall)))
    print('F1 score: %.2f, STD %.2f' % (np.nanmean(f1), np.nanstd(f1)))
    print('ROC AUC: %.2f, STD %.2f' % (np.nanmean(roc_auc), np.nanstd(roc_auc)))


if __name__ == "__main__":
    main()
