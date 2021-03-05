#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

import os
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from preprocessing import get_tmap_dataset, features_from_df, preprocess, FeatureSelector
from inspection import *
from cross_validate import run_cv, run_grid, performance, prediction_per_subject, custom_cv

# define file paths
DATA_DIR = '/scratch/nbe/tbi-meg/veera/tmap-data/'
FULL_NORMATIVE_FILENAME = 'tmap_data_aparc_sub_f8_absolute.csv'
AGE_COHORT_FILENAME = 'tmap_data_aparc_sub_f8_absolute_cohort.csv'
RANDOM_COHORT_FILENAME = 'tmap_data_aparc_sub_f8_random_cohort.csv'
# SUBJECTS_DATA_FPATH = 'subject_demographics.csv'

# define random seed for reproducibility
RANDOM_SEED = 50
# number of cross-validation splits
CV = 7
# fraction of data used for testing, if nested CV is not used
TEST_SPLIT = 0.1

# threshold for hierarchical clustering
CLUSTER_THRESH = 2
# number of features to select by mutual information criterion after the clustering,
# use 'all' to skip feature selection by mutual information
N_FEATURES = 'all'
# number of permutations in calculating permutation feature importance
N_PERM_REPEATS = 7

# the ids of the subjects labelled as positive
cases = ['%03d' % n for n in range(1, 28)]

# definition of the frequency bands
freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (17, 30),
    'gamma': (30, 40)
}

np.random.seed(RANDOM_SEED)
pd.set_option('use_inf_as_na', True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-f', '--fit-params', help='Parameters to pass to the classifier (JSON string)',
                        type=json.loads, default='{}')
    parser.add_argument('-g', '--grid', help='Use grid search for model selection',
                        action='store_true', default=False)
    parser.add_argument('-r', '--repeat', help='Use repeated cross-validation', action='store_true', default=False)
    parser.add_argument('-n', '--nested', help='Use nested cross-validation for model selection and validation',
                        action='store_true', default=False)
    parser.add_argument('--fs', help='Apply feature selection', action='store_true', default=False)
    parser.add_argument('--norm-data', help='Select what normative data to use', choices=['full', 'age', 'random'],
                        default='full')
    parser.add_argument('-v', '--visualize', help='Visualize the results', action='store_true', default=False)
    parser.add_argument('-p', '--perm-test', help='Use permutation test', action='store_true', default=False)

    args = parser.parse_args()

    # select data file based on the selected normative data
    if args.norm_data == 'full':
        data_fpath = os.path.join(DATA_DIR, FULL_NORMATIVE_FILENAME)
    elif args.norm_data == 'age':
        data_fpath = os.path.join(DATA_DIR, AGE_COHORT_FILENAME)
    elif args.norm_data == 'random':
        data_fpath = os.path.join(DATA_DIR, RANDOM_COHORT_FILENAME)

    print('Fetching data from file', data_fpath)
    # get the data matrix and class label of each sample (7 per subject)
    X_df, y = get_tmap_dataset(data_fpath, cases, freq_bands)

    subjects = list(dict.fromkeys([s.split('_')[0] for s in X_df.index]))
    # get the class label of each subject
    subs_y = np.asarray([1 if s[:3] in cases else 0 for s in subjects])
    subjects = np.asarray(subjects)

    if not args.nested:
        # if not using nested cross-validation, split the data into train and test sets
        subjects_train, subjects_test = train_test_split(subjects, test_size=TEST_SPLIT, random_state=RANDOM_SEED,
                                                         stratify=subs_y)
        subs_y_train = np.asarray([1 if s[:3] in cases else 0 for s in list(subjects_train)])
        subs_y_test = np.asarray([1 if s[:3] in cases else 0 for s in list(subjects_test)])
        train_idx = X_df.index.str.split(pat='_').str[0].isin(subjects_train)
        test_idx = X_df.index.str.split(pat='_').str[0].isin(subjects_test)
        X_train_df = X_df[train_idx]
        y_train = y[train_idx]
        X_test_df = X_df[test_idx]
        y_test = y[test_idx]
    else:
        # use the whole dataset in nested cross-validation
        X_train_df = X_df
        y_train = y
        subs_y_train = subs_y
        X_test_df = None

    # convert the pandas data frame to a numpy array
    X_train, feature_names = features_from_df(X_train_df)

    fit_params = args.fit_params
    classifier = SVC(random_state=RANDOM_SEED, kernel='rbf', class_weight='balanced', max_iter=-1, cache_size=1000, **fit_params)
    importances, coefs, search_space = None, None, None

    if args.grid or args.nested:
        # search space for the model hyperparameters
        search_space = [
            {'svc__gamma': [1e-1, 1e-2, 1e-3],
             'svc__C': [1, 5, 10]}
        ]
    if args.fs:
        # the feature selection step to be added to Scikit-learn pipeline
        selector = FeatureSelector(clf=classifier, k=N_FEATURES, thresh=CLUSTER_THRESH)
    else:
        # in Scikit-learn, a 'passthrough' pipeline step does nothing
        selector = 'passthrough'

    if args.perm_test:
        # test the statistical significance by a permutation test
        print('Starting permutation test')
        n = 100 # number of permutations
        null_dist = np.zeros(n)
        for i in range(n):
            print(f'{i + 1}/{n}')
            subs_y_perm = np.random.permutation(subs_y_train)
            y_perm = np.repeat(subs_y_perm, 7)
            pipeline, results_dict, _, _ = run_cv(classifier, X_train_df, y_perm, subs_y_perm, selector=selector,
                                                  n_splits=CV, nested=args.nested, rep_cv=args.repeat,
                                                  param_grid=search_space, random_state=RANDOM_SEED)
            null_dist[i] = np.nanmean(results_dict['accuracy'])
        print(np.quantile(null_dist, 0.95))
        return

    # run cross-validation
    pipeline, results_dict, importances, results_df = run_cv(classifier, X_train_df, y_train, subs_y_train,
                                                             selector=selector, n_splits=CV, nested=args.nested,
                                                             rep_cv=args.repeat, param_grid=search_space,
                                                             random_state=RANDOM_SEED)

    # save average classification performance and decision function values per subject
    results_fname = f'reports/subject_correct_predictions_{args.norm_data}.csv'
    if args.fs:
        results_fname = results_fname.replace('.csv', '_fs.csv')
    results_df.to_csv(results_fname)

    print('--- Total cross-validation performance ---')
    print('Accuracy: %.3f, STD %.2f' % (np.nanmean(results_dict['accuracy']), np.nanstd(results_dict['accuracy'])))
    print('Recall: %.3f, STD %.2f' % (np.nanmean(results_dict['recall']), np.nanstd(results_dict['recall'])))
    print('Specificity: %.3f, STD %.2f' % (np.nanmean(results_dict['specif']), np.nanstd(results_dict['specif'])))
    print('F1 score: %.3f, STD %.2f' % (np.nanmean(results_dict['f1']), np.nanstd(results_dict['f1'])))
    print('ROC AUC: %.3f, STD %.2f' % (np.nanmean(results_dict['roc_auc']), np.nanstd(results_dict['roc_auc'])))

    pipeline.fit(X_train, y_train)
    if X_test_df is not None:
        # test using an independent test set
        X_test, _ = features_from_df(X_test_df)
        X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test)
        y_pred = pipeline.predict(X_test)
        try:
            y_scores = pipeline.decision_function(X_test)
        except:
            y_scores = np.max(pipeline.predict_proba(X_test), axis=1)

        y_pred_subjects, y_scores_subjects = prediction_per_subject(subs_y_test, y_pred, y_scores,
                                                                    list(X_test_df.index), subjects_test)
        val_accuracy, val_precision, val_recall, val_specif, val_f1, val_roc_auc = performance(subs_y_test,
                                                                                               y_pred_subjects,
                                                                                               y_scores_subjects)
        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        print('--- Testing performance ---')
        print('Accuracy: %.2f' % val_accuracy)
        print('Recall: %.2f' % val_recall)
        print('Specificity: %.2f' % val_specif)
        print('F1 score: %.2f' % val_f1)
        print('ROC AUC: %.2f' % val_roc_auc)
    else:
        X = X_train

    if args.visualize:
        # define figure file paths
        corr_fig_fpath = f'fig/tmap_corr_{args.norm_data}.png'
        separation_fig_fpath = f'fig/tmap_class_separation_{args.norm_data}.png'
        importance_fig_fpath = f'fig/tmap_feature_importance_{args.norm_data}.png'
        importance_txt_fpath = f'reports/tmap_feature_importance_{args.norm_data}.txt'
        pdp_fpath = f'fig/tmap_partial_dependence_{args.norm_data}.png'

        print('Visualizing...')

        plot_correlation(X_train, feature_names, corr_fig_fpath)

        if not args.nested:
            plot_class_separation(pipeline, X_train, X_test, y_train, y_test, separation_fig_fpath)

        if importances is not None:
            save_feature_importance(importances, feature_names, importance_fig_fpath, importance_txt_fpath)
            perm_sorted_idx = np.mean(importances, axis=-1).argsort()[::-1][:12]
            if selector != 'passthrough':
                estimator = pipeline
            else:
                estimator = pipeline[1]
            plot_pdp(estimator, X, y, perm_sorted_idx, pdp_fpath, n_cols=3, feature_names=feature_names)


if __name__ == "__main__":
    main()
