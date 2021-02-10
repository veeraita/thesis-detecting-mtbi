#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

import os
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from preprocessing import get_tmap_dataset, features_from_df, preprocess, feature_selection, FeatureSelector
from inspection import *
from cross_validate import run_cv, run_cv_pca, run_grid, performance, prediction_per_subject, custom_cv

TYPE = 'random'
DATA_DIR = '/scratch/nbe/tbi-meg/veera/tmap-data/'
DATA_FILENAME = f'tmap_data_aparc_sub_f8_{TYPE}.csv'
SUBJECTS_DATA_FPATH = 'subject_demographics.csv'
RANDOM_SEED = 50
CV = 7
TEST_SPLIT = 0.1

CLUSTER_THRESH = 2
N_FEATURES = 'all'
N_PCA_COMP = 4
N_PERM_REPEATS = 7
LINEAR_FS = False
CONFOUNDS = False

cases = ['%03d' % n for n in range(1, 28)]

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
    parser.add_argument('--pca', help='Use PCA before fitting the model', action='store_true', default=False)
    parser.add_argument('-g', '--grid', help='Use grid search for model selection',
                        action='store_true', default=False)
    parser.add_argument('-r', '--repeat', help='Use repeated cross-validation', action='store_true', default=False)
    parser.add_argument('-n', '--nested', help='Use nested cross-validation for model selection and validation',
                        action='store_true', default=False)
    parser.add_argument('--fs', help='Apply feature selection', action='store_true', default=False)
    parser.add_argument('-c', '--cohorts', help='Use cohort-matched data', action='store_true', default=False)
    parser.add_argument('-v', '--visualize', help='Visualize the results', action='store_true', default=False)

    args = parser.parse_args()

    data_fpath = os.path.join(DATA_DIR, DATA_FILENAME)

    ext = ''
    if args.cohorts:
        data_fpath = data_fpath.replace('.csv', '_cohort.csv')
        ext = '_cohort'
    print('Fetching data from file', data_fpath)
    X_df, y, names = get_tmap_dataset(data_fpath, cases, freq_bands, SUBJECTS_DATA_FPATH)

    subjects = list(dict.fromkeys([s.split('_')[0] for s in X_df.index]))
    subs_y = np.asarray([1 if s[:3] in cases else 0 for s in list(subjects)])
    subjects = np.asarray(subjects)

    if not args.nested:
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
        X_train_df = X_df
        y_train = y
        X_test_df = None

    X_train, _, feature_names = features_from_df(X_train_df)

    fit_params = args.fit_params
    classifier = SVC(random_state=RANDOM_SEED, class_weight='balanced', max_iter=-1, cache_size=1000, **fit_params)
    importances, coefs, pca, search_space = None, None, None, None

    if args.pca:
        if args.grid or args.nested:
            search_space = [
                {'svc__gamma': [1e-2, 1e-3, 1e-4],
                 'svc__C': [1, 5, 10],
                 'pca__n_components': [3, 5, 7, 9]}
            ]
        pca = PCA(N_PCA_COMP)
        pipeline, accuracy, precision, recall, specif, f1, roc_auc, results_df = run_cv_pca(classifier, pca, X_train_df,
                                                                                            y_train, cases,
                                                                                            param_grid=search_space,
                                                                                            n_splits=CV,
                                                                                            nested=args.nested,
                                                                                            rep_cv=args.repeat,
                                                                                            random_state=RANDOM_SEED)
    else:
        if args.grid or args.nested:
            search_space = [
                {'svc__gamma': [1e-1, 1e-2, 1e-3],
                 'svc__C': [1, 5, 10]}
            ]
        if args.fs:
            selector = FeatureSelector(clf=classifier, k=N_FEATURES, thresh=CLUSTER_THRESH, linear=LINEAR_FS, step=5)
        else:
            selector = 'passthrough'
        pipeline, accuracy, precision, recall, specif, f1, roc_auc, importances, results_df = run_cv(classifier,
                                                                                                     X_train_df,
                                                                                                     y_train, cases,
                                                                                                     selector=selector,
                                                                                                     n_splits=CV,
                                                                                                     nested=args.nested,
                                                                                                     rep_cv=args.repeat,
                                                                                                     param_grid=search_space,
                                                                                                     random_state=RANDOM_SEED)
    if args.cohorts:
        results_fname = 'reports/subject_correct_predictions_cohorts.csv'
    else:
        results_fname = 'reports/subject_correct_predictions.csv'
    if args.fs:
        results_fname = results_fname.replace('.csv', '_fs.csv')
    results_df.to_csv(results_fname)

    print('--- Total cross-validation performance ---')
    print('Accuracy: %.3f, STD %.2f' % (np.nanmean(accuracy), np.nanstd(accuracy)))
    print('Precision: %.3f, STD %.2f' % (np.nanmean(precision), np.nanstd(precision)))
    print('Recall: %.3f, STD %.2f' % (np.nanmean(recall), np.nanstd(recall)))
    print('Specificity: %.3f, STD %.2f' % (np.nanmean(specif), np.nanstd(specif)))
    print('F1 score: %.3f, STD %.2f' % (np.nanmean(f1), np.nanstd(f1)))
    print('ROC AUC: %.3f, STD %.2f' % (np.nanmean(roc_auc), np.nanstd(roc_auc)))

    pipeline.fit(X_train, y_train)
    if X_test_df is not None:
        X_test, _, _ = features_from_df(X_test_df)
        # X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test)
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
        print('Precision: %.2f' % val_precision)
        print('Recall: %.2f' % val_recall)
        print('Specificity: %.2f' % val_specif)
        print('F1 score: %.2f' % val_f1)
        print('ROC AUC: %.2f' % val_roc_auc)
    else:
        X = X_train

    if not isinstance(pipeline[0], str):
        X_trans = pipeline[0].transform(X)
    else:
        X_trans = X

    if args.visualize:
        corr_fig_fpath = f'fig/tmap_{TYPE}_corr{ext}.png'
        separation_fig_fpath = f'fig/tmap_{TYPE}_class_separation{ext}.png'
        importance_fig_fpath = f'fig/tmap_{TYPE}_feature_importance{ext}.png'
        importance_txt_fpath = f'reports/tmap_{TYPE}_feature_importance{ext}.txt'
        pca_var_fig_fpath = f'fig/tmap_{TYPE}_pca_variance{ext}.png'
        pca_loadings_fig_fpath = f'fig/tmap_{TYPE}_pca_loadings{ext}.png'
        pca_loadings_fpath = f'reports/tmap_{TYPE}_pca_loadings{ext}.csv'
        pdp_fpath = f'fig/tmap_{TYPE}_partial_dependence{ext}.png'
        pdp_pca_fpath = f'fig/tmap_{TYPE}_pca_partial_dependence{ext}.png'

        print('Visualizing...')

        plot_correlation(X_train, feature_names, corr_fig_fpath)
        if args.pca:
            plot_pca_variance(X_train, pca_var_fig_fpath)

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
        if args.pca:
            save_pca_loadings(pipeline.named_steps['pca'], X_trans[2::7, :4], subs_y, feature_names,
                              pca_loadings_fig_fpath,
                              pca_loadings_fpath)
            plot_pdp(pipeline, X[2::7], subs_y, [0, 1, 2, 3], pdp_pca_fpath, figsize=(10, 4), n_cols=4,
                     feature_names=[f'PCA comp. {i + 1}' for i in range(4)])


if __name__ == "__main__":
    main()
