#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

import os
import argparse
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from preprocessing import get_dataset, features_from_df, preprocess, feature_selection, feature_selection_by_rfe, FeatureSelector
from inspection import *

DATA_DIR = '/scratch/nbe/tbi-meg/veera/fooof_data/'
DATA_FILENAME = 'meg_{}_features_{}_window_v3.h5'
SUBJECTS_DATA_FPATH = 'subject_demographics.csv'
RANDOM_SEED = 50
CV = 7
TEST_SPLIT = 0.1

CLUSTER_THRESH = 5
N_FEATURES = 130
N_PCA_COMP = 50
N_PERM_REPEATS = 7
SELECT_FEATURES = True
LINEAR_FS = False
CONFOUNDS = True

CASES = ['%03d' % n for n in range(1, 28)]
TBI_CONTROLS = ['%03d' % n for n in range(28, 48)]

np.random.seed(RANDOM_SEED)
pd.set_option('use_inf_as_na', True)


def plot_results(clf, X_train, X_test, y_train, y_test, feature_names, feature_importances, linear_coefs,
                 parc='aparc_sub', sites=False):
    corr_fig_fpath = 'fig/corr.png'
    separation_fig_fpath = 'fig/class_separation.png'
    importance_fig_fpath = f'fig/feature_importance_{parc}_thresh_{CLUSTER_THRESH}.png'
    importance_txt_fpath = f'reports/feature_importance_{parc}_thresh_{CLUSTER_THRESH}.txt'
    coef_fig_fpath = 'fig/coef.png'

    if sites:
        plot_class_separation(clf, X_train, X_test, y_train, y_test, separation_fig_fpath.replace('.png', '_sites.png'))
        save_feature_importance(feature_importances, feature_names, importance_fig_fpath.replace('.png', '_sites.png'),
                                importance_txt_fpath.replace('.png', '_sites.png'))
        plot_linear_coefs(linear_coefs, feature_names, coef_fig_fpath.replace('.png', '_sites.png'))
    else:
        plot_correlation(X_train, feature_names, corr_fig_fpath)
        plot_class_separation(clf, X_train, X_test, y_train, y_test, separation_fig_fpath)
        save_feature_importance(feature_importances, feature_names, importance_fig_fpath, importance_txt_fpath)
        plot_linear_coefs(linear_coefs, feature_names, coef_fig_fpath)


def performance(y_test, y_pred, y_scores, average='binary'):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1, average=average)
    recall = recall_score(y_test, y_pred, zero_division=1, average=average)
    f1 = f1_score(y_test, y_pred, zero_division=1, average=average)
    try:
        roc_auc = roc_auc_score(y_test, y_scores)
    except ValueError:
        roc_auc = np.nan

    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['negative', 'positive'],
                                zero_division=1))
    return accuracy, precision, recall, f1, roc_auc


def performance_per_subject(y_test_subjects, y_pred, y_scores, sample_names, subjects, average='binary'):
    y_pred_subjects = []
    y_scores_subjects = []
    for s in subjects:
        sum = 0
        scores = []
        for i in range(len(sample_names)):
            if sample_names[i].split('_')[0] == s:
                sum += y_pred[i]
                scores.append(y_scores[i])
        if sum > (len(y_pred) / len(y_test_subjects)) / 2:
            y_pred_subjects.append(1)
        else:
            y_pred_subjects.append(0)
        y_scores_subjects.append(np.mean(scores))

    for i in range(len(subjects)):
        print(y_test_subjects[i], y_pred_subjects[i], subjects[i])
    accuracy = accuracy_score(y_test_subjects, y_pred_subjects)
    precision = precision_score(y_test_subjects, y_pred_subjects, zero_division=1, average=average)
    recall = recall_score(y_test_subjects, y_pred_subjects, zero_division=1, average=average)
    f1 = f1_score(y_test_subjects, y_pred_subjects, zero_division=1, average=average)
    try:
        roc_auc = roc_auc_score(y_test_subjects, y_scores_subjects)
    except ValueError:
        roc_auc = np.nan

    print(classification_report(y_test_subjects, y_pred_subjects, labels=[0, 1], target_names=['negative', 'positive'],
                                zero_division=1))
    return accuracy, precision, recall, f1, roc_auc


def custom_cv(cv, X_df, subjects, subs_y):
    inds = []
    subject_wise_inds = []
    for ii, (train, test) in enumerate(cv.split(subjects, y=subs_y)):
        train_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[train])
        test_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[test])
        inds.append((train_idx, test_idx))
        subject_wise_inds.append((train, test))
    return inds, subject_wise_inds


def run_grid(clf, X, y, cv_inds, param_space, random_state=1):
    selector = FeatureSelector(clf, step=5, random_state=random_state)
    pipe = Pipeline(steps=[('selector', selector), ('clf', clf)])
    grid = GridSearchCV(pipe, param_space, scoring='accuracy', cv=cv_inds, n_jobs=4, verbose=1)
    grid.fit(X, y)
    print('Best params:', grid.best_params_)
    print('Estimated score: %.3f\n' % grid.best_score_)
    best_clf = grid.best_estimator_.named_steps['clf']
    best_feature_inds = grid.best_estimator_.named_steps['selector'].get_support(indices=True)

    return best_clf, best_feature_inds


@ignore_warnings(category=ConvergenceWarning)
def run_cv(clf, X_df, y, average='binary', select_features=True, nested=False, grid=False):
    cv_outer = StratifiedKFold(n_splits=CV, shuffle=True, random_state=RANDOM_SEED)
    print(f'Starting {cv_outer.n_splits}-fold cross validation')
    print(clf)
    if isinstance(clf, LogisticRegression):
        search_space = [
            {'C': [1, 10, 100],
             'solver': ['liblinear'],
             'penalty': ['l1', 'l2'],
             'class_weight': [None]},
            {'C': [1, 10, 100],
             'solver': ['lbfgs'],
             'class_weight': [None]}
        ]
    else:
        search_space = [
            {'clf__kernel': ['rbf'],
             'clf__gamma': [1e-3, 1e-4],
             'clf__C': [1, 10, 100],
             'clf__class_weight': [None],
             'selector__thresh': [4, 5, 6],
             'selector__k': [50, 100, 150]}
        ]
    accuracy = []
    precision = []
    recall = []
    f1 = []
    roc_auc = []
    feature_selection_scores = defaultdict(list)
    importances = np.zeros((cv_outer.n_splits, X_df.drop(['cohort'], axis=1).shape[1], N_PERM_REPEATS))
    coefs = np.zeros((cv_outer.n_splits, X_df.drop(['cohort'], axis=1).shape[1]))

    subjects = list(dict.fromkeys([s.split('_')[0] for s in X_df.index]))
    subs_y = np.asarray([1 if s[:3] in CASES else 0 for s in list(subjects)])
    subjects = np.asarray(subjects)
    cv_inds, sub_inds = custom_cv(cv_outer, X_df, subjects, subs_y)
    if grid:
        X, confounds, feature_names_orig = features_from_df(X_df)
        run_grid(clf, X, y, cv_inds, search_space, random_state=RANDOM_SEED)

    for ii, (train, test),  in enumerate(cv_outer.split(subjects, y=subs_y)):
        print('Fold', ii + 1)
        train_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[train])
        test_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[test])
        X_df_train = X_df[train_idx]
        y_train = y[train_idx]
        X_df_test = X_df[test_idx]
        y_test = y[test_idx]
        X_train, confounds_train, feature_names_orig = features_from_df(X_df_train)
        X_test, confounds_test, _ = features_from_df(X_df_test)
        if CONFOUNDS:
            confounds = np.vstack((confounds_train, confounds_test))
        else:
            confounds = None
        X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test, confounds)
        X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
        best_clf = clf
        best_feature_inds = None

        if select_features:
            X_train, X_test, feature_names, feature_inds = feature_selection(X_train_orig, X_test_orig, y_train,
                                                                             feature_names_orig, clf=best_clf,
                                                                             k=N_FEATURES, thresh=CLUSTER_THRESH,
                                                                             step=5, random_state=RANDOM_SEED,
                                                                             linear=LINEAR_FS)
            best_feature_inds = feature_inds
            X_train = X_train_orig[:, best_feature_inds]
            X_test = X_test_orig[:, best_feature_inds]
        best_clf.fit(X_train, y_train)

        if select_features and N_FEATURES != 'all':
            importances[ii, best_feature_inds] = feature_importance(best_clf, X_test, y_test, n_repeats=N_PERM_REPEATS,
                                                                    random_state=RANDOM_SEED)
        # Find linear coefficients
        if isinstance(clf, LogisticRegression):
            classifier_linear = clf
        else:
            linear_params = best_clf.get_params()
            classifier_linear = SVC(kernel='linear', C=linear_params['C'], class_weight=linear_params['class_weight'],
                                    random_state=RANDOM_SEED)
        classifier_linear.fit(X_train, y_train)
        coef = classifier_linear.coef_[0]
        if best_feature_inds is not None:
            coefs[ii, best_feature_inds] = coef
        else:
            coefs[ii] = coef

        y_pred = best_clf.predict(X_test)
        try:
            y_scores = best_clf.decision_function(X_test)
        except:
            y_scores = np.max(best_clf.predict_proba(X_test), axis=1)
        for i in range(len(y_pred)):
            print(y_test[i], y_pred[i], '%.3f' % y_scores[i], X_df_test.index[i])
        cv_accuracy, cv_precision, cv_recall, cv_f1, cv_roc_auc = performance_per_subject(subs_y[test], y_pred,
                                                                                          y_scores,
                                                                                          list(X_df_test.index),
                                                                                          subjects[test],
                                                                                          average=average)

        # cv_accuracy, cv_precision, cv_recall, cv_f1, cv_roc_auc = performance(y_test, y_pred, y_scores, average=average)

        accuracy.append(cv_accuracy)
        precision.append(cv_precision)
        recall.append(cv_recall)
        f1.append(cv_f1)
        roc_auc.append(cv_roc_auc)

    if nested:
        averages = {key: sum(score) / len(score) for key, score in feature_selection_scores.items()}
        print(averages)

    return np.asarray(accuracy), np.asarray(precision), np.asarray(recall), np.asarray(f1), np.asarray(roc_auc), \
           np.mean(importances, axis=0), np.mean(coefs, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--parc', help='The cortical parcellation to use', choices=['aparc', 'aparc_sub'],
                        default='aparc_sub')
    parser.add_argument('-t', '--target', help='Whether to do the analysis subject-wise or parcel-wise',
                        choices=['subjects', 'parcels'], default='subjects')
    parser.add_argument('-f', '--fit-params', help='Parameters to pass to the classifier (JSON string)',
                        type=json.loads, default='{}')
    parser.add_argument('-g', '--grid', help='Use grid search for model selection',
                        action='store_true', default=False)
    parser.add_argument('-s', '--sites', help='Classify measuring sites instead of patients vs. healthy subjects',
                        action='store_true', default=False)

    args = parser.parse_args()

    data_filename = DATA_FILENAME.format(args.parc, args.target)
    data_key = data_filename.split('.')[0]
    data_fpath = os.path.join(DATA_DIR, data_filename)

    if args.sites:
        cases = TBI_CONTROLS
        index_filter = '^sub|^03|^04|^028|^029'
    else:
    cases = CASES
    index_filter = '^0'
    # index_filter = None

    X_df, y, names = get_dataset(data_fpath, data_key, cases, SUBJECTS_DATA_FPATH,
                                 column_filter='alpha_amp|beta_amp|alpha_width|beta_width|alpha_freq|beta_freq|exponent',
                                 index_filter=index_filter)

    subjects = list(dict.fromkeys([s.split('_')[0] for s in X_df.index]))
    subs_y = np.asarray([1 if s[:3] in CASES else 0 for s in list(subjects)])
    subjects = np.asarray(subjects)

    subjects_train, subjects_test = train_test_split(subjects, test_size=TEST_SPLIT, random_state=RANDOM_SEED,
                                                    stratify=subs_y)
    subs_y_train = np.asarray([1 if s[:3] in CASES else 0 for s in list(subjects_train)])
    subs_y_test = np.asarray([1 if s[:3] in CASES else 0 for s in list(subjects_test)])
    train_idx = X_df.index.str.split(pat='_').str[0].isin(subjects_train)
    test_idx = X_df.index.str.split(pat='_').str[0].isin(subjects_test)
    X_train_df = X_df[train_idx]
    y_train = y[train_idx]
    X_test_df = X_df[test_idx]
    y_test = y[test_idx]

    X_train, confounds_train, feature_names = features_from_df(X_train_df)
    X_test, confounds_test, _ = features_from_df(X_test_df)

    if CONFOUNDS:
        confounds = np.vstack((confounds_train, confounds_test))
    else:
        confounds = None
    X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test, confounds)

    fit_params = args.fit_params
    classifier = SVC(random_state=RANDOM_SEED, max_iter=-1, cache_size=1000, **fit_params)
    # classifier = RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=200, **args.fit_params)
    # classifier = LogisticRegression(random_state=RANDOM_SEED, max_iter=600, **args.fit_params)

    accuracy, precision, recall, f1, roc_auc, importances, coefs = run_cv(classifier, X_train_df, y_train,
                                                                          select_features=SELECT_FEATURES,
                                                                          nested=args.nested_cv, grid=args.grid)
    save_linear_coefs(coefs, feature_names, 'reports/coefs.csv')

    print('--- Total cross-validation performance ---')
    print('Accuracy: %.2f, STD %.2f' % (np.nanmean(accuracy), np.nanstd(accuracy)))
    print('Precision: %.2f, STD %.2f' % (np.nanmean(precision), np.nanstd(precision)))
    print('Recall: %.2f, STD %.2f' % (np.nanmean(recall), np.nanstd(recall)))
    print('F1 score: %.2f, STD %.2f' % (np.nanmean(f1), np.nanstd(f1)))
    print('ROC AUC: %.2f, STD %.2f' % (np.nanmean(roc_auc), np.nanstd(roc_auc)))

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    try:
        y_scores = classifier.decision_function(X_test)
    except:
        y_scores = np.max(classifier.predict_proba(X_test), axis=1)

    val_accuracy, val_precision, val_recall, val_f1, val_roc_auc = performance_per_subject(subs_y_test, y_pred, y_scores,
                                                                                           list(X_test_df.index),
                                                                                           subjects_test)
    print('--- Testing performance ---')
    print('Accuracy: %.2f' % val_accuracy)
    print('Precision: %.2f' % val_precision)
    print('Recall: %.2f' % val_recall)
    print('F1 score: %.2f' % val_f1)
    print('ROC AUC: %.2f' % val_roc_auc)

    if not args.nested_cv and SELECT_FEATURES and N_FEATURES != 'all':
        plot_results(classifier, X_train, X_test, y_train, y_test, feature_names, importances, coefs, parc=args.parc,
                     sites=args.sites)


if __name__ == "__main__":
    main()
