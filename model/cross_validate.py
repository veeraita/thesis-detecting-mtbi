import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    classification_report
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from tempfile import mkdtemp
from shutil import rmtree
from collections import defaultdict

from preprocessing import features_from_df, preprocess
from inspection import feature_importance

pd.set_option('use_inf_as_na', True)


def performance(y_test, y_pred, y_scores, average='binary'):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1, average=average)
    recall = recall_score(y_test, y_pred, zero_division=1, average=average)
    specificity = recall_score(y_test, y_pred, zero_division=1, average=average, pos_label=0)
    f1 = f1_score(y_test, y_pred, zero_division=1, average=average)
    try:
        roc_auc = roc_auc_score(y_test, y_scores)
    except ValueError:
        roc_auc = np.nan

    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['negative', 'positive'],
                                zero_division=1))
    return accuracy, precision, recall, specificity, f1, roc_auc


def prediction_per_subject(y_test_subjects, y_pred, y_scores, sample_names, subjects):
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
        print(y_test_subjects[i], y_pred_subjects[i], '%.3f' % y_scores_subjects[i], subjects[i])

    return y_pred_subjects, y_scores_subjects


def custom_cv(cv, X_df, subjects, subs_y):
    inds = []
    subject_wise_inds = []
    for ii, (train, test) in enumerate(cv.split(subjects, y=subs_y)):
        train_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[train])
        test_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[test])
        inds.append((train_idx, test_idx))
        subject_wise_inds.append((train, test))
    return inds, subject_wise_inds


def run_grid(clf, selector, X, y, cv_inds, param_space):
    cachedir = mkdtemp()
    pipe = make_pipeline(selector, clf, memory=cachedir)
    grid = GridSearchCV(pipe, param_space, scoring='accuracy', cv=cv_inds, n_jobs=4, verbose=1)
    grid.fit(X, y)
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        if mean > 0.68:
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print('Best params:', grid.best_params_)
    print('Estimated score: %.3f\n' % grid.best_score_)
    best_clf = grid.best_estimator_[-1]
    best_selector = grid.best_estimator_[0]
    rmtree(cachedir)

    return best_clf, best_selector


@ignore_warnings(category=ConvergenceWarning)
def run_cv(clf, X_df, y, cases, selector='passthrough', average='binary', n_splits=7, n_perm_repeats=3, param_grid=None,
           nested=False, rep_cv=False, confound=False, random_state=1):
    sample_names = [s.split('_')[0] for s in X_df.index]
    subjects = list(dict.fromkeys(sample_names))
    subs_y = np.asarray([1 if s[:3] in cases else 0 for s in list(subjects)])
    subjects = np.asarray(subjects)

    subject_results = pd.DataFrame(0, columns=['classif_score', 'df_score'], index=subjects, dtype=float)

    if rep_cv:
        cv_outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5, random_state=random_state)
    else:
        cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print(f'Starting {cv_outer.get_n_splits()}-fold cross validation')
    print(clf)
    accuracy = []
    precision = []
    recall = []
    specif = []
    f1 = []
    roc_auc = []
    importances = np.zeros((cv_outer.get_n_splits(), X_df.shape[1], n_perm_repeats))

    if param_grid is not None and not nested:
        cv_inds, sub_inds = custom_cv(cv_outer, X_df, subjects, subs_y)
        if selector == 'passthrough':
            param_grid = [{k: v for k, v in params.items() if 'selector' not in k} for params in param_grid]
        X, confounds, feature_names_orig = features_from_df(X_df)
        best_clf, best_selector = run_grid(clf, selector, X, y, cv_inds, param_grid)
    else:
        best_selector = selector
        best_clf = clf

    cachedir = mkdtemp()
    pipe = make_pipeline(best_selector, best_clf, memory=cachedir)

    for ii, (train, test) in enumerate(cv_outer.split(subjects, y=subs_y)):
        print('Fold', ii + 1)
        train_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[train])
        test_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[test])
        X_df_train = X_df.iloc[train_idx]
        y_train = y[train_idx]
        X_df_test = X_df.iloc[test_idx]
        y_test = y[test_idx]
        X_train, confounds_train, feature_names_orig = features_from_df(X_df_train)
        X_test, confounds_test, _ = features_from_df(X_df_test)
        if confound:
            confounds = np.vstack((confounds_train, confounds_test))
        else:
            confounds = None
        X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test, confounds)
        X_train_orig, X_test_orig = X_train.copy(), X_test.copy()

        if nested:
            cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_inds, sub_inds = custom_cv(cv_inner, X_df_train, subjects[train], subs_y[train])
            if selector == 'passthrough':
                param_grid = [{k: v for k, v in params.items() if 'selector' not in k} for params in param_grid]
            best_clf, best_selector = run_grid(clf, selector, X_train, y_train, cv_inds, param_grid)
            pipe = make_pipeline(best_selector, best_clf)

        pipe.fit(X_train_orig, y_train)

        if selector != 'passthrough':
            feature_inds = pipe[0].get_support()

            X_train = pipe[0].transform(X_train_orig)
            X_test = pipe[0].transform(X_test_orig)

            importances[ii, feature_inds] = feature_importance(pipe[1], X_test, y_test, n_repeats=n_perm_repeats,
                                                               random_state=random_state)

        y_pred = pipe.predict(X_test_orig)
        try:
            y_scores = pipe.decision_function(X_test_orig)
        except:
            y_scores = np.max(pipe.predict_proba(X_test_orig), axis=1)

        y_pred_subjects, y_scores_subjects = prediction_per_subject(subs_y[test], y_pred, y_scores,
                                                                    list(X_df_test.index), subjects[test])
        cv_accuracy, cv_precision, cv_recall, cv_specif, cv_f1, cv_roc_auc = performance(subs_y[test], y_pred_subjects,
                                                                                         y_scores_subjects,
                                                                                         average=average)

        for i in range(len(subjects[test])):
            if rep_cv:
                div = 5
            else:
                div = 1
            if y_pred_subjects[i] == subs_y[test][i]:
                subject_results.at[subjects[test][i], 'classif_score'] += (1 / div)
            subject_results.at[subjects[test][i], 'df_score'] += (y_scores_subjects[i] / div)

        accuracy.append(cv_accuracy)
        precision.append(cv_precision)
        recall.append(cv_recall)
        specif.append(cv_specif)
        f1.append(cv_f1)
        roc_auc.append(cv_roc_auc)

    print(pipe)
    rmtree(cachedir)

    #results_df = pd.DataFrame.from_dict(subject_results, orient='index', columns=['score']).sort_index()

    return pipe, np.asarray(accuracy), np.asarray(precision), np.asarray(recall), np.asarray(specif), np.asarray(f1), \
           np.asarray(roc_auc), np.mean(importances, axis=0), subject_results


def run_cv_pca(clf, pca, X_df, y, cases, average='binary', n_splits=7, param_grid=None, rep_cv=False, confound=False,
               nested=False, random_state=1):
    sample_names = [s.split('_')[0] for s in X_df.index]
    subjects = list(dict.fromkeys(sample_names))
    subs_y = np.asarray([1 if s[:3] in cases else 0 for s in list(subjects)])
    subjects = np.asarray(subjects)

    subject_results = defaultdict(float)

    if rep_cv:
        cv_outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5, random_state=random_state)
    else:
        cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print(f'Starting {cv_outer.get_n_splits()}-fold cross validation')
    print(clf)
    accuracy = []
    precision = []
    recall = []
    specif = []
    f1 = []
    roc_auc = []

    if param_grid is not None and not nested:
        cv_inds, sub_inds = custom_cv(cv_outer, X_df, subjects, subs_y)
        X, confounds, feature_names_orig = features_from_df(X_df)
        best_clf, best_pca = run_grid(clf, pca, X, y, cv_inds, param_grid)
    else:
        best_clf = clf
        best_pca = pca

    pipe = make_pipeline(best_pca, best_clf)

    for ii, (train, test) in enumerate(cv_outer.split(subjects, y=subs_y)):
        print('Fold', ii + 1)
        train_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[train])
        test_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[test])
        X_df_train = X_df.iloc[train_idx]
        y_train = y[train_idx]
        X_df_test = X_df.iloc[test_idx]
        y_test = y[test_idx]
        X_train, confounds_train, feature_names_orig = features_from_df(X_df_train)
        X_test, confounds_test, _ = features_from_df(X_df_test)
        if confound:
            confounds = np.vstack((confounds_train, confounds_test))
        else:
            confounds = None
        X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test, confounds)

        if nested:
            cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_inds, sub_inds = custom_cv(cv_inner, X_df_train, subjects[train], subs_y[train])
            best_clf, best_pca = run_grid(clf, pca, X_train, y_train, cv_inds, param_grid)

            pipe = make_pipeline(best_pca, best_clf)

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_scores = pipe.decision_function(X_test)

        y_pred_subjects, y_scores_subjects = prediction_per_subject(subs_y[test], y_pred, y_scores,
                                                                    list(X_df_test.index), subjects[test])
        cv_accuracy, cv_precision, cv_recall, cv_specif, cv_f1, cv_roc_auc = performance(subs_y[test], y_pred_subjects,
                                                                              y_scores_subjects, average=average)

        # cv_accuracy, cv_precision, cv_recall, cv_f1, cv_roc_auc = performance(y_test, y_pred, y_scores, average=average)

        for i in range(len(subjects[test])):
            if y_pred_subjects[i] == subs_y[test][i]:
                if rep_cv:
                    subject_results[subjects[test][i]] += 0.2
                else:
                    subject_results[subjects[test][i]] += 1
            else:
                subject_results[subjects[test][i]] += 0

        accuracy.append(cv_accuracy)
        precision.append(cv_precision)
        recall.append(cv_recall)
        specif.append(cv_specif)
        f1.append(cv_f1)
        roc_auc.append(cv_roc_auc)

    print(pipe)

    results_df = pd.DataFrame.from_dict(subject_results, orient='index', columns=['score']).sort_index()

    return pipe, np.asarray(accuracy), np.asarray(precision), np.asarray(recall), np.asarray(specif), np.asarray(f1), \
           np.asarray(roc_auc), results_df
