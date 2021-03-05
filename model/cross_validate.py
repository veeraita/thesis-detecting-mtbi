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
    """Obtain performance estimates (for a single CV fold)"""
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
    """A subject is predicted positive if at least half of the samples from the subject are predicted positive"""
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
    """
    Return cross-validation splits so that all samples from a single subject are either in the train set
    or the test set but never both (this function is currently only needed in the grid search)
    """
    inds = []
    subject_wise_inds = []
    for ii, (train, test) in enumerate(cv.split(subjects, y=subs_y)):
        train_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[train])
        test_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[test])
        inds.append((train_idx, test_idx))
        subject_wise_inds.append((train, test))
    return inds, subject_wise_inds


def run_grid(clf, selector, X, y, cv_inds, param_space):
    """Run grid search for the hyperparameters"""
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
def run_cv(clf, X_df, y, subs_y, selector='passthrough', average='binary', n_splits=7, n_perm_repeats=3, param_grid=None,
           nested=False, rep_cv=False, random_state=1):
    """Run the cross-validation"""
    sample_names = [s.split('_')[0] for s in X_df.index]
    subjects = np.asarray(list(dict.fromkeys(sample_names)))

    # data frame for storing the average accuracy and the decision function value per subject
    subject_results = pd.DataFrame(0, columns=['classif_score', 'df_score'], index=subjects, dtype=float)

    if rep_cv:
        # repeat the cross-validation with different splits each time
        cv_outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5, random_state=random_state)
    else:
        cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(f'Starting {cv_outer.get_n_splits()}-fold cross validation')
    print(clf)
    # dict for storing the performance measures
    results_dict = defaultdict(list)
    # initialize an empty array for storing feature importances
    importances = np.zeros((cv_outer.get_n_splits(), X_df.shape[1], n_perm_repeats))

    if param_grid is not None and not nested:
        # do the hyperparameter grid search (if not using nested CV)
        cv_inds, sub_inds = custom_cv(cv_outer, X_df, subjects, subs_y)
        if selector == 'passthrough':
            param_grid = [{k: v for k, v in params.items() if 'selector' not in k} for params in param_grid]
        X, feature_names_orig = features_from_df(X_df)
        best_clf, best_selector = run_grid(clf, selector, X, y, cv_inds, param_grid)
    else:
        best_selector = selector
        best_clf = clf

    # make a pipeline to be cross-validated, consisting of an optional feature selector and the classifier
    cachedir = mkdtemp()
    pipe = make_pipeline(best_selector, best_clf, memory=cachedir)

    # start the actual cross-validation
    # note that we are splitting the list of subjects, not samples, to avoid leaking information
    for ii, (train, test) in enumerate(cv_outer.split(subjects, y=subs_y)):
        print('Fold', ii + 1)
        # convert the 'subject indices' to 'sample indices'
        train_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[train])
        test_idx = X_df.index.str.split(pat='_').str[0].isin(subjects[test])
        X_df_train = X_df.iloc[train_idx]
        y_train = y[train_idx]
        X_df_test = X_df.iloc[test_idx]
        y_test = y[test_idx]

        # prepare the training and testing set
        X_train, feature_names_orig = features_from_df(X_df_train)
        X_test, _ = features_from_df(X_df_test)
        X_train, y_train, X_test, y_test = preprocess(X_train, y_train, X_test, y_test)

        X_train_orig, X_test_orig = X_train.copy(), X_test.copy()

        if nested:
            # do the grid search in an inner loop
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

            # calculate feature importances
            importances[ii, feature_inds] = feature_importance(pipe[1], X_test, y_test, n_repeats=n_perm_repeats,
                                                               random_state=random_state)

        # make the predictions and evaluate the model
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

        results_dict['accuracy'].append(cv_accuracy)
        results_dict['precision'].append(cv_precision)
        results_dict['recall'].append(cv_recall)
        results_dict['specif'].append(cv_specif)
        results_dict['f1'].append(cv_f1)
        results_dict['roc_auc'].append(cv_roc_auc)

    print(pipe)
    rmtree(cachedir)

    return pipe, results_dict, np.mean(importances, axis=0), subject_results
