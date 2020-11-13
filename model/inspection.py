import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest,  mutual_info_classif
from collections import defaultdict


def feature_selection_by_clustering(X, thresh=1):
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
    selector = SelectKBest(mi_score, k=k)
    selector.fit(X, y)
    selected_feature_inds = selector.get_support(indices=True)
    return np.asarray(selected_feature_inds)


def feature_importance(clf, X, y, n_repeats=5, random_state=1):
    print('Calculating feature importances')
    result = permutation_importance(clf, X, y, scoring='f1', n_repeats=n_repeats, n_jobs=4,
                                    random_state=random_state)
    return result.importances


def save_feature_importance(importances, feature_names, fig_fpath, output_fpath):
    importances_mean = np.mean(importances, axis=1)
    importances_std = np.std(importances, axis=1)
    perm_sorted_idx = importances_mean.argsort()[::-1][:50]

    fig = plt.figure(figsize=(12, 8))
    plt.boxplot(importances[perm_sorted_idx][::-1].T, vert=False, labels=feature_names[perm_sorted_idx][::-1])
    fig.tight_layout()
    plt.savefig(fig_fpath)

    with open(output_fpath, 'w') as f:
        for i in perm_sorted_idx:
            if importances_mean[i] - 2 * importances_std[i] > 0:
                f.write(f"{feature_names[i]}\t"
                      f"{importances_mean[i]:.3f}"
                      f" +/- {importances_std[i]:.3f}\n")


def plot_correlation(X, feature_names, fig_fpath):
    fig, ax = plt.subplots(figsize=(50, 50))
    corr = spearmanr(X).correlation
    im = ax.imshow(corr, cmap='RdBu')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(0, len(feature_names)))
    ax.set_yticks(np.arange(0, len(feature_names)))
    ax.set_xticklabels(feature_names, rotation='vertical')
    ax.set_yticklabels(feature_names)
    plt.savefig(fig_fpath)
    return corr

def plot_linear_coefs(coefs, feature_names, fig_fpath):
    plt.tight_layout()
    coefs_df = pd.DataFrame(coefs, columns=['Coefficients'], index=feature_names)
    top = coefs_df.reindex(coefs_df.Coefficients.abs().sort_values(ascending=True).index).tail(50)
    ax = top.plot(kind='barh', figsize=(12, 15))
    fig = ax.get_figure()
    fig.savefig(fig_fpath, bbox_inches='tight')


def plot_class_separation(clf, X_train, X_test, y_train, y_test, fig_fpath):
    X_train_distances = pd.DataFrame(clf.decision_function(X_train))
    X_test_distances = pd.DataFrame(clf.decision_function(X_test))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.distplot(X_train_distances[y_train == 1], color='r', kde=False, ax=axes[0], norm_hist=True)
    sns.distplot(X_train_distances[y_train == 0], color='b', kde=False, ax=axes[0], norm_hist=True)
    sns.distplot(X_test_distances[y_test == 1], color='r', kde=False, ax=axes[1], norm_hist=True)
    sns.distplot(X_test_distances[y_test == 0], color='b', kde=False, ax=axes[1], norm_hist=True)
    for ax, title in zip(axes, ('Training', 'Testing')):
        ax.legend(['Positive', 'Negative'])
        ax.set_xlim((-2, 2))
        ax.axvline(x=1, linestyle='dashed', linewidth=0.8, c='black')
        ax.axvline(x=-1, linestyle='dashed', linewidth=0.8, c='black')
        ax.axvline(x=0, linestyle='dashed', linewidth=1, c='black')
        ax.set_title(f'Histogram of Projections \n {title} Separation From Decision Boundary')
    plt.savefig(fig_fpath)


def plot_learning_curves(train_losses, val_losses, f1_scores, fig_fpath):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.plot(np.transpose(train_losses), color='orange', alpha=0.4)
    ax.plot(np.mean(train_losses, axis=0), color='orange', linewidth=4, label='Train loss')
    ax.plot(np.transpose(val_losses), color='red', alpha=0.4)
    ax.plot(np.mean(val_losses, axis=0), color='red', linewidth=4, label='Validation loss')
    #ax.plot(train_loss, color='orange', label='training loss')
    #ax.plot(val_loss, color='red', label='validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    ax = axes[1]
    ax.plot(np.transpose(f1_scores), color='blue', alpha=0.5)
    ax.plot(np.mean(f1_scores, axis=0), color='blue', linewidth=4, label='F1 score')
    #ax.plot(f1, color='green', label='F1 score')
    #ax.plot(accuracy, color='blue', label='accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('F1')
    ax.legend()
    plt.savefig(fig_fpath)
