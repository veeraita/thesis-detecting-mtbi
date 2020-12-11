import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr


def feature_importance(clf, X, y, n_repeats=5, random_state=1):
    print('Calculating feature importances')
    result = permutation_importance(clf, X, y, scoring='f1', n_repeats=n_repeats, n_jobs=4,
                                    random_state=random_state)
    return result.importances


def save_feature_importance(importances, feature_names, fig_fpath, output_fpath):
    importances_mean = np.mean(importances, axis=-1)
    importances_std = np.std(importances, axis=-1)
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
    im = ax.imshow(corr, cmap='bwr')
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


def save_linear_coefs(coefs, feature_names, fpath):
    coefs_df = pd.DataFrame(coefs, columns=['Coefficients'], index=feature_names)
    coefs_df.to_csv(fpath)


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


def plot_pca_variance(pca, fig_fpath):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.set_xlabel('Number of PCA components')
    ax.set_ylabel('Fraction of explained variance')
    fig.savefig(fig_fpath)


def plot_pca_loadings(pca, X, y, feature_names, fig_fpath):
    n_loadings = 5
    transformed_data = pca.fit_transform(X)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_) * 100
    loadings_s = pd.DataFrame(loadings, index=feature_names)

    c1_loadings = loadings_s.reindex(loadings_s.iloc[:, 0].abs().sort_values(ascending=False).index).head(n_loadings)
    c2_loadings = loadings_s.reindex(loadings_s.iloc[:, 1].abs().sort_values(ascending=False).index).head(n_loadings)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(transformed_data[y == 0, 0], transformed_data[y == 0, 1], c='skyblue', marker='x', label='negative')
    ax.scatter(transformed_data[y == 1, 0], transformed_data[y == 1, 1], c='coral', marker='x', label='positive')

    for i in range(n_loadings):
        ax.plot([0, c1_loadings.iloc[i, 0]], [0, c1_loadings.iloc[i, 1]],
                color='k', linestyle='-', linewidth=2)
        ax.plot([0, c2_loadings.iloc[i, 0]], [0, c2_loadings.iloc[i, 1]],
                color='k', linestyle='-', linewidth=2)

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.legend()
    fig.savefig(fig_fpath)


def plot_learning_curves(train_losses, val_losses, f1_scores=None, accuracies=None, fig_fpath=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.plot(np.transpose(train_losses), color='orange', alpha=0.4)
    ax.plot(np.mean(train_losses, axis=0), color='orange', linewidth=4, label='Train loss')
    ax.plot(np.transpose(val_losses), color='red', alpha=0.4)
    ax.plot(np.mean(val_losses, axis=0), color='red', linewidth=4, label='Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    if f1_scores is not None:
        ax = axes[1]
        ax.plot(np.transpose(f1_scores), color='blue', alpha=0.5)
        ax.plot(np.mean(f1_scores, axis=0), color='blue', linewidth=4, label='F1 score')
        if accuracies is not None:
            ax.plot(np.transpose(accuracies), color='green', alpha=0.5)
            ax.plot(np.mean(accuracies, axis=0), color='green', linewidth=4, label='Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Score')
        ax.legend()

    plt.savefig(fig_fpath)


def plot_loss_dist(losses, fig_fpath, threshold=None):
    plt.figure()
    # sns.displot(loss_dist, bins=100, kde=True, color='blue', height=5, aspect=2)
    plt.hist(losses, bins=100)
    if threshold is not None:
        plt.axvline(threshold, 0.0, 10, color='r')
    plt.title('Loss distribution')
    plt.savefig('fig/loss_dist.png')
