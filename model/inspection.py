import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import re
from mpl_toolkits.axisartist.parasite_axes import SubplotHost, HostAxes, ParasiteAxes
from sklearn.inspection import permutation_importance, plot_partial_dependence
from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from matplotlib import rc

plt.style.use('seaborn-colorblind')
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.size': 12})
red = '#e41a1c'
blue = '#377eb8'


def parse_label(label):
    insert_space_words = [
        'anterior',
        'caudal',
        'frontal',
        'inferior',
        'isthmus',
        'lateral',
        'medial',
        'middle',
        'pars',
        'posterior',
        'rostral',
        'superior',
        'temporal',
        'transverse'
    ]
    pattern = '(' + '|'.join(insert_space_words) + ')'
    sub = r'\1 '
    label = re.sub(pattern, sub, label)
    label = ' / '.join(label.split('-')).replace('_', ' ').replace('  ', ' ')
    return label


def feature_importance(clf, X, y, n_repeats=5, random_state=1):
    print('Calculating feature importances')
    result = permutation_importance(clf, X, y, scoring='accuracy', n_repeats=n_repeats, n_jobs=4,
                                    random_state=random_state)
    return result.importances


def save_feature_importance(importances, feature_names, fig_fpath, output_fpath):
    print('Visualizing feature importance')
    importances_mean = np.mean(importances, axis=-1)
    importances_std = np.std(importances, axis=-1)
    perm_sorted_idx = importances_mean.argsort()[::-1][:40]

    fig = plt.figure(figsize=(10, 8))
    medianprops = {'c': red}
    boxprops = {'facecolor': 'aliceblue'}
    labels = [parse_label(label) for label in feature_names[perm_sorted_idx][::-1]]
    plt.boxplot(importances[perm_sorted_idx][::-1].T, vert=False, labels=labels, medianprops=medianprops,
                boxprops=boxprops, patch_artist=True)
    fig.tight_layout()
    plt.savefig(fig_fpath)

    with open(output_fpath, 'w') as f:
        for i in importances_mean.argsort()[::-1]:
            f.write(f"{feature_names[i]}\t"
                  f"{importances_mean[i]:.3f}"
                  f" +/- {importances_std[i]:.3f}\n")


def plot_correlation(X, feature_names, fig_fpath):
    print('Visualizing correlation')
    fig = plt.figure(figsize=(10, 10))
    # host = HostAxes(fig)
    # ax2 = ParasiteAxes(host, sharey=host)
    # host.parasites.append(ax2)
    ax = SubplotHost(fig, 111)
    fig.add_subplot(ax)
    # fig, ax = plt.subplots(figsize=(20, 20))

    corr = spearmanr(X).correlation
    im = ax.imshow(corr, cmap='bwr')
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    #cb.ax.tick_params(labelsize=40)
    ax.set_xticks([])
    ax.set_yticks([])
    labels = [x.split('-')[0] for x in feature_names if 'paracentral_1-lh' in x]

    ax2 = ax.twiny()
    new_axisline = ax2.get_grid_helper().new_fixed_axis
    offset = 0, -25
    ax2.axis['bottom'] = new_axisline(loc='bottom', axes=ax2, offset=offset)
    ax2.axis['top'].set_visible(False)
    ax2.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0.1, 1.1, 0.2)))
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(labels))
    ax2.axis['bottom'].minor_ticks.set_ticksize(0)
    ax2.axis['bottom'].major_ticks.set_ticksize(10)
    ax2.xaxis.set_tick_params(which='major', length=20, width=2)

    ax3 = ax.twinx()
    new_axisline = ax3.get_grid_helper().new_fixed_axis
    offset = -25, 0
    ax3.axis['left'] = new_axisline(loc='left', axes=ax3, offset=offset)
    ax3.axis['right'].set_visible(False)
    ax3.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax3.yaxis.set_major_formatter(ticker.NullFormatter())
    ax3.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(0.1, 1.1, 0.2)))
    ax3.yaxis.set_minor_formatter(ticker.FixedFormatter(labels[::-1]))
    ax3.axis['left'].minor_ticks.set_ticksize(0)
    ax3.axis['left'].major_ticks.set_ticksize(10)
    ax3.yaxis.set_tick_params(which='major', length=20, width=2)

    plt.savefig(fig_fpath)
    return corr

def plot_linear_coefs(coefs, feature_names, fig_fpath):
    plt.tight_layout()
    coefs_df = pd.DataFrame(coefs, columns=['Coefficients'], index=feature_names)
    top = coefs_df.reindex(coefs_df.Coefficients.abs().sort_values(ascending=True).index).tail(40)
    ax = top.plot(kind='barh', figsize=(10, 12))
    fig = ax.get_figure()
    fig.savefig(fig_fpath, bbox_inches='tight')


def save_linear_coefs(coefs, feature_names, fpath):
    coefs_df = pd.DataFrame(coefs, columns=['Coefficients'], index=feature_names)
    coefs_df.to_csv(fpath)


def plot_class_separation(clf, X_train, X_test, y_train, y_test, fig_fpath):
    print('Visualizing class separation')
    X_train_distances = pd.DataFrame(clf.decision_function(X_train))
    X_test_distances = pd.DataFrame(clf.decision_function(X_test))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.distplot(X_train_distances[y_train == 1], color=red, kde=False, ax=axes[0], bins=30, norm_hist=True)
    sns.distplot(X_train_distances[y_train == 0], color=blue, kde=False, ax=axes[0], bins=30, norm_hist=True)
    sns.distplot(X_test_distances[y_test == 1], color=red, kde=False, ax=axes[1], bins=10, norm_hist=True)
    sns.distplot(X_test_distances[y_test == 0], color=blue, kde=False, ax=axes[1], bins=10, norm_hist=True)
    for ax, title in zip(axes, ('Training', 'Testing')):
        ax.legend(['Patient', 'Control'])
        ax.set_xlim((-2, 2))
        ax.axvline(x=1, linestyle='dashed', linewidth=0.8, c='black')
        ax.axvline(x=-1, linestyle='dashed', linewidth=0.8, c='black')
        ax.axvline(x=0, linestyle='dashed', linewidth=1, c='black')
        ax.set_title(f'{title} Separation From \nDecision Boundary')
    plt.savefig(fig_fpath)


def plot_pdp(clf, X, y, features, fig_fpath, figsize=(10, 10), n_cols=2, feature_names=None):
    print('Visualizing partial dependence')
    fig, ax = plt.subplots(figsize=figsize)
    clf.fit(X, y)
    disp = plot_partial_dependence(clf, X, features, feature_names=[parse_label(label) for label in feature_names],
                                   percentiles=(0.05, 0.95), kind='average')
    disp.plot(ax=ax, n_cols=n_cols, line_kw={'color': red})
    disp.figure_.subplots_adjust(wspace=0.4, hspace=0.4)
    fig.tight_layout()
    fig.savefig(fig_fpath)


def plot_pca_variance(X, fig_fpath):
    print('Visualizing PCA variance')
    pca = PCA(n_components=20)
    pca.fit(X)
    fig, ax = plt.subplots(figsize=(10, 8))
    print(np.cumsum(pca.explained_variance_ratio_))
    ax.plot(np.cumsum(pca.explained_variance_ratio_), color=red)
    ax.set_xticks(np.arange(0, 20, 2))
    ax.set_xlabel('Number of PCA components')
    ax.set_ylabel('Fraction of explained variance')
    fig.savefig(fig_fpath)


def save_pca_loadings(pca, X, y, feature_names, fig_fpath, loadings_fpath):
    print('Visualizing PCA loadings')
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_s = pd.DataFrame(loadings, index=feature_names)

    df = pd.DataFrame(loadings_s)
    df.to_csv(loadings_fpath, float_format='%.3f')

    X_df = pd.DataFrame(X, columns=[f'Component {i+1}' for i in range(X.shape[1])])
    X_df['Label'] = ['patient' if label == 1 else 'control' for label in y]

    ax = sns.pairplot(X_df, hue='Label', palette='Set1')
    plt.setp(ax.legend.get_texts(), fontsize='12')
    plt.setp(ax.legend.get_title(), fontsize='12')
    ax.savefig(fig_fpath)
    plt.close()


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
