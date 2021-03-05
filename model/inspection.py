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

# styling
plt.style.use('seaborn-colorblind')
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.size': 12})
red = '#e41a1c'
blue = '#377eb8'


def parse_label(label):
    """Make the feature names look more presentable"""
    insert_space_words = [ # words to insert a space after
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

    # save importances also in text format
    with open(output_fpath, 'w') as f:
        for i in importances_mean.argsort()[::-1]:
            f.write(f"{feature_names[i]}\t"
                  f"{importances_mean[i]:.3f}"
                  f" +/- {importances_std[i]:.3f}\n")


def plot_correlation(X, feature_names, fig_fpath):
    print('Visualizing correlation')
    fig = plt.figure(figsize=(10, 10))
    ax = SubplotHost(fig, 111)
    fig.add_subplot(ax)

    corr = spearmanr(X).correlation
    im = ax.imshow(corr, cmap='bwr')
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
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
