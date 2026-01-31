import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes as pltax
import numpy as np
import anndata as ad

import warnings
from warnings import warn

from scipy.sparse import issparse
from scipy.stats.stats import pearsonr, spearmanr

def cal_var(adata, show=False, color=['b', 'r'], save=None):
    """
    Show distribution plots of cells sharing features and variability score.

    Parameters
    ----------

    adata

    show

    color

    save

    """

    if issparse(adata.X):
        adata.var['n_cells'] = adata.X.sum(axis=0).tolist()[0] ## double check this
        adata.var['prop_shared_cells'] = adata.var['n_cells']/len(adata.obs_names.tolist())
        adata.var['variability_score'] = [1-abs(n-0.5) for n in adata.var['prop_shared_cells']]
    else:
        adata.var['n_cells'] = adata.X.sum(axis=0)
        adata.var['prop_shared_cells'] = adata.var['n_cells']/len(adata.obs_names.tolist())
        adata.var['variability_score'] = [1-abs(n-0.5) for n in adata.var['prop_shared_cells']]

    if save!= None:
        fig, axs = plt.subplots(ncols=2)
        plt.subplots_adjust(wspace=0.7)

        ax0 = sns.distplot(adata.var['prop_shared_cells'], bins=40, ax=axs[0], color=color[0])
        ax0.set(xlabel='cells sharing a feature', ylabel='density')
        ax1 = sns.distplot(adata.var['variability_score'], bins=40, ax=axs[1], color=color[1])
        ax1.set(xlabel='variability score', ylabel='density')
        plt.savefig(save, bbox_inches="tight")
        show = False

    if show: # plotting
        fig, axs = plt.subplots(ncols=2)
        plt.subplots_adjust(wspace=0.7)

        ax0 = sns.distplot(adata.var['prop_shared_cells'], bins=40, ax=axs[0], color=color[0])
        ax0.set(xlabel='cells sharing a feature', ylabel='density')
        ax1 = sns.distplot(adata.var['variability_score'], bins=40, ax=axs[1], color=color[1])
        ax1.set(xlabel='variability score', ylabel='density')
        plt.show()


def select_var_feature(adata, min_score=0.5, nb_features=None, show=True, copy=False, save=None):
    """
    This function computes a variability score to rank the most variable features across all cells.
    Then it selects the most variable features according to either a specified number of features (nb_features) or a maximum variance score (max_score).

    Parameters
    ----------

    adata: adata object

    min_score: minimum threshold variability score to retain features,
    where 1 is the score of the most variable features and 0.5 is the score of the least variable features.

    nb_features: default value is None, if specify it will select a the top most variable features.
    if the nb_features is larger than the total number of feature, it filters based on the max_score argument

    show: default value True, it will plot the distribution of var.

    copy: return a new adata object if copy == True.

    Returns
    -------
    Depending on ``copy``, returns a new AnnData object or overwrite the input


    """
    if copy:
        inplace=False
    else:
        inplace=True

    adata = adata.copy() if not inplace else adata

    # calculate variability score
    cal_var(adata, show=show, save=save) # adds variability score for each feature
    # adata.var['variablility_score'] = abs(adata.var['prop_shared_cells']-0.5)
    var_annot = adata.var.sort_values(ascending=False, by ='variability_score')

    # calculate the max score to get a specific number of feature
    if nb_features != None and nb_features < len(adata.var_names):
            min_score = var_annot['variability_score'][nb_features]


    adata_tmp = adata[:,adata.var['variability_score']>=min_score].copy()

    ## return the filtered AnnData objet.
    if not inplace:
        adata_tmp = adata[:,adata.var['variability_score']>=min_score]
        return(adata_tmp)
    else:
        adata._inplace_subset_var(adata.var['variability_score']>=min_score)


def binarize(adata, copy=False):
    """convert the count matrix into a binary matrix.

    Parameters
    ----------

    adata: AnnData object

    copy: return a new adata object if copy == True.

    Returns
    -------
    Depending on ``copy``, returns a new AnnData object or overwrite the input

    """

    if copy:
        adata2 = adata.copy()
        adata2.X[adata2.X != 0] = 1
        return(adata2)
    else:
        adata.X[adata.X != 0] = 1