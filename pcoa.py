#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#
# Author:   HIGASHI Koichi
# Created:  2014-11-19
#

import argparse
import pandas as pd
import numpy as np
import distance
from sklearn import manifold
import scipy.stats
import matplotlib.pyplot as plt
import itertools

def executePCoA(data, distance_metric, drawBiplot, n_arrows, groupfile):
    
    matrix = data.values
    n_features, n_samples = matrix.shape
    print(n_features,'features, ',n_samples,'samples')
    
    # compute distance
    if distance_metric == 'Jaccard':
        distance_matrix = distance.Jaccard(matrix.T)
    elif distance_metric == 'BrayCurtis':
        distance_matrix = distance.BrayCurtis(matrix.T)
    elif distance_metric == 'JSD':
        distance_matrix = distance.JSDivergence(matrix.T)
    
    # execute PCoA
    mds = manifold.MDS(n_components=2, max_iter=3000, dissimilarity="precomputed", n_jobs=1)
    positions = mds.fit(distance_matrix).embedding_
    positions_with_sampleIndex = pd.DataFrame(positions, index=data.columns)
    
    # General settings of the canvas
    fig = plt.figure(figsize=(12,12))
    ax = fig.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['left'].set_position(('data',0))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    
    if drawBiplot:
        circle = plt.Circle((0,0), radius=1.0, fc='none', linestyle='dashed', color='gray')
        ax.add_patch(circle)
        # compute correlations between feature vectors and data points.
        cor_pc1 = np.array([0.]*n_features)
        cor_pc2 = np.array([0.]*n_features)
        arrow_length = np.array([0.]*n_features)
        for i,current_feature in enumerate(data.index):
            if np.ptp(data.loc[current_feature].values) == 0:
                x = 0
                y = 0
            else:
                x = scipy.stats.pearsonr( data.loc[current_feature].values, positions[:,0] )[0]
                y = scipy.stats.pearsonr( data.loc[current_feature].values, positions[:,1] )[0]
            cor_pc1[i] = x
            cor_pc2[i] = y
            arrow_length[i] = np.sqrt( x**2 + y**2 )
        arrows = pd.DataFrame( np.hstack(( np.matrix(cor_pc1).T, np.matrix(cor_pc2).T, np.matrix(arrow_length).T )), index=data.index, columns=['x','y','len'])
        sorted_arrows = arrows.sort_values(by=['len'],ascending=False)
        # Top-{n_arrows} contributing features are drawed
        for name in sorted_arrows.index[:n_arrows]:
            ax.arrow(0.0,0.0, arrows.loc[name,'x'], arrows.loc[name,'y'], ec='k', alpha=0.2)
            ax.annotate(name, xy=(arrows.loc[name,'x'],arrows.loc[name,'y']), xytext=(0,0), textcoords='offset points', color='k', fontsize=10)
    
    # draw plots using colors if samples are binned into groups
    if groupfile:
        group_names = []
        group2sample = {}
        df = pd.read_table(groupfile, header=None, index_col=0)
        for sample in df.index:
            # Use the value of second column (1) as a grouping category.
            group = df.loc[sample, 1]
            if group in group2sample:
                group2sample[group].append(sample)
            else:
                group2sample[group] = [sample]
                group_names.append(group)
        colors = itertools.cycle(['r','g','b','c','m','y','k'])
        markers = itertools.cycle(['o','^','s','*','x'])
        for i,current_group in enumerate(group_names):
            if len(group2sample[current_group]) == 0:
                continue
            ax.scatter(positions_with_sampleIndex.loc[group2sample[current_group],0],
                       positions_with_sampleIndex.loc[group2sample[current_group],1],
                       s=100, marker=next(markers), color=next(colors), label='%s'%current_group)
        plt.legend(bbox_to_anchor=(0., 1.01, 1., 1.01), loc=3, ncol=6, mode="expand", borderaxespad=0.)
    else:
        for i,sample_name in enumerate(data.columns):
            ax.annotate(sample_name, xy=(positions[i,0],positions[i,1]), xytext=(5,5), textcoords='offset points', color='k', fontsize=16)
        ax.scatter(positions[:,0], positions[:,1], c='k', s=50)
    
    x_label = 'PCo1'
    y_label = 'PCo2'
    ax.annotate(x_label, xy=(0.0, -1.0), xytext=(0.0,-40.0), textcoords='offset points', ha='center', color='k', fontsize=18)
    ax.annotate(y_label, xy=(-1.0, 0.0), xytext=(-40.0,0.0), textcoords='offset points', ha='center', color='k', fontsize=18, rotation=90)
    fig.savefig('result.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "-f", "--file", action="store", dest="data_file", help="tab-separated text file. rows are variables, columns are samples.")
    parser.add_argument( "-d", "--distance_metric", action="store", dest="dist", choices=['Jaccard', 'BrayCurtis', 'JSD'], help="choose distance metric used for PCoA.")
    parser.add_argument( "-b", "--biplot", action="store_true", dest="biplot", default=False, help="output biplot (with calculating factor loadings).")
    parser.add_argument( "-n", "--number_of_arrows", action="store", type=int, dest="n_arrows", default=0, help="how many top-contributing arrows should be drawed.")
    parser.add_argument( "-g", "--grouping_file", action="store", dest="group_file", default=None, help="plot samples by same colors and markers when they belong to the same group. Please indicate Tab-separated 'Samples vs. Group file' ( first columns are sample names, second columns are group names ).")
    args = parser.parse_args()
    
    if args.data_file == None:
        print("ERROR: requires options")
        parser.print_help()
        quit()
    
    datafile = args.data_file
    distance_metric = args.dist
    drawBiplot = args.biplot
    n_arrows = args.n_arrows
    groupfile = args.group_file
    
    data = pd.read_table(datafile,index_col=0)
    executePCoA(data, distance_metric, drawBiplot, n_arrows, groupfile)
    print('done.')
