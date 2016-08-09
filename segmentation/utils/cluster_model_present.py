"""
Functions for presenting analysis on clustering from a mixture model.

In all the functions the `cluster` parameter should be in the format returned
by `cluster_analysis.analyse_clusters`.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2014
"""

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

import cluster_analysis


def get_sizes_purities(clusters):
    """Return two lists containing the sizes and purities of `clusters`."""
    sizes = []
    purities = []
    for cluster in clusters:
        sizes.append(cluster["size"])
        purities.append(cluster["purity"])
    return sizes, purities


def print_biggest_clusters(clusters, n_biggest=0):
    """Print the `n_biggest` clusters in `clusters`."""
    sizes, tmp = get_sizes_purities(clusters)
    indices = list(np.argsort(sizes)[-n_biggest:])  # http://stackoverflow.com/questions/16878715/how-to-find-the-index-of-n-largest-elements-in-a-list-or-np-array-python
    indices.reverse()
    print "-"*79
    for i_cluster in indices:
        cluster = clusters[i_cluster]
        print "Cluster", i_cluster
        for label in cluster["counts"]:
            print '"' + label + '":', cluster["counts"][label]
        print "Size:", cluster["size"]
        print "Purity:", cluster["purity"]
        print "-"*79


def print_cluster_stats(clusters):
    """Print overall statistics of `clusters`."""
    sizes, tmp = get_sizes_purities(clusters)
    print "Number of tokens:", np.sum(sizes)
    print "Mean cluster size:", np.mean(sizes)
    print "Std. of cluster size:", np.std(sizes)
    print "Number of clusters:", len(clusters)


def print_clustering_scores(labels_true, labels_pred):
    """Print scores achieved by the clustering."""
    print "Purity:", cluster_analysis.purity(labels_true, labels_pred)
    h, c, V = metrics.homogeneity_completeness_v_measure(
        labels_true, labels_pred
        )
    print "V-measure:", V
    print "ARI:", metrics.adjusted_rand_score(labels_true, labels_pred)
    print "1-to-1 accuracy:", cluster_analysis.one_to_one_accuracy(
        labels_true, labels_pred
        )


def print_model_stats(record):
    """Print statistics of the Gibbs sampling process."""
    print "Final log marginal prob:", record["log_marg"][-1]
    print "Total time:", sum(record["sample_time"])/60.0, "min"


def plot_log_marg_components(record):
    """
    Print iterations vs. log marginal prob and iterations vs. no. of
    components.
    """
    n_iter = len(record["sample_time"])
    
    plt.figure()
    
    plt.subplot(311)
    plt.subplot(311)
    plt.plot(range(n_iter), record["log_marg"], label="P(X, z)")
    if "log_prob_z" in record:
        plt.plot(range(n_iter), record["log_prob_z"], label="P(z)")
    if "log_prob_X_given_z" in record:
        plt.plot(range(n_iter), record["log_prob_X_given_z"], label="P(X|z)")
        plt.legend(loc = "lower center")
    plt.xlabel("Iteration")
    plt.ylabel("Log prob")

    plt.subplot(312)
    plt.plot(range(n_iter), record["log_marg*length"], label="P(X, z)*length")
    plt.xlabel("Iteration")
    plt.ylabel("Log prob")
    plt.legend(loc = "lower center")

    plt.subplot(313)
    plt.plot(range(n_iter), record["components"])
    plt.xlabel("Iteration")
    plt.ylabel("Components")


def scatter_tokens_vs_clusters_for_type(clusters):
    """Scatter plot no. of tokens vs. no. of clusters for every word type."""

    # Dict for no. tokens and no. clusters for each type
    tokens_and_clusters_for_type = {}  # key: type, entry: [tokens, clusters]
    for cluster in clusters:
        for label in cluster["counts"]:
            if label not in tokens_and_clusters_for_type:
                tokens_and_clusters_for_type[label] = [0, 0]
            tokens_and_clusters_for_type[label][0] += cluster["counts"][label]
            tokens_and_clusters_for_type[label][1] += 1

    tokens_for_type = [i[0] for i in tokens_and_clusters_for_type.values()]
    clusters_for_type = [i[1] for i in tokens_and_clusters_for_type.values()]

    plt.figure()
    plt.scatter(tokens_for_type, clusters_for_type)
    plt.xlabel("No of tokens for type")
    plt.ylabel("No of clusters for type")
