"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

import matplotlib.pyplot as plt
import random
import numpy as np


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.itervalues():
        sp.set_visible(False)


def plot_embeddings_with_mapping(embeddings, clusters, cluster_to_label_map,
        filter_labels=[], filter_clusters=[], n_samples=None):
    """If `n_samples` is provided, only plot this many embeddings."""

    clusters = np.array(clusters)

    if filter_labels != []:
        # Filter to only include these labels
        new_clusters = []
        new_embeddings = []
        for i in range(len(clusters)):
            if clusters[i] in cluster_to_label_map and cluster_to_label_map[clusters[i]] in filter_labels:
                new_clusters.append(clusters[i])
                new_embeddings.append(embeddings[i, :])
        clusters = np.array(new_clusters)
        embeddings = np.array(new_embeddings)

    if filter_clusters != []:
        # Filter to only include these clusters
        new_clusters = []
        new_embeddings = []
        for i in range(len(clusters)):
            if clusters[i] in filter_clusters:
                new_clusters.append(clusters[i])
                new_embeddings.append(embeddings[i, :])
        clusters = np.array(new_clusters)
        embeddings = np.array(new_embeddings)

    if n_samples is not None:
        indices = np.array(random.sample(range(len(clusters)), n_samples))
        embeddings = embeddings[indices, :]
        clusters = clusters[indices]
    n_embeds = embeddings.shape[0]

    # Get indices order
    mapped_labels = np.array(
        [cluster_to_label_map[i] if i in cluster_to_label_map else "unk" for i in clusters]
        )
    dtype = [("cluster", int), ("mapped_label", "S1")]
    values = zip(clusters, mapped_labels)
    sorter = np.array(values, dtype=dtype)
    indices_order = np.argsort(sorter, order = ["cluster", "mapped_label"])

    # Sort data
    clusters_sorted = clusters[indices_order]
    embeddings_sorted = embeddings[indices_order, :]
    mapped_labels_sorted = mapped_labels[indices_order]

    # Get cluster tick positions
    cluster_ticks = [0]
    for i in range(len(clusters_sorted) - 1):
        if clusters_sorted[i] != clusters_sorted[i + 1]:
            cluster_ticks.append(i + 1)
    cluster_ticks.append(n_embeds)

    # Get mapping positions and labels
    mapping_ticks = []
    mapping_labels = []
    for i in cluster_to_label_map:
        where = np.where(clusters_sorted == i)[0]
        if len(where) == 0:
            continue
        pos =  int(np.mean(where))
        mapping_ticks.append(pos)
        label = cluster_to_label_map[i]
        mapping_labels.append(r"" + str(i) + r" $\rightarrow$ " + label + "")

    # Plot the embeddings
    fig, host = plt.subplots()
    labels_offset = 1.01
    par2 = host.twinx()
    par2.spines["right"].set_position(("axes", labels_offset))
    make_patch_spines_invisible(par2)
    par2.spines["right"].set_visible(True)
    par2_linewidth = 0.5
    par2.invert_yaxis()
    par2.set_yticks(cluster_ticks)
    par2.set_yticklabels([])
    par2.tick_params(axis="y", width=par2_linewidth, length=10)
    par2.spines["right"].set_linewidth(par2_linewidth)
    par2.set_yticks(mapping_ticks, minor=True)
    par2.set_yticklabels(mapping_labels, minor=True)
    par2.set_ylabel("Cluster and mapping")
    for line in par2.yaxis.get_minorticklines():
            line.set_visible(False)
    host.imshow(embeddings_sorted, interpolation="nearest", aspect="auto")
    host.set_yticks([])
    host.set_xticklabels([])
    host.set_ylabel("Word embedding vector")
    host.set_xlabel("Embedding dimensions")

# plot_embeddings_with_mapping(
#     embeddings, clusters, cluster_to_label_map, labels=["the", "and", "yknow",
#     "because", "different", "they're"], n_samples=None
#     )
# plt.show()
