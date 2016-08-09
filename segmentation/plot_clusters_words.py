#!/usr/bin/env python

"""
Visualize the biggest clusters for a given segmentation file.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from datetime import datetime
from os import path
import argparse
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from analyze_top_words import function_words
from segment_eval import analyze_segmentation, landmarks_to_intervals, intervals_to_max_overlap
from utils import cluster_analysis, cluster_model_present

min_count = 1
n_biggest = 30
cmap = "Blues"


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("segment_fn", type=str, help="pickled segmentation file")
    parser.add_argument("--speaker", type=str, help="target speaker")
    parser.add_argument("--clusters", type=str, help="comma separated cluster IDs")
    # parser.add_argument(
    #     "--non_func", action="store_true", help="show analysis only for non-function words"
    #     )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print datetime.now()

    if args.clusters is not None:
        filter_clusters = [int(i) for i in args.clusters.split(",")]

    # Read segmentation
    print "Reading:", args.segment_fn
    with open(args.segment_fn, "rb") as f:
        unsup_landmarks = pickle.load(f)
        unsup_transcript = pickle.load(f)

    d = path.split(path.split(args.segment_fn)[0])[0]
    data_dir = path.join("data", path.normpath(d).replace("models" + os.sep, ""))

    if args.speaker is not None and path.split(data_dir)[-1] != args.speaker:
        # Filter datasets according to speaker
        new_unsup_landmarks = {}
        new_unsup_transcript = {}
        for utt_label in unsup_landmarks:
            if utt_label.startswith(args.speaker):
                new_unsup_landmarks[utt_label] = unsup_landmarks[utt_label]
                new_unsup_transcript[utt_label] = unsup_transcript[utt_label]
        unsup_landmarks = new_unsup_landmarks
        unsup_transcript = new_unsup_transcript
        data_dir = path.join(data_dir, args.speaker)

    word_gt_fn = path.join(data_dir, "word_gt.pkl")
    print "Reading:", word_gt_fn
    with open(word_gt_fn, "rb") as f:
        gt_bounds = pickle.load(f)
        gt_labels = pickle.load(f)
    interval_mapping_func = intervals_to_max_overlap

    # # Perform frame-overlap analysis
    # print "Calculating frame overlap"
    # gt_intervals = {}
    # for utt in gt_bounds:
    #     gt_intervals[utt] = landmarks_to_intervals(gt_bounds[utt])
    # frame_labels_true = []
    # frame_labels_pred = []
    # for utt in gt_intervals:
    #     cur_frame_labels_true = []
    #     for i, (start, end) in enumerate(gt_intervals[utt]):
    #         cur_frame_labels_true.extend([gt_labels[utt][i] for j in range(start, end)])
    #     cur_frame_labels_pred = []
    #     for i, (start, end) in enumerate(unsup_landmarks[utt]):
    #         cur_frame_labels_pred.extend([unsup_transcript[utt][i] for j in range(start, end)])
    #     frame_labels_true.extend(cur_frame_labels_true)
    #     frame_labels_pred.extend(cur_frame_labels_pred)
    # frame_clusters = cluster_analysis.analyse_clusters(frame_labels_true, frame_labels_pred)

    # # Temp
    # cluster_to_label_map = {}
    # cluster_to_label_map[147] = 'k uw l'
    # cluster_to_label_map[878] = 'd ey k'
    # biggest_clusters = [147, 878]

    # Perform standard phone-level analysis
    analysis = analyze_segmentation(
        unsup_landmarks, unsup_transcript, gt_bounds, gt_labels, interval_mapping_func
        )
    cluster_to_label_map = analysis["cluster_to_label_map"]
    cluster_to_label_map_many = analysis["cluster_to_label_map_many"]
    labels_true = analysis["labels_true"]
    labels_pred = analysis["labels_pred"]
    clusters = cluster_analysis.analyse_clusters(labels_true, labels_pred)
    sizes, _ = cluster_model_present.get_sizes_purities(clusters)
    biggest_clusters = list(np.argsort(sizes))
    biggest_clusters.reverse()

    overlap_dict = {}
    show_clusters_list = []
    show_clusters_sizes = []
    for i_cluster in biggest_clusters:

        if args.clusters is not None and i_cluster not in filter_clusters:
            continue

        cluster = clusters[i_cluster]
        counts = cluster["counts"]

        # # if args.non_func and (i_cluster not in cluster_to_label_map_many or
        # #         cluster_to_label_map_many[i_cluster] in function_words):
        # #     continue

        # # Sorted frame-overlap: first the mapped phones and then the others in descending order
        # counts = frame_clusters[i_cluster]["counts"]
        # mapping = cluster_to_label_map_many[i_cluster].split(" ")
        # sorted_counts = []
        # # print i_cluster
        # # print counts
        # # print mapping
        # # for phone in mapping:
        # #     if phone in counts:
        # #         sorted_counts.append((phone, counts[phone]))
        # #         counts.pop(phone)
        sorted_counts = sorted(counts.items(), key=lambda x:x[1], reverse=True)
        sorted_counts = [(i[0], i[1]) for i in sorted_counts if i[1] > min_count]  # remove counts that occur once
        overlap_dict[i_cluster] = sorted_counts
        show_clusters_sizes.append(sum([i[1] for i in sorted_counts]))

        show_clusters_list.append(i_cluster)
        if len(show_clusters_list) == n_biggest:
            break

    # Sort according to shown sizes
    show_clusters_list = [x for (y, x) in sorted(zip(show_clusters_sizes, show_clusters_list))]
    show_clusters_list.reverse()

    mapping_array = np.zeros((
        len(overlap_dict), max([len(overlap_dict[i]) for i in sorted(overlap_dict)])
        ))
    for i, i_cluster in enumerate(show_clusters_list):
        overlaps = [j[1] for j in overlap_dict[i_cluster]]
        mapping_array[i, :len(overlaps)] = overlaps

    print datetime.now()

    fig, ax = plt.subplots()
    heatmap = ax.imshow(mapping_array, cmap=cmap, interpolation="nearest", aspect="auto")
    plt.yticks(range(len(show_clusters_list)), show_clusters_list)
    plt.xticks([])
    for y, i_cluster in enumerate(show_clusters_list):
        for x, (phone, count) in enumerate(overlap_dict[i_cluster]):
            plt.text(x, y, phone, horizontalalignment="center", verticalalignment="center")
    plt.colorbar(heatmap)

    plt.show()


if __name__ == "__main__":
    main()
