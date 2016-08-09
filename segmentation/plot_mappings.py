#!/usr/bin/env python

"""
Print some mappings for evaluation for a particular speaker.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

from os import path
# from scipy.io.wavfile import read
import argparse
import cPickle as pickle
import matplotlib.pyplot as plt
import os
# import subprocess
import random
import sys
# import uuid

from segment_eval import analyze_segmentation, landmarks_to_intervals, intervals_to_max_overlap
from utils import cluster_analysis, cluster_model_present

# eval_utt = "s3801b_034077-034440"
# eval_utt = "s3802b_034819-034930"
# eval_utt = "s3801b_014360-014414"
# cluster = 1132
# buckeye_dir = "/endgame/projects/phd/datasets/buckeye"

# shell = lambda command: subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()[0]


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("segment_fn", type=str, help="pickled segmentation file")
    parser.add_argument("--n_tokens_min", type=int, help="minimum number of tokens")
    parser.add_argument("--target_word", type=str, help="target_word")
    parser.add_argument("--max_length", type=int, help="plots are all given with this maximum length")
    parser.add_argument("--n_samples", type=int, help="sample this many utterances")
    parser.add_argument("--many_map", action="store_true")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Read segmentation
    print "Reading:", args.segment_fn
    with open(args.segment_fn, "rb") as f:
        unsup_landmarks = pickle.load(f)
        unsup_transcript = pickle.load(f)

    d = path.split(path.split(args.segment_fn)[0])[0]
    data_dir = path.join("data", path.normpath(d).replace("models" + os.sep, ""))

    # Read ground truth phoneme transcriptions
    gt_phone_fn = path.join(data_dir, "phone_gt.pkl")
    print "Reading:", gt_phone_fn
    with open(gt_phone_fn, "rb") as f:
        gt_phone_bounds = pickle.load(f)
        gt_phone_labels = pickle.load(f)

    # Read ground truth word transcriptions
    gt_word_fn = path.join(data_dir, "word_gt.pkl")
    print "Reading:", gt_word_fn
    with open(gt_word_fn, "rb") as f:
        gt_word_bounds = pickle.load(f)
        gt_word_labels = pickle.load(f)

    # Get mapping
    interval_mapping_func = intervals_to_max_overlap
    analysis = analyze_segmentation(
        unsup_landmarks, unsup_transcript, gt_word_bounds, gt_word_labels, interval_mapping_func
        )
    cluster_to_label_map = analysis["cluster_to_label_map"]
    cluster_to_label_map_many = analysis["cluster_to_label_map_many"]
    labels_true = analysis["labels_true"]
    labels_pred = analysis["labels_pred"]
    clusters = cluster_analysis.analyse_clusters(labels_true, labels_pred)

    # Setup plotting
    plot_dir = path.join("plots", "mappings", path.normpath(d).replace("models" + os.sep, ""))
    if not path.isdir(plot_dir):
        os.makedirs(plot_dir)
    plt.rcParams["figure.figsize"]          = 10, 6
    plt.rcParams["text.usetex"]             = True
    plt.rcParams["font.size"]               = 12
    plt.rcParams["font.sans-serif"]         = "Computer Modern Sans serif"
    plt.rcParams["font.serif"]              = "Computer Modern Roman"

    # Print random mappings
    random.seed(1)
    if not args.n_samples is None:
        sample_keys = random.sample(unsup_transcript.keys(), args.n_samples)
    else:
        sample_keys = unsup_transcript.keys()
    n_tokens_min = 4
    for eval_utt in sample_keys:
        if not args.n_tokens_min is None and len(gt_word_labels[eval_utt]) < n_tokens_min:
            continue

        # Get data
        gt_marks = gt_word_bounds[eval_utt]
        unsup_marks = [i[1] for i in unsup_landmarks[eval_utt]]
        gt_labels = gt_word_labels[eval_utt]
        if args.many_map:
            unsup_labels_many = [cluster_to_label_map_many[i] if i in cluster_to_label_map_many else "[unk]" for i in unsup_transcript[eval_utt]]
            unsup_labels = unsup_labels_many
        else:
            unsup_labels = [cluster_to_label_map[i] if i in cluster_to_label_map else "[unk]" for i in unsup_transcript[eval_utt]]
        unsup_clusters = unsup_transcript[eval_utt]

        if not args.target_word is None and args.target_word not in gt_labels:
            continue

        # Generate plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line = plt.Line2D((0, max(gt_marks)), (0, 0), lw=2, c="k")
        ax.add_line(line)
        for mark in [0,] + gt_marks:
            line = plt.Line2D((mark, mark), (0.05, 0.25), lw=1, c="k")
            ax.add_line(line)
        prev_mark = 0
        for label, mark in zip(gt_labels, gt_marks):
            ax.text(mark - (mark - prev_mark)/2.0, 0.15, label, verticalalignment="center", horizontalalignment="center")
            prev_mark = mark
        for mark in [0,] + unsup_marks:
            line = plt.Line2D((mark, mark), (-0.05, -0.25), lw=1, c="k")
            ax.add_line(line)
        prev_mark = 0
        for label, cluster, mark in zip(unsup_labels, unsup_clusters, unsup_marks):
            ax.text(mark - (mark - prev_mark)/2.0, -0.1, label, verticalalignment="center", horizontalalignment="center")
            ax.text(mark - (mark - prev_mark)/2.0, -0.15 - 0.05, "[$" + str(cluster) + "$]", verticalalignment="center", horizontalalignment="center")
            prev_mark = mark
        if not args.max_length is None:
            plt.xlim([0 - 1, max(gt_marks) + 1])
        else:
            plt.xlim([0 - 1, args.max_length + 1])
        plt.ylim([-1, 1])
        plt.axis("off")
        plot_fn = path.join(plot_dir, eval_utt + ".pdf")
        plt.savefig(plot_fn)
        print "Saving:", plot_fn, "[" + str(max(gt_marks)) + "]"

        # break



if __name__ == "__main__":
    main()
