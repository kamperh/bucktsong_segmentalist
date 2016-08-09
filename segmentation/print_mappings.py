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
# import matplotlib.pyplot as plt
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

    # Print random mappings
    random.seed(1)
    sample_keys = random.sample(unsup_transcript.keys(), 100)
    n_true_tokens_min = 4
    for eval_utt in sample_keys:
        if len(gt_word_labels[eval_utt]) > n_true_tokens_min:
            print
            print gt_word_bounds[eval_utt]
            print [i[1] for i in unsup_landmarks[eval_utt]]
            print
            print gt_word_labels[eval_utt]
            print [cluster_to_label_map[i] if i in cluster_to_label_map else "" for i in unsup_transcript[eval_utt]]
            print [cluster_to_label_map_many[i] if i in cluster_to_label_map_many else "" for i in unsup_transcript[eval_utt]]
            print unsup_transcript[eval_utt]
            print [clusters[i]["size"] for i in unsup_transcript[eval_utt]]
            print [clusters[i]["purity"] for i in unsup_transcript[eval_utt]]
            print
            break


if __name__ == "__main__":
    main()
