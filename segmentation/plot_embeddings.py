#!/usr/bin/env python

"""
Plot the embeddings from a given segmentation file.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from utils import plotting
from utils import cluster_analysis
from utils import cluster_model_present
import segment_eval

logging.disable(logging.CRITICAL)


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("segment_fn", type=str, help="pickled segmentation file")
    parser.add_argument(
        "--level", type=str, help="unit at which evaluation is performed (default: %(default)s)",
        choices=["phone", "word"], default="word"
        )
    parser.add_argument("--speaker", type=str, help="target speaker")
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

    # # Temp
    # for utt in unsup_transcript:
    #     if 931 in unsup_transcript[utt]:
    #         print utt
    #         print unsup_transcript[utt]
    #         print unsup_landmarks[utt]
    # return

    # phone_gt_fn = path.join(data_dir, "phone_gt.pkl")
    # print "Reading:", phone_gt_fn
    # with open(phone_gt_fn, "rb") as f:
    #     phone_gt_bounds = pickle.load(f)
    #     phone_gt_labels = pickle.load(f)

    if args.level == "phone":
        gt_fn = path.join(data_dir, "phone_gt.pkl")
        interval_mapping_func = segment_eval.intervals_to_seq
    elif args.level == "word":
        gt_fn = path.join(data_dir, "word_gt.pkl")
        interval_mapping_func = segment_eval.intervals_to_max_overlap
    else:
        assert False

    print "Reading:", gt_fn
    with open(gt_fn, "rb") as f:
        gt_bounds = pickle.load(f)
        gt_labels = pickle.load(f)

    analysis = segment_eval.analyze_segmentation(
        unsup_landmarks, unsup_transcript, gt_bounds, gt_labels, interval_mapping_func
        )
    cluster_to_label_map = analysis["cluster_to_label_map"]
    labels_true = analysis["labels_true"]
    labels_pred = analysis["labels_pred"]
    clusters = cluster_analysis.analyse_clusters(labels_true, labels_pred)
    n_biggest = 20
    sizes, _ = cluster_model_present.get_sizes_purities(clusters)
    biggest_clusters = list(np.argsort(sizes)[-n_biggest:])

    dummy_segmenter = segment_eval.init_segmenter_from_output(args.segment_fn, data_dir=data_dir)

    # Perform analysis
    clusters = []
    embeddings = []
    for i_utt in range(dummy_segmenter.utterances.D):
        cur_utt = dummy_segmenter.ids_to_utterance_labels[i_utt]
        cur_embeds = dummy_segmenter.utterances.get_segmented_embeds_i(i_utt)
        cur_clusters = dummy_segmenter.get_unsup_transcript_i(i_utt)
        for cluster, i_embed in zip(cur_clusters, cur_embeds):
            if cluster != -1:
                cur_embed = dummy_segmenter.acoustic_model.components.X[i_embed]
                embeddings.append(cur_embed)
                clusters.append(cluster)
    embeddings = np.array(embeddings)
    plotting.plot_embeddings_with_mapping(
        embeddings, clusters, cluster_to_label_map,#, n_samples=500,
        filter_clusters=biggest_clusters
        )
    plt.show()

    # # Read segmentation
    # print "Reading:", args.segment_fn
    # with open(args.segment_fn, "rb") as f:
    #     unsup_landmarks = pickle.load(f)
    #     unsup_transcript = pickle.load(f)

    # intervals = unsup_landmarks[unsup_landmarks.keys()[0]]
    # print intervals
    # print segment_eval.intervals_to_landmarks(intervals)


if __name__ == "__main__":
    main()
