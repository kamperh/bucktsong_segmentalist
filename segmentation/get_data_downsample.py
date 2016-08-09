#!/usr/bin/env python

"""
Get the Buckeye data used to perform segmentation.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import glob
import numpy as np
import re
import sys
import os

wordembeds_dir = "../../../../downsample/output"  # relative to data/subset/embeding_label/
forced_alignment_dir = path.join("..", "features", "data", "forced_alignment")
output_dir = "data"


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("subset", type=str, choices=["devpart1", "zs", "tsonga"], help="target subset")
    parser.add_argument("landmarks", type=str, choices=["gtphone", "unsup_syl"], help="landmarks set")
    parser.add_argument(
        "feature_type", type=str, help="input feature type", choices=["mfcc", "cae.d_10", "cae.d_13"]
        )
    parser.add_argument(
        "n_samples", type=int, help="the number of samples used in downsampling"
        )
    parser.add_argument(
        "--n_landmarks_max", type=int,
        help="maximum number of landmarks to cross (default: %(default)s)", default=6
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                              GET DATA FUNCTIONS                             #
#-----------------------------------------------------------------------------#

def get_vec_ids_dict(lengths_dict, n_landmarks_max):
    """
    Every N(N + 1)/2 length vector `vec_ids` contains all the indices for a
    particular utterance. For t = 1, 2, ..., N the entries `vec_ids[i:i
    + t]` contains the IDs of embedding[0:t] up to embedding[t - 1:t], with i =
    t(t - 1)/2. Written out: `vec_ids` = [embed[0:1], embed[0:2], embed[1:2],
    embed[0:3], ..., embed[N-1:N]].
    """
    vec_ids_dict = {}
    for utt in sorted(lengths_dict.keys()):
        i_embed = 0
        n_slices = lengths_dict[utt]
        vec_ids = -1*np.ones((n_slices**2 + n_slices)/2, dtype=int)
        for cur_start in range(n_slices):
            for cur_end in range(cur_start, min(n_slices, cur_start + n_landmarks_max)):
                cur_end += 1
                t = cur_end
                i = t*(t - 1)/2
                vec_ids[i + cur_start] = i_embed
                i_embed += 1
        vec_ids_dict[utt] = vec_ids
        # print utt, lengths_dict[utt], vec_ids
    return vec_ids_dict


def get_durations_dict(landmarks_dict, n_landmarks_max):
    durations_dict = {}
    for utt in sorted(landmarks_dict.keys()):
        landmarks = [0,] + landmarks_dict[utt]
        N = len(landmarks)  # should be n_slices + 1
        durations = -1*np.ones(((N - 1)**2 + (N - 1))/2, dtype=int)
        j = 0
        for t in xrange(1, N):
            for i in range(t):
                if t - i > N - 1:
                    j += 1
                    continue
                durations[j] = landmarks[t] - landmarks[i]
                j += 1
        durations_dict[utt] = durations
    return durations_dict


def get_tokens_from_fa(fa_fn):
    print "Reading:", fa_fn
    labels_dict = {}
    bounds_dict = {}
    with open(fa_fn, "r") as f:
        for line in f:
            utt, start, end, label = line.strip().split()
            start = int(round(float(start) * 100))
            end = int(round(float(end) * 100))
            utt = utt.replace("_", "-")

            if label in ["SIL", "SPN"]:
                continue

            if not utt in labels_dict:
                labels_dict[utt] = []
                bounds_dict[utt] = []

            labels_dict[utt].append(label)
            bounds_dict[utt].append(end)
    return labels_dict, bounds_dict


def get_word_gt_dicts(subset, utt_ids):

    if subset == "devpart1" or subset == "zs":
        fa_fn = path.join(forced_alignment_dir, "english.wrd")
    elif subset == "tsonga":
        fa_fn = path.join(forced_alignment_dir, "xitsonga.wrd")

    word_labels_fa_dict, word_bounds_fa_dict = get_tokens_from_fa(fa_fn)
    
    word_labels_dict = {}
    word_bounds_dict = {}
    print "Getting word boundaries and labels"
    for key in sorted(utt_ids):
        utt, interval = key.split("_")
        start, end = interval.split("-")
        start = int(start)
        end = int(end)
        word_bounds = [i - start for i in word_bounds_fa_dict[utt] if i > start and i <= end]
        word_labels = [
            word_labels_fa_dict[utt][word_bounds_fa_dict[utt].index(i)]
            for i in word_bounds_fa_dict[utt] if i > start and i <= end
            ]
        # if len(word_bounds) == 0:
        #     print "Warning: no word labels found for:", key
        #     word_bounds.append(end)
        #     word_labels.append("unk")
        if len(word_bounds) == 0 or not word_bounds[-1] == end - start:
            # Truncation occurred
            assert len(word_bounds) <= 1 or end - start > word_bounds[-2]
            word_bounds.append(end - start)
            word_labels.append([
                word_labels_fa_dict[utt][word_bounds_fa_dict[utt].index(i)]
                for i in word_bounds_fa_dict[utt] if i > end
                ][0])
        assert word_bounds[-1] == end - start
        assert len(word_bounds) == len(word_labels)
        word_bounds_dict[key] = word_bounds
        word_labels_dict[key] = word_labels
    return word_bounds_dict, word_labels_dict


def get_phone_gt_dicts(subset, utt_ids):

    if subset == "devpart1" or subset == "zs":
        fa_fn = path.join(forced_alignment_dir, "english.phn")
    elif subset == "tsonga":
        fa_fn = path.join(forced_alignment_dir, "xitsonga.phn")

    phone_labels_fa_dict, phone_bounds_fa_dict = get_tokens_from_fa(fa_fn)
    
    phone_labels_dict = {}
    phone_bounds_dict = {}
    print "Getting phone boundaries and labels"
    for key in sorted(utt_ids):
        utt, interval = key.split("_")
        start, end = interval.split("-")
        start = int(start)
        end = int(end)
        phone_bounds = [i - start for i in phone_bounds_fa_dict[utt] if i > start and i <= end]
        phone_labels = [
            phone_labels_fa_dict[utt][phone_bounds_fa_dict[utt].index(i)]
            for i in phone_bounds_fa_dict[utt] if i > start and i <= end
            ]
        if len(phone_bounds) == 0 or not phone_bounds[-1] == end - start:
            # Truncation occurred
            assert len(phone_bounds) <= 1 or end - start > phone_bounds[-2]
            phone_bounds.append(end - start)
            phone_labels.append([
                phone_labels_fa_dict[utt][phone_bounds_fa_dict[utt].index(i)]
                for i in phone_bounds_fa_dict[utt] if i > end
                ][0])
        # if len(phone_bounds) == 0:
        #     print "Warning: no phone labels found for:", key
        #     phone_bounds.append(end)
        #     phone_labels.append("unk")
        # if not phone_bounds[-1] == end - start:
        #     # Truncation occurred
        #     assert end - start > phone_bounds[-2]
        #     phone_bounds.append(end - start)
        assert phone_bounds[-1] == end - start
        phone_bounds_dict[key] = phone_bounds
        phone_labels_dict[key] = phone_labels
    return phone_bounds_dict, phone_labels_dict


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    data_dir = path.join(
        output_dir, args.subset, args.feature_type + ".n_" + str(args.n_samples) + "." + args.landmarks
        )
    print "Output directory:", data_dir
    if not path.isdir(data_dir):
        os.makedirs(data_dir)

    # Create symbolic link to embeddings file
    src_dense_embeddings_fn = path.join(
        wordembeds_dir, args.subset, "downsample_dense." + args.feature_type + ".n_" +
        str(args.n_samples) + "." + args.landmarks + ".npz"
        )
    target_dense_embeddings_fn = path.join(data_dir, "dense_embeddings.npz")
    if not path.islink(target_dense_embeddings_fn):
        print "Linking:", src_dense_embeddings_fn
        os.symlink(src_dense_embeddings_fn, target_dense_embeddings_fn)

    # Create symbolic link to landmarks file
    src_landmarks_fn = path.join(
        wordembeds_dir, args.subset, "landmarks." + args.landmarks + ".pkl"
        )
    target_landmarks_fn = path.join(data_dir, "landmarks.pkl")
    if not path.islink(target_landmarks_fn):
        print "Linking:", src_landmarks_fn
        os.symlink(src_landmarks_fn, target_landmarks_fn)

    print "Reading:", target_landmarks_fn
    with open(target_landmarks_fn, "rb") as f:
        landmarks_dict = pickle.load(f)
    print "No. of utterances:", len(landmarks_dict)

    lengths_dict = dict([(i, len(landmarks_dict[i])) for i in landmarks_dict.keys()])
    utt_ids = landmarks_dict.keys()

    print "Getting vec_ids"
    vec_ids_dict = get_vec_ids_dict(lengths_dict, args.n_landmarks_max)
    print "Getting durations"
    durations_dict = get_durations_dict(landmarks_dict, args.n_landmarks_max)
    word_gt_bounds_dict, word_gt_labels_dict = get_word_gt_dicts(args.subset, utt_ids)
    key = word_gt_bounds_dict.keys()[0]
    phone_gt_bounds_dict, phone_gt_labels_dict = get_phone_gt_dicts(args.subset, utt_ids)
    assert (
        len(vec_ids_dict) == len(durations_dict) == len(word_gt_bounds_dict) ==
        len(word_gt_labels_dict) == len(phone_gt_bounds_dict) ==
        len(phone_gt_labels_dict)
        )
    print "No. of utterances:", len(word_gt_bounds_dict)

    # Analyze phone set
    phone_set = set()
    for utt in phone_gt_labels_dict:
        phone_set.update(phone_gt_labels_dict[utt])
    print "No. of phones:", len(phone_set)
    print "Phone set:", sorted(list(phone_set))

    vec_ids_dict_fn = path.join(data_dir, "vec_ids.pkl")
    durations_dict_fn = path.join(data_dir, "durations.pkl")
    word_gt_fn = path.join(data_dir, "word_gt.pkl")
    phone_gt_fn = path.join(data_dir, "phone_gt.pkl")
    print "Writing:", vec_ids_dict_fn
    with open(vec_ids_dict_fn, "wb") as f:
        pickle.dump(vec_ids_dict, f, -1)
    print "Writing:", durations_dict_fn
    with open(durations_dict_fn, "wb") as f:
        pickle.dump(durations_dict, f, -1)
    print "Writing:", word_gt_fn
    with open(word_gt_fn, "wb") as f:
        pickle.dump(word_gt_bounds_dict, f, -1)
        pickle.dump(word_gt_labels_dict, f, -1)
    print "Writing:", phone_gt_fn
    with open(phone_gt_fn, "wb") as f:
        pickle.dump(phone_gt_bounds_dict, f, -1)
        pickle.dump(phone_gt_labels_dict, f, -1)


if __name__ == "__main__":
    main()
