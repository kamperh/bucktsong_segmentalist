#!/usr/bin/env python

"""
Extract a subset of the data for the given speaker.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import numpy as np
import os
import sys

output_dir = "data"


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("data_dir", type=str, help="data directory")
    parser.add_argument("speaker", type=str, help="target speaker")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_data_speaker(speaker, data_dir):

    src_landmarks_dict_fn = path.join(data_dir, "landmarks.pkl")
    src_dense_embeddings_fn = path.join(data_dir, "dense_embeddings.npz")
    src_vec_ids_dict_fn = path.join(data_dir, "vec_ids.pkl")
    src_durations_dict_fn = path.join(data_dir, "durations.pkl")
    src_word_gt_dict_fn = path.join(data_dir, "word_gt.pkl")
    src_phone_gt_dict_fn = path.join(data_dir, "phone_gt.pkl")

    print "Reading:", src_landmarks_dict_fn
    with open(src_landmarks_dict_fn, "rb") as f:
        src_landmarks_dict = pickle.load(f)
    print "Reading:", src_dense_embeddings_fn
    src_dense_embeddings = np.load(src_dense_embeddings_fn)
    print "Reading:", src_vec_ids_dict_fn
    with open(src_vec_ids_dict_fn, "rb") as f:
        src_vec_ids_dict = pickle.load(f)
    print "Reading:", src_durations_dict_fn
    with open(src_durations_dict_fn, "rb") as f:
        src_durations_dict = pickle.load(f)
    print "Reading:", src_word_gt_dict_fn
    with open(src_word_gt_dict_fn, "rb") as f:
        src_word_gt_bound_dict = pickle.load(f)
        src_word_gt_label_dict = pickle.load(f)
    print "Reading:", src_phone_gt_dict_fn
    with open(src_phone_gt_dict_fn, "rb") as f:
        src_phone_gt_bound_dict = pickle.load(f)
        src_phone_gt_label_dict = pickle.load(f)
    print "Total no. of utterances:", len(src_phone_gt_bound_dict)

    print "Speaker:", speaker
    keys = [
        i for i in src_phone_gt_bound_dict.keys() if i.startswith(speaker) or
        i.startswith("nchlt-tso-" + speaker)
        ]
    print "No. of utterances from speaker:", len(keys)

    print "Filtering"
    landmarks_dict = {}
    dense_embeddings = {}
    vec_ids_dict = {}
    durations_dict = {}
    word_gt_bound_dict = {}
    word_gt_label_dict = {}
    phone_gt_bound_dict = {}
    phone_gt_label_dict = {}
    for key in keys:
        landmarks_dict[key] = src_landmarks_dict[key]
        dense_embeddings[key] = src_dense_embeddings[key]
        vec_ids_dict[key] = src_vec_ids_dict[key]
        durations_dict[key] = src_durations_dict[key]
        word_gt_bound_dict[key] = src_word_gt_bound_dict[key]
        word_gt_label_dict[key] = src_word_gt_label_dict[key]
        phone_gt_bound_dict[key] = src_phone_gt_bound_dict[key]
        phone_gt_label_dict[key] = src_phone_gt_label_dict[key]
    print "No. of utterances:", len(phone_gt_bound_dict)

    data_dir = path.join(data_dir, speaker)
    if not path.isdir(data_dir):
        os.makedirs(data_dir)

    landmarks_dict_fn = path.join(data_dir, "landmarks.pkl")
    dense_embeddings_fn = path.join(data_dir, "dense_embeddings.npz")
    vec_ids_dict_fn = path.join(data_dir, "vec_ids.pkl")
    durations_dict_fn = path.join(data_dir, "durations.pkl")
    word_gt_dict_fn = path.join(data_dir, "word_gt.pkl")
    phone_gt_dict_fn = path.join(data_dir, "phone_gt.pkl")
    print "Writing:", landmarks_dict_fn
    with open(landmarks_dict_fn, "wb") as f:
        pickle.dump(landmarks_dict, f, -1)
    print "Writing:", dense_embeddings_fn
    np.savez_compressed(dense_embeddings_fn, **dense_embeddings)
    print "Writing:", vec_ids_dict_fn
    with open(vec_ids_dict_fn, "wb") as f:
        pickle.dump(vec_ids_dict, f, -1)
    print "Writing:", durations_dict_fn
    with open(durations_dict_fn, "wb") as f:
        pickle.dump(durations_dict, f, -1)
    print "Writing:", word_gt_dict_fn
    with open(word_gt_dict_fn, "wb") as f:
        pickle.dump(word_gt_bound_dict, f, -1)
        pickle.dump(word_gt_label_dict, f, -1)
    print "Writing:", phone_gt_dict_fn
    with open(phone_gt_dict_fn, "wb") as f:
        pickle.dump(phone_gt_bound_dict, f, -1)
        pickle.dump(phone_gt_label_dict, f, -1)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    data_dir = args.data_dir
    print "Output directory:", data_dir

    get_data_speaker(args.speaker, data_dir)


if __name__ == "__main__":
    main()
