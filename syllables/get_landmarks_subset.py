#!/usr/bin/env python

"""
Extract a subset of the landmarks.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import os
import sys

input_landmarks_fn = path.join("landmarks", "buckeye", "landmarks.unsup_syl.pkl")


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("subset", type=str, choices=["devpart1", "zs"], help="target subset")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    speakers_list_fn = path.join("..", "features", "data", args.subset + "_speakers.list")
    print "Reading:", speakers_list_fn
    with open(speakers_list_fn) as f:
        speakers = [i.strip() for i in f]
    print "No. of speakers:", len(speakers)

    print "Reading:", input_landmarks_fn
    with open(input_landmarks_fn, "rb") as f:
        all_landmarks = pickle.load(f)
    print "No. of all utterances:", len(all_landmarks)

    subset_landmarks = {}
    for utt_label in all_landmarks:
        if utt_label[:3] in speakers:
            subset_landmarks[utt_label] = all_landmarks[utt_label]
    print "No. of utterances in subset:", len(subset_landmarks)
    
    output_landmarks_fn = path.join("landmarks", args.subset, "landmarks.unsup_syl.pkl")
    d = path.split(output_landmarks_fn)[0]
    if not path.isdir(d):
        os.makedirs(d)
    print "Writing:", output_landmarks_fn
    with open(output_landmarks_fn, "wb") as f:
        pickle.dump(subset_landmarks, f, -1)


if __name__ == "__main__":
    main()
