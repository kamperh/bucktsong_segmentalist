#!/usr/bin/env python

"""
Get the segmentation intervals over given landmarks.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import sys

output_dir = "output"
n_landmarks_max = 6


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("subset", type=str, choices=["devpart1", "zs", "tsonga"], help="target subset")
    parser.add_argument("landmarks", type=str, choices=["gtphone", "unsup_syl"], help="landmarks set")
    parser.add_argument(
        "--n_landmarks_max", type=int,
        help="maximum number of landmarks to cross (default: %(default)s)", default=n_landmarks_max
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_seglist_from_landmarks(landmarks, n_landmarks_max):
    seglist = []
    prev_landmark = 0
    for i in range(len(landmarks)):
        for j in landmarks[i:i + n_landmarks_max]:
            seglist.append((prev_landmark, j))
        prev_landmark = landmarks[i]
    return seglist


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    landmarks_pickle_fn = path.join(output_dir, args.subset, "landmarks." + args.landmarks + ".pkl")
    print "Reading:", landmarks_pickle_fn
    with open(landmarks_pickle_fn, "rb") as f:
        landmarks_dict = pickle.load(f)
    print "No. of utterances:", len(landmarks_dict)

    print "Getting seglist"
    seglist_dict = {}
    for utt in landmarks_dict.keys():
        seglist_dict[utt] = get_seglist_from_landmarks(landmarks_dict[utt], args.n_landmarks_max)

    seglist_fn = path.join(output_dir, args.subset, "seglist." + args.landmarks + ".pkl")
    print "Writing:", seglist_fn
    with open(seglist_fn, "wb") as f:
        pickle.dump(seglist_dict, f, -1)


if __name__ == "__main__":
    main()
