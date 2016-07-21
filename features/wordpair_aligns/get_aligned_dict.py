#!/usr/bin/env python

"""
Combine the paths into a dictionary with tuples of the utterance IDs as keys.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015-2016
"""

import argparse
import cPickle as pickle
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("pairs_fn", type=str, help="")
    parser.add_argument("paths_pkl_fn", type=str, help="")
    parser.add_argument("dict_pkl_fn", type=str, help="output filename")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def read_pairs(pairs_fn):
    """Return a list of tuples with pairs of utterance IDs."""
    pairs = []
    for line in open(pairs_fn):
        label1, label2 = line.split()
        pairs.append((label1, label2))
    return pairs


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print "Reading:", args.pairs_fn
    pairs = read_pairs(args.pairs_fn)

    print "Reading:", args.paths_pkl_fn
    with open(args.paths_pkl_fn, "rb") as f:
        paths = pickle.load(f)

    paths_dict = {}
    for (label1, label2), path in zip(pairs, paths):
        paths_dict[(label1, label2)] = path

    print "Writing:", args.dict_pkl_fn
    with open(args.dict_pkl_fn, "wb") as f:
        pickle.dump(paths_dict, f, -1)


if __name__ == "__main__":
    main()
