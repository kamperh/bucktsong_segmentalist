#!/usr/bin/env python

"""
Get a subset of an npz by only including keys starting in the certain way.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

import argparse
import sys
import numpy as np


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("input_npz_fn", type=str, help="input Numpy archive")
    parser.add_argument(
        "keys_fn", type=str,
        help="only entries with keys starting with one of the entries in this file are retained"
        )
    parser.add_argument("output_npz_fn", type=str, help="output Numpy archive")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print "Reading:", args.input_npz_fn
    input_npz = np.load(args.input_npz_fn)

    print "Reading:", args.keys_fn
    keys = set([i.strip().split(" ")[0] for i in open(args.keys_fn)])

    output_npz = {}
    n_entries = 0
    print "Filtering entries"
    for key in input_npz:
        if key.startswith(tuple(keys)):
            output_npz[key] = input_npz[key]
            n_entries += 1
    print "No. of entries found:", n_entries

    print "Writing:", args.output_npz_fn
    np.savez(args.output_npz_fn, **output_npz)


if __name__ == "__main__":
    main()
