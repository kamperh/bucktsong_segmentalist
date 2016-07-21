#!/usr/bin/env python

"""
If an overlapping item occur in more than one pair, remove the non-first pair.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""


import argparse
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("input_pairs_fn", type=str)
    parser.add_argument("output_pairs_fn", type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print "Reading:", args.input_pairs_fn
    terms = set()
    i = 0
    for line in open(args.input_pairs_fn):
        # psuedoterm, utt1, speaker1, start1, end1, utt2, speaker2, start2, end2 = line.strip().split(" ")
        cluster, utt1, start1, end1, utt2, start2, end2 = line.strip().split(" ")
        start1 = int(start1)
        end1 = int(end1)
        start2 = int(start2)
        end2 = int(end2)

        # Does first term overlap with existing term?
        for cluster_i, utt_i, start_i, end_i in terms:
            if utt_i != utt1 or (start1 == start_i and end1 == end_i):
                continue
            if start_i <= start1 < end_i or start_i < end1 <= end_i:
                print "!1", cluster, utt1, start1, end1
                i += 1

        # Does second term overlap with existing term?
        for cluster_i, utt_i, start_i, end_i in terms:
            if utt_i != utt2 or (start2 == start_i and end2 == end_i):
                continue
            if start_i <= start2 < end_i or start_i < end2 <= end_i:
                print "!2", cluster, utt2, start2, end2
                i += 1

        terms.add((cluster, utt1, start1, end1))
        terms.add((cluster, utt2, start2, end2))

    print "No. terms:", len(terms)
    print i


if __name__ == "__main__":
    main()
