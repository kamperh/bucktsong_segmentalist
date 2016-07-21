#!/usr/bin/env python

"""
Get a list of all the segments in the word pairs file.

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
    parser.add_argument("wordpairs_fn", type=str, help="contains all wordpairs")
    parser.add_argument("list_fn", type=str, help="file to output the segments to")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print "Reading:", args.wordpairs_fn
    terms = set()
    for line in open(args.wordpairs_fn):
        # psuedoterm, utt1, speaker1, start1, end1, utt2, speaker2, start2, end2 = line.strip().split(" ")
        cluster, utt1, start1, end1, utt2, start2, end2 = line.strip().split(" ")
        start1 = int(start1)
        end1 = int(end1)
        start2 = int(start2)
        end2 = int(end2)
        terms.add((cluster, utt1, start1, end1))
        terms.add((cluster, utt2, start2, end2))

    print "Writing terms to list:", args.list_fn
    f = open(args.list_fn, "w")
    for cluster, utt, start, end in terms:
        f.write(cluster + "_" + utt + "_" + "%06d" % start + "-" + "%06d" % end + "\n")
    f.close()


if __name__ == "__main__":
    main()

