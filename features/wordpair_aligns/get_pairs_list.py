#!/usr/bin/env python

"""
Process a pairs file according to the keys provided.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
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
    parser.add_argument("pairs_fn", type=str, help="the pairs file")
    parser.add_argument("keys_fn", type=str, help="the keys file")
    parser.add_argument("output_pairs_fn", type=str, help="the processed pairs file")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print "Reading keys:", args.keys_fn
    keys = [i.strip() for i in open(args.keys_fn)]

    print "Writing processed pairs:", args.output_pairs_fn
    output_pairs_f = open(args.output_pairs_fn, "w")
    n_pair = 0
    i_pair = 0
    for line in open(args.pairs_fn):
        n_pair += 1
        # psuedoterm, utt1, speaker1, start1, end1, utt2, speaker2, start2, end2 = line.strip().split(" ")
        psuedoterm, utt1, start1, end1, utt2, start2, end2 = line.strip().split(" ")
        term1_key = psuedoterm + "_" + utt1 + "_" + "%06d" % int(start1) + "-" + "%06d" % int(end1)
        term2_key = psuedoterm + "_" + utt2 + "_" + "%06d" % int(start2) + "-" + "%06d" % int(end2)

        if term1_key in keys and term2_key in keys:
            output_pairs_f.write(term1_key + " " + term2_key + "\n")
            i_pair += 1
    output_pairs_f.close()        

    print "Wrote", i_pair, "pairs out of", n_pair


if __name__ == "__main__":
    main()
