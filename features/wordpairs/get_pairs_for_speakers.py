#!/usr/bin/env python

"""
Get pairs for a subset of speakers.

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
    parser.add_argument("input_pairs_fn", type=str, help="input pairs file in Aren's format")
    parser.add_argument("speakers_fn", type=str, help="file containing the speakers")
    parser.add_argument("output_pairs_fn", type=str, help="output pairs file in Aren's format")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print "Reading:", args.speakers_fn
    with open(args.speakers_fn) as f:
        speakers = [line.strip() for line in f]

    print "Reading:", args.input_pairs_fn
    print "Writing:", args.output_pairs_fn
    input_pairs_f = open(args.input_pairs_fn)
    output_pairs_f = open(args.output_pairs_fn, "w")
    n_total = 0
    n_pairs = 0
    for line in input_pairs_f:
        _, utt1, _, _, utt2, _, _ = line.split()
        n_total += 1
        if utt1[:3] in speakers and utt2[:3] in speakers:
            output_pairs_f.write(line)
            n_pairs += 1
    input_pairs_f.close()
    output_pairs_f.close()
    
    print "Wrote", n_pairs, "out of", n_total, "pairs"    


if __name__ == "__main__":
    main()
