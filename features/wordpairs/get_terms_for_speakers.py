#!/usr/bin/env python

"""
Get a list of terms for a subset of speakers.

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
    parser.add_argument("input_terms_fn", type=str, help="input terms file in Aren's format")
    parser.add_argument("speakers_fn", type=str, help="file containing the speakers")
    parser.add_argument("output_terms_fn", type=str, help="output terms file in Aren's format")
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

    print "Reading:", args.input_terms_fn
    print "Writing:", args.output_terms_fn
    input_terms_f = open(args.input_terms_fn)
    output_terms_f = open(args.output_terms_fn, "w")
    n_total = 0
    n_terms = 0
    for line in input_terms_f:
        line_split = line.strip().split()
        if len(line_split) == 6:
            _, _, utt, _, _, _ = line_split
        elif len(line_split) == 1:
            utt = line_split[0].split("_")[1]
        else:
            assert False, "invalid input list"
        n_total += 1
        if utt[:3] in speakers or utt[:14] in speakers:
            output_terms_f.write(line)
            n_terms += 1
    input_terms_f.close()
    output_terms_f.close()
    
    print "Wrote", n_terms, "out of", n_total, "terms"    


if __name__ == "__main__":
    main()
