#!/usr/bin/env python

"""
Convert a given list to segment format.

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
    parser.add_argument("list_fn", type=str, help="")
    parser.add_argument("segments_fn", type=str, help="")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print "Writing:", args.segments_fn
    f = open(args.segments_fn, "w")
    with open(args.list_fn) as list_f:
        for label, _, utterance, _, start, end in [i.split() for i in list_f]:
            f.write(label + "_" + utterance + "_%06d-%06d\n" % (int(start), int(end)))
    f.close()


if __name__ == "__main__":
    main()
