#!/usr/bin/env python

"""
Convert a Numpy .npz archive to a Numpy matrix .npy file.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2015
"""

import argparse
import numpy as np
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("npz_fn", type=str, help="the archive filename")
    parser.add_argument("npy_fn", type=str, help="the matrix filename")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print "Reading:", args.npz_fn
    npz = np.load(args.npz_fn)
    print "Loaded", len(npz.keys()), "entries"

    print "Writing:", args.npy_fn
    matrices = []
    for key in npz:
        matrices.append(npz[key])

    np.save(args.npy_fn, np.vstack(matrices))


if __name__ == "__main__":
    main()


