#!/usr/bin/env python

"""
Analyze a given npz file following some of the Buckeye naming conventions.

Author: Herman Kamper
Contact: kamperh@gmail.com
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
    parser.add_argument("npz_fn", type=str, help="the npz archive to analyze")
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
    keys = sorted(npz.keys())

    print "E.g. key:", keys[0]
    print "E.g. feature:", npz[keys[0]][0, :]

    lengths = []
    for key in keys:
        N, D = npz[key].shape
        lengths.append(N)
    print "Maximum length (frames):", max(lengths)
    print "No. frames:", sum(lengths)

    matrices = []
    for key in keys:
        matrices.append(npz[key])
    data = np.vstack(matrices)
    print "Overall mean vector:", data[:,:].mean(axis=0)
    print "Overall variance vector:", data[:,:].var(axis=0)
    print "Overall mean:", data[:,:].mean()
    print "Overall variance:", data[:,:].var()

    filter_terms = keys[0].split("_")[0]
    print "Investigating entries starting with:", filter_terms
    keys = [i for i in keys if i.startswith(filter_terms)]
    matrices = []
    for key in keys:
        matrices.append(npz[key])
    print "Found", len(matrices), "entries matching filter"
    data = np.vstack(matrices)

    print "Mean vector for", filter_terms, "entries:", data[:,:].mean(axis=0)
    print "Variance vector for", filter_terms, "entries:", data[:,:].var(axis=0)


if __name__ == "__main__":
    main()
