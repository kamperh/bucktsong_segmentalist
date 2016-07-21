#!/usr/bin/env python

"""
Write the extracted syllable landmarks into a dictionary which is pickled.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import scipy.io
import sys
import os

# mat_syl_fn = path.join("thetaOscillator", "wavs_test_bounds_t.mat")


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("dataset", type=str, choices=["buckeye", "tsonga"])
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    mat_syl_fn = path.join("thetaOscillator", args.dataset + "_bounds_t.mat")
    landmarks_dir = path.join("landmarks", args.dataset)
    landmarks_fn = path.join(landmarks_dir, "landmarks.unsup_syl.pkl")

    if not path.isdir(landmarks_dir):
        os.makedirs(landmarks_dir)

    print "Reading:", mat_syl_fn
    mat = scipy.io.loadmat(mat_syl_fn)
    n_wavs = mat["wav_files"].shape[0]

    landmarks = {}
    for i_wav in xrange(n_wavs):
        wav_label = path.splitext(path.split(str(mat["wav_files"][i_wav][0][0]))[-1])[0]
        bounds = [int(round(float(i[0])*100.0)) for i in mat["bounds_t"][i_wav][0]]
        landmarks[wav_label] = bounds[1:]  # remove first (0) landmark

    print "Writing:", landmarks_fn
    with open(landmarks_fn, "wb") as f:
        pickle.dump(landmarks, f, -1)


if __name__ == "__main__":
    main()
