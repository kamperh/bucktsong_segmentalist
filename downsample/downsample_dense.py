#!/usr/bin/env python

"""
Perform dense downsampling over indicated segmentation intervals.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from datetime import datetime
from os import path
import argparse
import cPickle as pickle
import numpy as np
import scipy.signal as signal
import sys

output_dir = "output"


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("subset", type=str, choices=["devpart1", "zs", "tsonga"], help="target subset")
    parser.add_argument("landmarks", type=str, choices=["gtphone", "unsup_syl"], help="landmarks set")
    parser.add_argument(
        "feature_type", type=str, help="input feature type", choices=["mfcc", "cae.d_10", "cae.d_13"]
        )
    parser.add_argument("--n", type=int, help="number of samples (default: %(default)s)", default=10)
    parser.add_argument(
        "--frame_dims", type=int, default=None,
        help="only keep these number of dimensions"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def downsample_utterance(features, seglist, n):
    """
    Return the downsampled matrix with each row an embedding for a segment in
    the seglist.
    """
    embeddings = []
    for i, j in seglist:
        y = features[i:j, :].T
        assert False, "check if the above shouldn't be j+1"
        y_new = signal.resample(y, n, axis=1).flatten("C")
        embeddings.append(y_new)
    return np.asarray(embeddings)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    if args.feature_type == "mfcc":
        input_npz_fn = path.join(
            "..", "features", "subsets", args.subset, args.subset + ".mfcc.cmvn_dd.npz"
            )
    elif args.feature_type == "cae.d_10":
        input_npz_fn = path.join(
            "..", "cae", "encoded", "encoded." +
            args.subset + ".mfcc.cmvn_dd.100-100-100-100-100-100-100-10-100."
            "batch_size2048.corruption0.max_epochs5.correspondence_ae."
            "devpart1_utd.max_epochs120.reverseTrue.layer-2.npz"
            )        
    elif args.feature_type == "cae.d_13":
        input_npz_fn = path.join(
            "..", "cae", "encoded", "encoded." +
            args.subset + ".mfcc.cmvn_dd.100-100-100-100-100-100-100-13-100."
            "batch_size2048.corruption0.max_epochs5.correspondence_ae." +
            args.subset + "_utd.max_epochs120.reverseTrue.layer-2.npz"
            )
    else:
        assert False

    print "Reading:", input_npz_fn
    input_npz = np.load(input_npz_fn)
    d_frame = input_npz[input_npz.keys()[0]].shape[1]
    print "No. of utterances:", len(input_npz.keys())

    seglist_pickle_fn = path.join(output_dir, args.subset, "seglist." + args.landmarks + ".pkl")
    print "Reading:", seglist_pickle_fn
    with open(seglist_pickle_fn, "rb") as f:
        seglist_dict = pickle.load(f)
    print "No. of utterances:", len(seglist_dict)

    print "Frame dimensionality:", d_frame
    if args.frame_dims is not None and args.frame_dims < d_frame:
        d_frame = args.frame_dims
        print "Reducing frame dimensionality:", d_frame

    print "No. of samples:", args.n

    print datetime.now()
    print "Downsampling"
    downsample_dict = {}
    for i, utt in enumerate(input_npz.keys()):
        downsample_dict[utt] = downsample_utterance(
            input_npz[utt][:, :args.frame_dims], seglist_dict[utt], args.n
            )
    print datetime.now()

    output_npz_fn = path.join(
        output_dir, args.subset, "downsample_dense." + args.feature_type +
        ".n_" + str(args.n) + "." + args.landmarks + ".npz"
        )
    print "Writing:", output_npz_fn
    np.savez_compressed(output_npz_fn, **downsample_dict)


if __name__ == "__main__":
    main()
