#!/usr/bin/env python

"""
Code the data to raw features without any normalization.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2011-2015
"""

from os import path
import argparse
import glob
import os
import sys

from utils import shell

# Data set and coding variables
buckeye_wavs = "/endgame/projects/phd/datasets/buckeye/*/*.wav"
tsonga_wavs = "/endgame/projects/phd/zerospeech/data/tsonga/xitsonga_wavs/*.wav"
config_fn = path.join("config", "hcopy.wav.mfcc.wb.conf")


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

    if args.dataset == "buckeye":
        wavs = buckeye_wavs
    elif args.dataset == "tsonga":
        wavs = tsonga_wavs

    target_dir = path.join(args.dataset, "raw")
    scp_dir = path.join(args.dataset, "scp")
    log_dir = path.join(args.dataset, "log")

    for d in [target_dir, scp_dir, log_dir]:
        if not path.isdir(d):
            os.makedirs(d)
    target_dir = path.abspath(target_dir)

    raw_scp = path.join(scp_dir, args.dataset + ".mfcc.raw.scp")
    print "Writing raw coding SCP:", raw_scp
    f = open(raw_scp, "w")
    for wav_fn in sorted(glob.glob(wavs)):
        basename = path.splitext(path.split(wav_fn)[-1])[0]
        basename = basename.replace("_", "-")

        mfcc_fn = path.join(target_dir, basename + ".mfcc")

        f.write(wav_fn + " " + mfcc_fn + "\n")
    f.close()

    print "Coding raw MFCCs"
    shell(
        "HCopy -T 7 -A -D -V -S " + raw_scp + " -C " + config_fn + " > " + 
        path.join(log_dir, args.dataset + ".mfcc.raw.log")
        )


if __name__ == "__main__":
    main()
