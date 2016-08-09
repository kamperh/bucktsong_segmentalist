#!/usr/bin/env python

"""
Get all the speaker-dependent sets for this subset.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015, 2016
"""

from os import path
import argparse
import sys

import get_data_speaker

devpart1_speakers = [
    "s02", "s04", "s05", "s08", "s12", "s16", "s03", "s06", "s10", "s11",
    "s13", "s38"
    ]
zs_speakers = [
    "s20", "s25", "s27", "s01", "s26", "s31", "s29", "s23", "s24", "s32",
    "s33", "s30"
    ]
tsonga_speakers = [
    "001m", "102f", "103f", "104f", "126f", "127f", "128m", "129f", "130m",
    "131f", "132m", "133f", "134m", "135m", "136f", "138m", "139f", "140f",
    "141m", "142m", "143m", "144m", "145m", "146f"
    ]


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("subset", type=str, choices=["devpart1", "zs", "tsonga"], help="target subset")
    parser.add_argument("landmarks", type=str, choices=["gtphone", "unsup_syl"], help="landmarks set")
    parser.add_argument("feature_type", type=str, help="input feature type", choices=["mfcc", "cae.d_13"])
    parser.add_argument(
        "n_samples", type=int, help="the number of samples used in downsampling"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    if args.subset == "devpart1":
        speakers = devpart1_speakers
    elif args.subset == "zs":
        speakers = zs_speakers
    elif args.subset == "tsonga":
        speakers = tsonga_speakers

    data_dir = path.join(
        "data", args.subset, args.feature_type + ".n_" + str(args.n_samples) + "." + args.landmarks
        )
    print "Output directory:", data_dir

    for speaker in speakers:
        print
        get_data_speaker.get_data_speaker(speaker, data_dir)


if __name__ == "__main__":
    main()

