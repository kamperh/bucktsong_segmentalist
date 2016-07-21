#!/usr/bin/env python

"""
Perform cepstral mean and variance normalization and add deltas to features.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import os
import sys

from utils import shell


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

    # Directory and configuration variables
    scp_dir = path.join(args.dataset, "scp")
    log_dir = path.join(args.dataset, "log")
    config_dir = "config"

    cmn_dir = path.join(args.dataset, "cmn")
    cvn_dir = path.join(args.dataset, "cvn")
    cmvn_dd_dir = path.join(args.dataset, "cmvn_dd")

    cmn_config_fn = path.join(config_dir, "cmn.conf")
    cvn_config_fn = path.join(config_dir, "cvn." + args.dataset + ".conf")
    cmvn_dd_config_fn = path.join(config_dir, "cmvn_dd." + args.dataset + ".conf")

    segments_scp_fn = path.join(scp_dir, args.dataset + ".mfcc.raw.segments.scp")
    cmvn_dd_scp_fn = path.join(scp_dir, args.dataset + ".mfcc.cmvn_dd.scp")
    cmvn_dd_coding_scp_fn = path.join(scp_dir, args.dataset + ".mfcc.cmvn_dd.coding.scp")

    for d in [cmn_dir, cvn_dir, cmvn_dd_dir]:
        if not path.isdir(d):
            os.makedirs(d)
    cmvn_dd_dir = path.abspath(cmvn_dd_dir)

    # Get mask from config
    mask = [i.strip().split() for i in open(cvn_config_fn) if "CMEANMASK" in i][0][-1]

    print "Getting CMN vectors"
    shell(
        "HCompV -A -D -V -T 1 -C " + cmn_config_fn + " -c " + cmn_dir + " -k " + mask
        + " -q m -S " + segments_scp_fn
        )

    print "Getting CVN vectors"
    shell(
        "HCompV -A -D -V -T 1 -C " + cvn_config_fn + " -c " + cvn_dir + " -k " + mask
        + " -q v -S " + segments_scp_fn
        )

    print "Writing CMVN DD SCP:", cmvn_dd_scp_fn
    print "Writing CMVN DD coding SCP:", cmvn_dd_coding_scp_fn
    cmvn_dd_scp = open(cmvn_dd_scp_fn, "w")
    cmvn_dd_coding_scp = open(cmvn_dd_coding_scp_fn, "w")
    segments_scp = open(segments_scp_fn)
    for line in segments_scp:
        cmvn_dd_mfcc_fn = path.join(cmvn_dd_dir, line.split("=")[0])
        cmvn_dd_scp.write(cmvn_dd_mfcc_fn + "\n")
        cmvn_dd_coding_scp.write(line.strip() + " " + cmvn_dd_mfcc_fn + "\n")
    segments_scp.close()
    cmvn_dd_scp.close()
    cmvn_dd_coding_scp.close()

    # Get unit covariance file
    unit_covar_fn = "unit.covar"
    f = open(unit_covar_fn, "w")
    f.write("<VARSCALE> 39\n")
    f.write(" 1.000000e+00" *39)
    f.close()

    print "Coding to CMVN DD"
    shell(
        "HCopy -T 7 -A -D -V -S " + cmvn_dd_coding_scp_fn + " -C " + cmvn_dd_config_fn + " > " + path.join(log_dir, "cmvn_dd.log")
        )

    os.remove(unit_covar_fn)


if __name__ == "__main__":
    main()
