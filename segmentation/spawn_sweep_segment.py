#!/usr/bin/env python

"""
Spawn jobs for different options given as comma-separated arguments.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

import argparse
import itertools
import subprocess
import sys

sweep_options = [
    "rnd_seed", "S_0_scale", "am_K", "lms", "min_duration", "intrp_lambda",
    "segment_n_iter", "p_boundary_init"
    ]


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument(
        "data_dir", type=str, help="data directory"
        )
    parser.add_argument(
        "--serial", help="runs in parallel by default", action="store_true"
        )
    parser.add_argument(
        "--sd", help="speaker-dependent segmentation; speaker-independent "
        "segmentation is performed by default", action="store_true"
        )
    parser.add_argument(
        "--bigram",
        help="run bigram segmentation; unigram is performed by default",
        action="store_true"
        )
    for option in sweep_options:
        parser.add_argument("--" + option, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    sweep_options_list = []
    sweep_options_value_list = []
    for option in sweep_options:
        attr = getattr(args, option)
        if attr is not None:
            sweep_options_list.append(option)
            sweep_options_value_list.append(attr.split(","))
    print "Speaker dependent:", args.sd

    print "Commands:"
    cmd_list = []
    for cur_sweep_params_values in itertools.product(*sweep_options_value_list):
        option_str = ""
        for i in xrange(len(sweep_options_list)):
            cur_option = sweep_options_list[i]
            cur_option_value = cur_sweep_params_values[i]
            option_str += "--" + cur_option + " "
            option_str += cur_option_value + " "
        if args.bigram:
            if not args.sd:
                cmd = "./bigram_segment.py --log_to_file "
            else:
                assert False
        else:
            if not args.sd:
                cmd = "./segment.py --log_to_file "
            else:
                cmd = "./spawn_segment_sd.py "
        cmd += "--eval "
        cmd += option_str
        cmd += "--data_dir " + args.data_dir if not (args.sd or args.bigram) else args.data_dir
        cmd_list.append(cmd)
        print cmd

    print
    print "-"*79
    if not args.serial:
        # Parallel
        procs = []
        for cmd in cmd_list:
            proc = subprocess.Popen(cmd, shell=True)
            procs.append(proc)
        exit_codes = [proc.wait() for proc in procs]
    else:
        # Serial
        exit_codes = []
        procs = []
        for cmd in cmd_list:
            proc = subprocess.Popen(cmd, shell=True)
            exit_codes.append(proc.wait())
            procs.append(1)
    print "-"*79

    print
    print str(sum([1 for i in exit_codes if i == 0])) + " out of " + str(len(procs)) + " succeeded"


if __name__ == "__main__":
    main()
