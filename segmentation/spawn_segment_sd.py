#!/usr/bin/env python

"""
Perform speaker-dependent segmentation of all the speakers in a subset.

Options should be changed in `segment.py`.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import os
import hashlib
import subprocess
import sys

from get_data_sd import devpart1_speakers, zs_speakers, tsonga_speakers
from segment import default_options_dict
from spawn_segment_sd_eval import segment_sd_eval

# devpart1_speakers = ["s02", "s04", "s05"]  # temp
sd_options = ["rnd_seed", "S_0_scale", "am_K", "lms", "min_duration"]


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument(
        "data_dir", type=str, help="data directory; should contain a "
        "subdirectory for every speaker"
        )
    parser.add_argument(
        "--am_K", type=str, help="default: %(default)s",
        default=default_options_dict["am_K"]
        )
    parser.add_argument(
        "--S_0_scale", type=float, help="default: %(default)s",
        default=default_options_dict["S_0_scale"]
        )
    parser.add_argument(
        "--wip", type=float, help="default: %(default)s",
        default=default_options_dict["wip"]
        )
    parser.add_argument(
        "--p_boundary_init", type=float, help="default: %(default)s",
        default=default_options_dict["p_boundary_init"]
        )    
    parser.add_argument(
        "--lms", type=float, help="default: %(default)s",
        default=default_options_dict["lms"]
        )
    parser.add_argument(
        "--rnd_seed", type=int, help="default: %(default)s",
        default=default_options_dict["rnd_seed"]
        )
    parser.add_argument(
        "--min_duration", type=int, help="default: %(default)s",
        default=default_options_dict["min_duration"]
        )
    parser.add_argument(
        "--segment_n_iter", type=int, help="default: %(default)s",
        default=default_options_dict["segment_n_iter"]
        )
    parser.add_argument(
        "--time_power_term", type=float, help="default: %(default)s",
        default=default_options_dict["time_power_term"]
        )
    parser.add_argument(
        "--eval", help="evaluate output from segmentation", action="store_true"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def eval_model_list(model_list_fn):
    print("Evaluating: " + model_list_fn)
    output_fn = path.join(path.split(model_list_fn)[0], "segment_eval.txt")
    print("Writing: " + output_fn)
    with open(output_fn, "w") as f:
        sys.stdout = f
        result_dict = segment_sd_eval(model_list_fn)
    sys.stdout = sys.__stdout__
    result_dict_fn = path.join(path.split(model_list_fn)[0], "segment_eval_dict.pkl")
    print("Writing: " + result_dict_fn)
    with open(result_dict_fn, "wb") as f:
        pickle.dump(result_dict, f, -1)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    _, subset, _ = path.normpath(args.data_dir).split(os.sep)

    if subset == "devpart1":
        speakers = devpart1_speakers
    elif subset == "zs":
        speakers = zs_speakers
    elif subset == "tsonga":
        speakers = tsonga_speakers

    # Check that all speakers are in the set directory
    for speaker in speakers:
        assert path.isdir(path.join(args.data_dir, speaker))

    # Set options (this is actually just for hashing purposes)
    options_dict = default_options_dict.copy()
    options_dict["data_dir"] = path.normpath(args.data_dir)
    options_dict["am_K"] = args.am_K
    options_dict["S_0_scale"] = args.S_0_scale
    options_dict["wip"] = args.wip
    options_dict["p_boundary_init"] = args.p_boundary_init
    options_dict["lms"] = args.lms
    options_dict["rnd_seed"] = args.rnd_seed
    options_dict["min_duration"] = args.min_duration
    options_dict["segment_n_iter"] = args.segment_n_iter
    options_dict["time_power_term"] = args.time_power_term
    options_dict["log_to_file"] = True

    hasher = hashlib.md5(repr(sorted(options_dict.items())).encode("ascii"))
    hash_str = hasher.hexdigest()[:10]
    model_dir = path.join(
        options_dict["model_dir"], options_dict["data_dir"].replace("data" + os.sep, ""), "sd_" + hash_str
        )
    if not path.isdir(model_dir):
        os.makedirs(model_dir)
    output_fn = path.join(model_dir, "models.txt")
    options_dict_fn = path.join(model_dir, "options_dict.pkl")

    print "Writing:", options_dict_fn
    with open(options_dict_fn, "wb") as f:
        pickle.dump(options_dict, f, -1)

    output_fn = path.join(model_dir, "models.txt")
    print "Writing:", output_fn
    with open(output_fn, "w") as f:
        f.write("")  # empty out the log file

    # Construct command list
    cmd_list = []
    for speaker in speakers:
        data_dir = path.join(args.data_dir, speaker)
        cmd = "./segment.py --eval --log_to_file --data_dir " + data_dir + " "
        for option in sd_options:
            cmd += "--" + option + " " + str(options_dict[option]) + " "
        cmd += ">> " + output_fn
        cmd_list.append(cmd)

    # Spawn jobs
    # http://stackoverflow.com/questions/23611396/python-execute-cat-subprocess-in-parallel
    procs = []
    for cmd in cmd_list:
        proc = subprocess.Popen(cmd, shell=True)
        procs.append(proc)
    exit_codes = [proc.wait() for proc in procs]

    print str(sum([1 for i in exit_codes if i == 0])) + " out of " + str(len(procs)) + " succeeded"

    if args.eval:
        eval_model_list(output_fn)


if __name__ == "__main__":
    main()
