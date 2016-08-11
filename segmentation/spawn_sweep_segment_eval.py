#!/usr/bin/env python

"""
Evaluate swept segmentation output.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import sys

print_options = ["rnd_seed", "S_0_scale", "am_K", "lms", "min_duration", "intrp_lambda"]
print_results = [
     "n_clusters", "mean_n_landmarks_crossed", "n_singleton", "word_bounds_2",
    "word_token_2", "avg_purity", "uWER_many", "dp_error", "uWER"
    ]


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("spawn_fn", type=str, help="output from spawn job; can also be pickle filename")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    if path.splitext(args.spawn_fn)[-1] == ".pkl":
        segment_eval_dict_pickles = [args.spawn_fn]
    else:
        with open(args.spawn_fn) as f:
            lines = [line.strip() for line in f]

        sd = True if lines[0].split(": ")[-1] == "True" else False
        print "Speaker dependent:", sd

        if sd:
            segment_eval_dict_pickles = [
                line.split(": ")[-1] for line in lines if line.endswith("segment_eval_dict.pkl")
                ]
        else:
            segment_eval_dict_pickles = [
                path.join(path.split(line.split(": ")[-1])[0],
                "segment_eval_dict.pkl") for line in lines if
                line.startswith("Log:")
                ]

    results = []  # list of (dir, options_dict, result_dict)
    for segment_eval_dict_fn in segment_eval_dict_pickles:
        directory = path.split(segment_eval_dict_fn)[0]
        options_dict_fn = path.join(directory, "options_dict.pkl")
        with open(segment_eval_dict_fn, "rb") as f:
            result_dict = pickle.load(f)
        with open(options_dict_fn, "rb") as f:
            options_dict = pickle.load(f)
        results.append((directory, options_dict, result_dict))

    # Present results
    options = results[0][1].keys()
    print "Possible options:", options
    scores = results[0][2].keys()
    print "Possible scores:", scores

    print
    print "-"*79
    print "# Directory\t" + "\t".join(print_options) + "\t" + "\t".join(print_results)
    for directory, options_dict, result_dict in results:
        print (
            directory + "\t" + "\t".join([str(options_dict[i]) if i !=
            "intrp_lambda" else str(options_dict["lm_params"]["intrp_lambda"]
            if "lm_params" in options_dict else "") for i in print_options]) +
            "\t" + "\t".join([str(result_dict[i]) for i in print_results])
            )
    print "-"*79


if __name__ == "__main__":
    main()
