#!/usr/bin/env python

"""
Evaluate the output from speaker-dependent segmentation.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import numpy as np
import re
import sys

from segment_eval import precision_recall_f
from utils import dp_align


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("model_list_fn", type=str, help="speaker-dependent model list file")
    parser.add_argument("--write_dict", action="store_true", help="write the result dictionary to file")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                    SPEAKER-DEPENDENT EVALUATION FUNCTIONS                   #
#-----------------------------------------------------------------------------#

def segment_sd_eval(model_list_fn):

    print "Reading:", model_list_fn
    with open(model_list_fn) as f:
        model_dirs = [path.split(i.strip().replace("Log: ", ""))[0] for i in f if i.startswith("Log:")]

    overall_result_dict = {}
    for model_dir in model_dirs:
        result_dict_fn = path.join(model_dir, "segment_eval_dict.pkl")
        print "Reading:", result_dict_fn
        with open(result_dict_fn, "rb") as f:
            result_dict = pickle.load(f)
        for key in result_dict:
            if not key in overall_result_dict:
                overall_result_dict[key] = []
            overall_result_dict[key].append(result_dict[key])

    # Load last options_dict
    with open(path.join(path.split(model_list_fn)[0], "options_dict.pkl"), "rb") as f:
        options_dict = pickle.load(f)
    print
    print "Options:", options_dict

    result_dict = {}
    print
    print "Avg. no. clusters:", np.mean(overall_result_dict["n_clusters"])
    print "Std. no. clusters:", np.std(overall_result_dict["n_clusters"])
    print "Avg. no. unmapped clusters:", np.mean(overall_result_dict["n_unmapped_clusters"])
    print "Std. no. unmapped clusters:", np.std(overall_result_dict["n_unmapped_clusters"])
    mean_n_landmarks_crossed = np.sum(
        [i*j for i, j in zip(overall_result_dict["mean_n_landmarks_crossed"], overall_result_dict["n_tokens"])]
        ) / np.sum(overall_result_dict["n_tokens"])
    print "Mean landmarks crossed:", mean_n_landmarks_crossed
    result_dict["mean_n_landmarks_crossed"] = mean_n_landmarks_crossed
    print
    print "Avg. no. singleton clusters:", np.mean(overall_result_dict["n_singleton"])
    print "Std. no. singleton clusters:", np.std(overall_result_dict["n_singleton"])
    if "n_singleton_unmapped" in overall_result_dict:
        print "Avg. no. unmapped singleton clusters:", np.mean(overall_result_dict["n_singleton_unmapped"])
        print "Std. no. unmapped singleton clusters:", np.std(overall_result_dict["n_singleton_unmapped"])
        print "Mean landmarks crossed singleton clusters:", np.sum(
            [i*j for i, j in
            zip(overall_result_dict["mean_n_singleton_landmarks_crossed"],
            overall_result_dict["n_singleton"])]
            ) / np.sum(overall_result_dict["n_singleton"])
    result_dict["n_singleton"] = np.mean(overall_result_dict["n_singleton"])
    result_dict["n_clusters"] = np.mean(overall_result_dict["n_clusters"])

    print
    print "Avg. clustering average purity:", np.mean(overall_result_dict["avg_purity"])
    print "Std. clustering average purity:", np.std(overall_result_dict["avg_purity"])
    print "Avg. clustering one-to-one accuracy:", np.mean(overall_result_dict["one_to_one"])
    print "Std. clustering one-to-one accuracy:", np.std(overall_result_dict["one_to_one"])
    ned = np.sum(
        [i*j for i, j in zip(overall_result_dict["ned"], overall_result_dict["n_ned_pairs"])]
        ) / np.sum(overall_result_dict["n_ned_pairs"])
    print "NED:", ned
    result_dict["avg_purity"] = np.mean(overall_result_dict["avg_purity"])
    result_dict["one_to_one"] = np.mean(overall_result_dict["one_to_one"])
    result_dict["ned"] = ned

    dp_error = dp_align.DPError()
    for dp_error_sd in overall_result_dict["dp_error"]:
        dp_error += dp_align.DPError(**dp_error_sd)
    result_dict["dp_error"] = {
        "n_del": dp_error.n_del,
        "n_ins": dp_error.n_ins,
        "n_sub": dp_error.n_sub,
        "n_match": dp_error.n_match,
        "n_total": dp_error.n_total,
        }
    print "Errors:", dp_error
    dp_error_many = dp_align.DPError()
    for dp_error_many_sd in overall_result_dict["dp_error_many"]:
        dp_error_many += dp_align.DPError(**dp_error_many_sd)
    result_dict["dp_error_many"] = {
        "n_del": dp_error_many.n_del,
        "n_ins": dp_error_many.n_ins,
        "n_sub": dp_error_many.n_sub,
        "n_match": dp_error_many.n_match,
        "n_total": dp_error_many.n_total,
        }
    print "No. of utterances:", np.sum(overall_result_dict["n_utterances"])
    print "No. of tokens:", np.sum(overall_result_dict["n_tokens"])
    print "uWER:", dp_error.get_wer()
    print "uWER_many:", dp_error_many.get_wer()
    result_dict["uWER"] = dp_error.get_wer()
    result_dict["uWER_many"] = dp_error_many.get_wer()

    def get_precision_recall_f(key, subtract_n_utterances=False):
        n_boundaries_correct = 0
        n_boundaries_seg = 0
        n_boundaries_ref = 0
        for i_speaker, (precision, recall, f) in enumerate(overall_result_dict[key]):
            cur_n_boundaries_seg = overall_result_dict["n_tokens"][i_speaker]
            cur_n_boundaries_ref = overall_result_dict["dp_error"][i_speaker]["n_total"]
            if subtract_n_utterances:
                cur_n_boundaries_seg -= overall_result_dict["n_utterances"][i_speaker]
                cur_n_boundaries_ref -= overall_result_dict["n_utterances"][i_speaker]
            cur_n_boundaries_correct = int(precision*cur_n_boundaries_seg)
            n_boundaries_correct += cur_n_boundaries_correct
            n_boundaries_seg += cur_n_boundaries_seg
            n_boundaries_ref += cur_n_boundaries_ref
        print "Boundaries correct:", n_boundaries_correct
        print "Boundaries segmented:", n_boundaries_seg
        print "Boundaries reference:", n_boundaries_ref
        return precision_recall_f(n_boundaries_correct, n_boundaries_seg, n_boundaries_ref)

    if "word_bounds_onephone" in overall_result_dict:
        print
        print "Word boundaries:"
        precision, recall, f = get_precision_recall_f("word_bounds_onephone", True)
        result_dict["word_bounds_onephone"] = (precision, recall, f)
        print "tolerance = one phone: P = " + str(precision) + ", R = " + str(recall) + ", F = " + str(f)
        precision, recall, f = get_precision_recall_f("word_bounds_2", True)
        result_dict["word_bounds_2"] = (precision, recall, f)
        print "tolerance = 2: P = " + str(precision) + ", R = " + str(recall) + ", F = " + str(f)

    if "word_token_onephone" in overall_result_dict:
        print "Word token scores:"
        precision, recall, f = get_precision_recall_f("word_token_onephone")
        result_dict["word_token_onephone"] = (precision, recall, f)
        print "tolerance = one phone: P = " + str(precision) + ", R = " + str(recall) + ", F = " + str(f)
        precision, recall, f = get_precision_recall_f("word_token_2")
        result_dict["word_token_2"] = (precision, recall, f)
        print "tolerance = 2: P = " + str(precision) + ", R = " + str(recall) + ", F = " + str(f)

    # if "phone_bounds_2" in overall_result_dict:
    #     # The below is not correct since there are many more reference phone boundaries
    #     print "Phone boundaries:"
    #     precision, recall, f = get_precision_recall_f("phone_bounds_2")
    #     print "tolerance = 2: P = " + str(precision) + ", R = " + str(recall) + ", F = " + str(f)

    if "log_marg*length" in overall_result_dict:
        print
        print "Avg. final log [P(X, z)*length]:", np.mean(overall_result_dict["log_marg*length"])
        print "Std. final log [P(X, z)*length]:", np.std(overall_result_dict["log_marg*length"])
        result_dict["log_marg*length"] = np.mean(overall_result_dict["log_marg*length"])

    return result_dict


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    result_dict = segment_sd_eval(args.model_list_fn)

    if args.write_dict:
        result_dict_fn = path.join(path.split(args.model_list_fn)[0], "segment_eval_dict.pkl")
        print("Writing: " + result_dict_fn)
        with open(result_dict_fn, "wb") as f:
            pickle.dump(result_dict, f, -1)


if __name__ == "__main__":
    main()
