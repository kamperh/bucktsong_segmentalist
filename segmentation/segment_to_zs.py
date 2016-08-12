#!/usr/bin/env python

"""
Convert speaker-independent output to the ZeroSpeech format.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("segment_fn", type=str, help="pickled segmentation file")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    classes = {}  # each entry is a list of (utterance, start, end) for that class

    # Read segmentation
    print "Reading:", args.segment_fn
    with open(args.segment_fn, "rb") as f:
        unsup_landmarks = pickle.load(f)
        unsup_transcript = pickle.load(f)

    for utt in unsup_landmarks:
        utt_label, interval = utt.split("_")
        utt_start, utt_end = interval.split("-")
        utt_start = int(utt_start)
        utt_end = int(utt_end)
        for i_cluster, (token_start, token_end) in zip(
                unsup_transcript[utt], unsup_landmarks[utt]):
            token_start = float(utt_start + token_start)/100.
            token_end = float(utt_start + token_end)/100.
            i_class = i_cluster
            if i_class not in classes:
                classes[i_class] = []
            classes[i_class].append((utt_label, token_start, token_end))

    zs_fn = path.join(path.split(args.segment_fn)[0], "classes.txt")
    n_tokens = 0
    n_classes = 0
    print "Writing:", zs_fn
    with open(zs_fn, "w") as f:
        for c in sorted(classes):
            n_classes += 1
            f.write("Class " + str(c) + "\n")
            for utt, start, end in sorted(classes[c]):
                if utt.startswith("nchlt-tso-"):
                    utt = utt.replace("-", "_")
                f.write(utt + " " + str(start) + " " + str(end) + "\n")
                n_tokens += 1
            f.write("\n")
    print "No. of classes:", n_classes
    print "No. of tokens:", n_tokens


if __name__ == "__main__":
    main()
