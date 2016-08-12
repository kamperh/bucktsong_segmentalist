#!/usr/bin/env python

"""
Convert speaker-dependent output to the ZeroSpeech format.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import sys

# speaker_cluster_offset = 10000  # clusters of each speaker will be offset by
#                                 # this, i.e. the first speaker's first cluster
#                                 # will be at 0, while the second speaker's
#                                 # first cluster will be at 10000


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("model_list_fn", type=str, help="speaker-dependent model list file")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print "Reading:", args.model_list_fn
    with open(args.model_list_fn) as f:
        model_dirs = [path.split(i.strip().replace("Log: ", ""))[0] for i in f if i.startswith("Log:")]

    classes = {}  # each entry is a list of (utterance, start, end) for that class

    n_cluster_offset = 0
    for model_dir in sorted(model_dirs):
        pickle_fn = path.join(model_dir, "segment.pkl")

        # Read segmentation
        print "Reading:", pickle_fn
        with open(pickle_fn, "rb") as f:
            unsup_landmarks = pickle.load(f)
            unsup_transcript = pickle.load(f)

        print "Cluster offset:", n_cluster_offset

        for utt in unsup_landmarks:
            utt_label, interval = utt.split("_")
            utt_start, utt_end = interval.split("-")
            utt_start = int(utt_start)
            utt_end = int(utt_end)
            for i_cluster, (token_start, token_end) in zip(
                    unsup_transcript[utt], unsup_landmarks[utt]):
                token_start = float(utt_start + token_start)/100.
                token_end = float(utt_start + token_end)/100.
                i_class = i_cluster + n_cluster_offset
                if i_class not in classes:
                    classes[i_class] = []
                classes[i_class].append((utt_label, token_start, token_end))

        # Offset clusters of next speaker
        clusters = set([cluster for sublist in unsup_transcript.values() for cluster in sublist])
        # assert len(clusters) <= speaker_cluster_offset
        n_cluster_offset += len(clusters)  # += speaker_cluster_offset

    zs_fn = path.join(path.split(args.model_list_fn)[0], "classes.txt")
    n_tokens = 0
    print "Writing:", zs_fn
    with open(zs_fn, "w") as f:
        for c in sorted(classes):
            f.write("Class " + str(c) + "\n")
            for utt, start, end in sorted(classes[c]):
                if utt.startswith("nchlt-tso-"):
                    utt = utt.replace("-", "_")
                f.write(utt + " " + str(start) + " " + str(end) + "\n")
                n_tokens += 1
            f.write("\n")
    print "No. of tokens:", n_tokens


if __name__ == "__main__":
    main()
