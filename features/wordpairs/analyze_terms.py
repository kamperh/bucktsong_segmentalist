#!/usr/bin/env python

"""
Analyze a given segments file according to the Switchboard naming convention.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

import argparse
import numpy as np
import operator
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("segments_fn", type=str, help="")
    parser.add_argument(
        "--min_count", type=int, help="show statistics for word types with at least this number of tokens", default=None
        )
    parser.add_argument(
        "--min_count_fn", type=str, help="write segments with the minimum count to this file",
        default=None
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

    print
    print "Reading:", args.segments_fn
    labels = [i.strip().split()[0] for i in open(args.segments_fn)]

    word_tokens = []
    lengths = []
    for label in labels:
        word_tokens.append("_".join(label.split("_")[:-2]))
        start, stop = [float(i) for i in label.split("_")[-1].split("-")]
        lengths.append(stop - start)
    print
    print "No. word tokens:", len(word_tokens)
    print "No. word types:", len(set(word_tokens))
    print "Total length (min):", np.sum(lengths)*10e-3 / 60.0
    print "Minimum token length (seconds):", min(lengths)*10e-3
    print "Maximum token length (seconds):", max(lengths)*10e-3

    word_counts = {}
    for word in word_tokens:
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
    sorted_words = sorted(word_counts.iteritems(), key=operator.itemgetter(1))  # http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value\
    print
    print "-"*39
    n_words = 20
    print "Top", n_words, "word types:"
    for word, count in sorted_words[::-1][:n_words]:
        print word + ":", count
    print "-"*39

    if not args.min_count is None:
        min_word_types = set()
        print "Finding words with at least", args.min_count, "tokens"
        for word in word_counts:
            if word_counts[word] >= args.min_count:
                min_word_types.add(word)

        length = 0.0
        n_word_tokens = 0
        min_word_labels = []
        for label in labels:
            word_token = "_".join(label.split("_")[:-2])
            if not word_token in min_word_types:
                continue
            min_word_labels.append(label)
            start, stop = [float(i) for i in label.split("_")[-1].split("-")]
            length += stop - start
            n_word_tokens += 1
        print "No. word types:", len(min_word_types)
        print "No. word tokens:", n_word_tokens
        print "Total length (min):", length*10e-3 / 60.0

        if not args.min_count_fn is None:
            print "Writing segments with minimum count:", args.min_count_fn
            all_segments = [i.split(" ") for i in open(args.segments_fn)]
            min_count_segments = [i for i in all_segments if i[0] in min_word_labels]
            f = open(args.min_count_fn, "w")
            for i in min_count_segments:
                f.write(" ".join(i))
            f.close()

        print


if __name__ == "__main__":
    main()
