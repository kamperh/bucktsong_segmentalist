#!/usr/bin/env python

"""
Analyze a given Buckeye speaker list.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2015
"""

import argparse
import sys

# bukeye_dir = "/endgame/projects/phd/datasets/buckeye/"
speakers_list_fn = "../data/buckeye_speakers.txt"


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("list_fn", type=str, help="utterances list to analyze")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Read speaker information: each value is speaker gender, speaker age, interviewer gender
    speaker_info = dict(
        [(line.split()[0], line.split()[1:]) for line in open(speakers_list_fn) if line.startswith("s")]
        )

    # Read list
    print "Reading list:", args.list_fn
    utterances = [i.strip() for i in open(args.list_fn)]
    print
    print "No. utterances:", len(utterances)

    speakers = set([i[:3] for i in utterances])
    print "No. speakers:", len(speakers)

    male_speakers = [i for i in speakers if speaker_info[i][0] == "m"]
    female_speakers = [i for i in speakers if speaker_info[i][0] == "f"]

    print
    print "Female speakers:", sorted(list(female_speakers))
    print "Male speakers:", sorted(list(male_speakers))

    print "No. female speakers:", len(female_speakers)
    print "No. male speakers:", len(male_speakers)
    print "No. young speakers:", len([i for i in speakers if speaker_info[i][1] == "y"])
    print "No. old speakers:", len([i for i in speakers if speaker_info[i][1] == "o"])
    print "No. female interviewers:", len([i for i in speakers if speaker_info[i][2] == "f"])
    print "No. male interviewers:", len([i for i in speakers if speaker_info[i][2] == "m"])


if __name__ == "__main__":
    main()
