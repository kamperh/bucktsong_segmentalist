#!/usr/bin/env python

"""
Strip non-VAD regions from the given pairs terms.

If the two terms overlap in time, they are also removed.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015-2016
"""

from os import path
import argparse
import numpy as np
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("vad_fn", type=str, help="")
    parser.add_argument("input_pairs_fn", type=str, help="")
    parser.add_argument("output_pairs_fn", type=str, help="")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def strip_nonvad(utt, start, end, vads):
    """
    Return
    ------
    (nonvad_start, nonvad_end) : (int, int)
        Updated term indices. None is returned if the term does not fall in a
        VAD region.
    """

    # Get the VAD regions
    vad_starts = [i[0] for i in vads]
    vad_ends = [i[1] for i in vads]

    # Find VAD region with maximum overlap
    overlaps = []
    for (vad_start, vad_end) in zip(vad_starts, vad_ends):
        if vad_end <= start:
            overlaps.append(0)
        elif vad_start >= end:
            overlaps.append(0)
        else:
            overlap = end - start
            if vad_start > start:
                overlap -= vad_start - start
            if vad_end < end:
                overlap -= end - vad_end
            overlaps.append(overlap)
    
    if np.all(np.array(overlaps) == 0):
        # This term isn't in VAD.
        return None

    i_vad = np.argmax(overlaps)
    vad_start = vad_starts[i_vad]
    vad_end = vad_ends[i_vad]
    print "VAD with max overlap:", (vad_start, vad_end)

    # Now strip non-VAD regions
    if vad_start > start:
        start = vad_start
    if vad_end < end:
        end = vad_end

    return (start, end)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    vad_fn = args.vad_fn

    print "Reading VAD regions:", vad_fn
    vad_dict = {}
    for utt, vad_start, vad_end in [i.strip().split(" ") for i in open(vad_fn)]:
        vad_start = int(round(float(vad_start)*100))
        vad_end = int(round(float(vad_end)*100))
        if utt not in vad_dict:
            vad_dict[utt] = []
        vad_dict[utt].append((vad_start, vad_end))

    # Now keep only VAD regions
    print "-"*39
    f = open(args.output_pairs_fn, "w")
    print "Writing updated pairs:", args.output_pairs_fn
    for line in open(args.input_pairs_fn):

        line = line.strip().split(" ")

        if len(line) == 9:
            # Aren's format
            cluster, utt1, speaker1, start1, end1, utt2, speaker2, start2, end2 = line
            start1 = int(start1)
            end1 = int(end1)
            start2 = int(start2)
            end2 = int(end2)
            utt1 = utt1.replace("_", "-")
            utt2 = utt2.replace("_", "-")
        elif len(line) == 6:
            # Sameer's format
            utt1, start1, end1, utt2, start2, end2 = line
            speaker1 = utt1[:3]
            speaker2 = utt2[:3]
            cluster = "?"            
            start1 = int(np.floor(float(start1)*100))
            end1 = int(np.floor(float(end1)*100))
            start2 = int(np.floor(float(start2)*100))
            end2 = int(np.floor(float(end2)*100))

        if utt1 == "s0103a" or utt2 == "s0103a":  # utterance missing from forced alignments
            continue

        if (utt1 == utt2) and (start2 <= start1 <= end2 or start2 <= end1 <= end2):
            print "Warning, pairs from overlapping speech:", utt1, start1, end1, start2, end2
            continue

        # Process the first term in the pair
        print "Processing:", utt1, "(" + str(start1) + ", " + str(end1) + "), cluster", cluster
        if speaker1 == "nch":
            wav1_fn = utt1.replace("-", "_") + ".wav"
            wav2_fn = utt1.replace("-", "_") + ".wav"
        else:
            wav1_fn = "%s/%s.wav" % (speaker1, utt1)
            wav2_fn = "%s/%s.wav" % (speaker2, utt2)
        sox_play_str = (
            "play \"| sox %s -p trim %f %f\"" % (wav1_fn,
            float(start1)/100., float(end1 - start1)/100.)
            )
        print "Raw term play command:", sox_play_str
        nonvad_indices = strip_nonvad(utt1, start1, end1, vad_dict[utt1])
        if nonvad_indices is None:
            continue
        nonvad_start1, nonvad_end1 = nonvad_indices
        if nonvad_start1 != start1 or nonvad_end1 != end1:
            pass
            print "Term changed"
        sox_play_str = (
            "play \"| sox %s -p trim %f %f\"" % (wav1_fn,
            float(nonvad_start1)/100., float(nonvad_end1 - nonvad_start1)/100.)
            )
        print "Term play command after VAD:", sox_play_str

        print
        print "Processing:", utt2, "(" + str(start2) + ", " + str(end2) + "), cluster", cluster
        sox_play_str = (
            "play \"| sox %s -p trim %f %f\"" % (wav2_fn,
            float(start2)/100., float(end2 - start2)/100.)
            )
        print "Raw term play command:", sox_play_str
        nonvad_indices = strip_nonvad(utt2, start2, end2, vad_dict[utt2])
        if nonvad_indices is None:
            continue
        nonvad_start2, nonvad_end2 = nonvad_indices
        if nonvad_start2 != start2 or nonvad_end2 != end2:
            pass
            print "Term changed"
        sox_play_str = (
            "play \"| sox %s -p trim %f %f\"" % (wav2_fn,
            float(nonvad_start2)/100., float(nonvad_end2 - nonvad_start2)/100.)
            )
        print "Term play command after VAD:", sox_play_str

        f.write(
            cluster + " " + utt1 + " " + str(nonvad_start1) + " " +
            str(nonvad_end1) + " " + utt2 + " " + str(nonvad_start2) + " " +
            str(nonvad_end2) + "\n"
            )

        print "-"*39
        # break
    print "Wrote updated pairs:", args.output_pairs_fn


if __name__ == "__main__":
    main()

