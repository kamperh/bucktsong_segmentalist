#!/usr/bin/env python

"""
Get the audio file for a particular cluster.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import numpy as np
import os
import subprocess
import sys
import uuid

buckeye_dir = "/endgame/projects/phd/datasets/buckeye"
tsonga_dir = "/endgame/projects/phd/zerospeech/data/tsonga/xitsonga_wavs"

shell = lambda command: subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()[0]


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("segment_fn", type=str, help="pickled segmentation file")
    parser.add_argument("cluster_id", type=str, help="e.g. 'PT1'")
    parser.add_argument("--pad", type=float, help="if given, add padding between tokens", default=0.25)
    parser.add_argument(
        "--no_shuffle", dest="shuffle", action="store_false",
        help="do not shuffle tokens, sort them by utterance label"
        )
    parser.set_defaults(shuffle=True)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def cat_bucktsong_wavs(tokens, wav_fn, pad=None):

    if path.isfile(wav_fn):
        print "Warning: Deleting:", wav_fn
        os.remove(wav_fn)

    tmp_basename = str(uuid.uuid4())
    tmp_wav = tmp_basename + ".wav"

    print "Writing:", wav_fn
    for utt_label, start, end in tokens:
        if utt_label.startswith("nchlt-tso-"):
            input_wav = path.join(tsonga_dir, utt_label.replace("-", "_") + ".wav")
        else:
            input_wav = path.join(buckeye_dir, utt_label[:3], utt_label + ".wav")
        duration = end - start
        sox_cmd = "sox " + input_wav + " " + tmp_wav + " trim " + str(start) + " " + str(duration)
        if pad is not None:
            sox_cmd += " pad 0 " + str(pad)
        shell(sox_cmd)

        # Concatenate wavs
        if path.isfile(wav_fn):
            tmp_wav2 = tmp_basename + ".2.wav"
            shell("sox " + wav_fn + " " + tmp_wav + " " + tmp_wav2)
            os.rename(tmp_wav2, wav_fn)
            os.remove(tmp_wav)
        else:
            os.rename(tmp_wav, wav_fn)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    d = path.split(args.segment_fn)[0]

    print "Reading:", args.segment_fn
    with open(args.segment_fn, "rb") as f:
        unsup_landmarks = pickle.load(f)
        unsup_transcript = pickle.load(f)

    cluster_id = int(args.cluster_id.lower().replace("pt", ""))

    tokens = []  # (utt_label, start, end), e.g. ("s3803a", 413.97, 414.50)
    for utt in unsup_transcript:
        utt_label, interval = utt.split("_")
        utt_start, utt_end = interval.split("-")
        utt_start = int(utt_start)
        utt_end = int(utt_end)
        if cluster_id in unsup_transcript[utt]:
            indices = np.where(np.array(unsup_transcript[utt]) == cluster_id)[0]
            for token_start, token_end in np.array(unsup_landmarks[utt])[indices]:
                tokens.append(
                    (utt_label, float(utt_start + token_start)/100., float(utt_start + token_end)/100.)
                    )

    cat_bucktsong_wavs(tokens, path.join(d, args.cluster_id.lower() + ".wav"), args.pad)


if __name__ == "__main__":
    main()
