#!/usr/bin/env python

"""
Get landmarks at ground truth phone boundaries.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from os import path
import argparse
import cPickle as pickle
import glob
import numpy as np
import re
import sys

# buckeye_dir = "/endgame/projects/phd/datasets/buckeye/"
output_dir = "output"
forced_alignment_dir = path.join("..", "features", "data", "forced_alignment")


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("subset", type=str, choices=["devpart1", "zs", "tsonga"], help="target subset")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_landmarks_from_forced_alignment(fa_fn):
    print "Reading:", fa_fn
    prev_utterance = ""
    prev_token_label = ""
    prev_end_time = -1
    start_time = -1
    landmarks = []
    landmarks_dict = {}
    with open(fa_fn, "r") as f:
        for line in f:
            utterance, start_token, end_token, token_label = line.strip().split()
            start_token = float(start_token)
            end_token = float(end_token)
            utterance = utterance.replace("_", "-")

            # if not utterance == "s0305b":
            #     continue

            if token_label in ["SIL", "SPN"]:
                continue
            if prev_end_time != start_token or prev_utterance != utterance:
                if prev_end_time != -1:
                    # segments.append((prev_utterance, start_time, prev_end_time))
                    # print prev_utterance, start_time, prev_end_time, landmarks, 
                    cur_vad_start = int(round(float(start_time) * 100))
                    cur_vad_end = int(round(float(prev_end_time) * 100))
                    cur_landmarks = [int(round(float(i) * 100)) for i in landmarks]
                    assert cur_vad_end == cur_landmarks[-1]
                    key = prev_utterance + "_%06d-%06d" % (cur_vad_start, cur_vad_end)
                    landmarks_dict[key] = [i - cur_vad_start for i in cur_landmarks]
                    # print landmarks_dict[key], key
                    landmarks = []
                start_time = start_token

            landmarks.append(end_token)
            prev_end_time = end_token
            prev_token_label = token_label
            prev_utterance = utterance
    
        # print prev_utterance, start_time, prev_end_time, landmarks, 
        cur_vad_start = int(round(float(start_time) * 100))
        cur_vad_end = int(round(float(prev_end_time) * 100))
        cur_landmarks = [int(round(float(i) * 100)) for i in landmarks]
        assert cur_vad_end == cur_landmarks[-1]
        key = prev_utterance + "_%06d-%06d" % (cur_vad_start, cur_vad_end)
        landmarks_dict[key] = [i - cur_vad_start for i in cur_landmarks]
        # print landmarks_dict[key], key

    return landmarks_dict


# def get_landmarks_dict_from_transcript(phones_transcript_fn):
#     basename = path.splitext(path.split(phones_transcript_fn)[-1])[0]
#     landmarks_dict = {}
#     with open(phones_transcript_fn) as f:
#         vad_prev_line_start = True
#         vad_start_time = 0.
#         vad_prev_time = 0.
#         line = ""
#         landmarks = []
#         while not line.startswith("#"):
#             # Transcription begins after the "#"
#             line = f.readline()
#         for line in f:

#             line = line.strip().split()
#             if len(line) < 3:
#                 # The end of the file, or unlabelled line
#                 continue

#             time, _, phone_label = line[:3]
#             time = float(time)

#             if not re.match("^[a-z]{1,3}", phone_label):
#                 # This is not a phone
#                 if not vad_prev_line_start:
#                     cur_vad_start = int(round(float(vad_start_time) * 100))
#                     cur_vad_end = int(round(float(vad_prev_time) * 100))
#                     cur_landmarks = [int(round(float(i) * 100)) for i in landmarks]
#                     assert cur_vad_end == cur_landmarks[-1]
#                     key = basename + "_%06d-%06d" % (cur_vad_start, cur_vad_end)
#                     landmarks_dict[key] = cur_landmarks
#                 vad_prev_line_start = True
#                 vad_start_time = time
#                 landmarks = []
#             else:
#                 # This is a phone
#                 landmarks.append(time)
#                 vad_prev_time = time
#                 vad_prev_line_start = False
#     return landmarks_dict


# def get_landmarks_from_transcript(phones_transcript_fn):
#     landmarks = []
#     with open(phones_transcript_fn) as f:
#         line = ""
#         while not line.startswith("#"):
#             # Transcription begins after the "#"
#             line = f.readline()
#         for line in f:
#             line = line.strip().split()
#             if len(line) < 3:
#                 # The end of the file, or unlabelled line
#                 continue
#             time, _, phone_label = line[:3]
#             time = float(time)
#             if re.match("^[a-z]{1,3}", phone_label):
#                 new_landmark = int(round(float(time) * 100))
#                 if len(landmarks) == 0 or new_landmark != landmarks[-1]:
#                     landmarks.append(new_landmark)
#     return landmarks


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    speakers_list_fn = path.join("..", "features", "data", args.subset + "_speakers.list")
    print "Reading:", speakers_list_fn
    with open(speakers_list_fn) as f:
        speakers = [i.strip() for i in f]
    print "No. of speakers:", len(speakers)

    if args.subset == "devpart1" or args.subset == "zs":
        fa_fn = path.join(forced_alignment_dir, "english.phn")
    elif args.subset == "tsonga":
        fa_fn = path.join(forced_alignment_dir, "xitsonga.phn")

    # print "Searching:", buckeye_dir
    # landmarks_recording_dict = {}
    # for speaker in speakers:
    #     for phones_transcript_fn in glob.glob(path.join(buckeye_dir, speaker, speaker + "*.phones")):
    #         basename = path.splitext(path.split(phones_transcript_fn)[-1])[0]
    #         landmarks_recording_dict[basename] = get_landmarks_from_transcript(phones_transcript_fn)

    npz_fn = path.join("..", "features", "subsets", args.subset, args.subset + ".mfcc.cmvn_dd.npz")
    print "Reading:", npz_fn
    npz = np.load(npz_fn)
    print "No. of utterances:", len(npz.keys())

    # landmarks_dict = {}
    # print "Getting landmarks"
    # for key in sorted(npz.keys()):
    #     utt, interval = key.split("_")
    #     start, end = interval.split("-")
    #     start = int(start)
    #     end = int(end)
    #     landmarks = [i - start for i in landmarks_recording_dict[utt] if i > start and i <= end]
    #     if len(landmarks) == 0:
    #         landmarks = [npz[key].shape[0] - 1,]
    #     landmarks_dict[key] = landmarks
    #     # if len(landmarks_dict[key]) == 0:
    #     #     print "wow"
    #     #     print str(key) + str(landmarks_dict[key]) + " " +str(npz[key].shape) + " " + str([i for i in landmarks_recording_dict[utt] if i > start and i <= end])
    #     #     # assert False
    #     # elif landmarks_dict[key][-1] - npz[key].shape[0] > 2:
    #     #     print str(key) + str(landmarks_dict[key]) + " " +str(npz[key].shape) + " " + str([i for i in landmarks_recording_dict[utt] if i > start and i <= end])
    #     #     print "2"
    #     #     # assert False
    #     # break
    #     print key
    #     print landmarks
    #     break

    landmarks_dict = get_landmarks_from_forced_alignment(fa_fn)
    print "Total no. of utterances:", len(landmarks_dict)

    subset_landmarks_dict = {}
    for npz_key in npz:
        label, interval = npz_key.split("_")
        start, stop = interval.split("-")
        start = int(start)
        stop = int(stop)

        if not npz_key in landmarks_dict:
            # Shorter audio might have led to truncation
            start_of_key = "-".join(npz_key.split("-")[:-1])
            landmarks_key = [i for i in landmarks_dict.keys() if i.startswith(start_of_key)]
            assert len(landmarks_key) == 1
            landmarks = landmarks_dict[landmarks_key[0]]
            landmarks[-1] = stop - start
        else:
            landmarks = landmarks_dict[npz_key]

        assert landmarks[-1] == stop - start

        subset_landmarks_dict[npz_key] = landmarks 
    landmarks_dict = subset_landmarks_dict
    print "No. of utterances:", len(landmarks_dict)

    key = sorted(landmarks_dict.keys())[0]
    print (
        "Example landmarks for '" + key + "' with " + str(npz[key].shape[0])
        + " frames: " + str(landmarks_dict[key])
        )

    # print "Searching:", buckeye_dir
    # landmarks_dict2 = {}
    # for speaker in speakers:
    #     for phones_transcript_fn in glob.glob(path.join(buckeye_dir, speaker, speaker + "*.phones")):
    #         landmarks_dict2.update(get_landmarks_dict_from_transcript(phones_transcript_fn))
    # print "No. of utterances:", len(landmarks_dict2)

    landmarks_fn = path.join(output_dir, args.subset, "landmarks.gtphone.pkl")
    print "Writing:", landmarks_fn
    with open(landmarks_fn, "wb") as f:
        pickle.dump(landmarks_dict, f, -1)


if __name__ == "__main__":
    main()
