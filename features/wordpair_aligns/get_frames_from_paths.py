#!/usr/bin/env python

"""
Extract from the features file the aligned frames in the paths for word pairs.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

import cPickle
import numpy as np
import os
import sys
import datetime


if len(sys.argv) != 6:
    print (
        "usage: " + os.path.basename(__file__) + " features_npz_fn pairs_fn "
        "paths_pkl_fn word1_npy_fn word2_npy_fn"
        )
    sys.exit(1)
features_npz_fn = sys.argv[-5]
pairs_fn = sys.argv[-4]
paths_pkl_fn = sys.argv[-3]
word1_npy_fn = sys.argv[-2]
word2_npy_fn = sys.argv[-1]

# Read files
print "Reading files"
features = np.load(features_npz_fn)
pairs = [line.strip().split(" ") for line in open(pairs_fn)]
f = open(paths_pkl_fn, "rb")
paths = cPickle.load(f)
print "No. paths loaded:", len(paths)
f.close()

n_pairs_notify = len(pairs)//79

# Extract features
print "Start time: " + str(datetime.datetime.now())
print "Notification:", n_pairs_notify, "out of", len(pairs)
print "Extracting features: ",
D = features[features.keys()[0]].shape[1]
# word1_all_frames = np.zeros((0, D))
# word2_all_frames = np.zeros((0, D))
word1_all_frames = []
word2_all_frames = []
for i_pair in range(len(pairs)):
    word1, word2 = pairs[i_pair]
    path = paths[i_pair]
    word1_frames = features[word1]
    word2_frames = features[word2]
    assert (word1_frames.shape[0] - 1, word2_frames.shape[0] - 1) == path[0]
    word1_path, word2_path = zip(*path)
    word1_all_frames.append(word1_frames[word1_path, :])
    word2_all_frames.append(word2_frames[word2_path, :])
    # print word1_frames[word1_path, :]
    # print word1_path
    # print word2_frames[word2_path, :][-10:]
    # print word2_path
    # word1_all_frames = np.concatenate((word1_all_frames, word1_frames[word1_path, :]), axis=0)
    # word2_all_frames = np.concatenate((word2_all_frames, word2_frames[word2_path, :]), axis=0)
    # print path
    if i_pair % n_pairs_notify == 0 and i_pair != 0:
        sys.stdout.write('.')
        # break
print
print "Concatenating features"
word1_all_frames = np.concatenate(list(word1_all_frames), axis=0)
word2_all_frames = np.concatenate(list(word2_all_frames), axis=0)
print "End time: " + str(datetime.datetime.now())

print "Word 1 frames shape:", word1_all_frames.shape
print "Writing word 1 frames:", word1_npy_fn
np.save(word1_npy_fn, word1_all_frames)
print "Word 2 frames shape:", word2_all_frames.shape
print "Writing word 2 frames:", word2_npy_fn
np.save(word2_npy_fn, word2_all_frames)

