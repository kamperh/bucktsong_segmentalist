#!/usr/bin/env python

"""
Evaluate segmentation output.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from collections import Counter
from datetime import datetime
from os import path
import argparse
import cPickle as pickle
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys

from utils import cluster_analysis
from utils import cluster_model_present
from utils import dp_align
import bigram_segment
import segment

buckeye_speaker_genders = {
    "s01": "f", "s02": "f", "s03": "m", "s04": "f", "s05": "f", "s06": "m",
    "s07": "f", "s08": "f", "s09": "f", "s10": "m", "s11": "m", "s12": "f",
    "s13": "m", "s14": "f", "s15": "m", "s16": "f", "s17": "f", "s18": "f",
    "s19": "m", "s20": "f", "s21": "f", "s22": "m", "s23": "m", "s24": "m",
    "s25": "f", "s26": "f", "s27": "f", "s28": "m", "s29": "m", "s30": "m",
    "s31": "f", "s32": "m", "s33": "m", "s34": "m", "s35": "m", "s36": "m",
    "s37": "f", "s38": "m", "s39": "f", "s40": "m"
    }


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("segment_fn", type=str, help="pickled segmentation file")
    parser.add_argument(
        "--level", type=str, help="unit at which evaluation is performed (default: %(default)s)",
        choices=["phone", "word"], default="word"
        )
    parser.add_argument("--write_dict", action="store_true", help="write the result dictionary to file")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                      SEGMENTATION EVALUATION FUNCTIONS                      #
#-----------------------------------------------------------------------------#

def landmarks_to_intervals(landmarks):
    intervals = []
    j_prev = 0
    for j in landmarks:
        intervals.append((j_prev, j))
        j_prev = j
    return intervals


def intervals_to_landmarks(intervals):
    landmarks = []
    for i, j in intervals:
        landmarks.append(j)
    return landmarks


def intervals_to_seq(intervals, gt_intervals, gt_labels,
        n_min_frames_overlap=3):
    """
    Convert the given intervals to ground truth sequences.

    The convention of Versteegh et al. [Interspeech, 2015] is followed where
    the transcription of an interval is taken as the forced aligned phones or
    words overlapping with that interval. In the case of phones, if the left or
    right edge of the interval contains part of a phone, that phone is included
    in the transcription if the overlap corresponds to at least 30 ms or at
    least 50% of the phone's duration. The `n_min_frames_overlap` controls the
    number of frames overlap. For words, a value of 20 can be used.
    """
    seq = [[] for i in range(len(intervals))]
    for i_interval, (interval_left, interval_right) in enumerate(intervals):
        for i_unit, (gt_left, gt_right) in enumerate(gt_intervals):
            label = gt_labels[i_unit]
            if gt_right <= interval_left or gt_left >= interval_right:
                continue
            else:
                overlap = interval_right - interval_left
                if gt_left > interval_left:
                    overlap -= gt_left - interval_left
                if gt_right < interval_right:
                    overlap -= interval_right - gt_right
                # print label, (gt_left, gt_right), i_interval, (interval_left, interval_right), interval_right - gt_right, overlap
            duration = gt_right - gt_left
            # print duration, float(overlap)/duration
            if ((overlap == (interval_right - interval_left))   # entire interval is covered
                    or (overlap >= n_min_frames_overlap)        # overlap is at least 30 ms (or 200 ms)
                    or (duration != 0 and
                        float(overlap)/duration >= 0.5)   # at least 50% of unit
                    ):
                seq[i_interval].append(label)
    return seq


def intervals_to_max_overlap(intervals, gt_intervals, gt_labels):
    """
    Each interval is mapped to the grount truth label with maximum overlap.
    """
    seq = []
    for unsup_interval in intervals:
        overlaps = []
        for gt_interval in gt_intervals:
            if gt_interval[1] <= unsup_interval[0]:
                overlaps.append(0)
            elif gt_interval[0] >= unsup_interval[1]:
                overlaps.append(0)
            else:
                overlap = unsup_interval[1] - unsup_interval[0]
                if gt_interval[0] > unsup_interval[0]:
                    overlap -= (gt_interval[0] - unsup_interval[0])
                if gt_interval[1] < unsup_interval[1]:
                    overlap -= (unsup_interval[1] - gt_interval[1])
                overlaps.append(overlap)
        seq.append([gt_labels[np.argmax(overlaps)]]) 
    return seq


def init_segmenter_from_output(pickle_fn, data_dir=None):
    """Creates a dummy segmenter from previous output."""

    print("Reading: " + pickle_fn)
    with open(pickle_fn, "rb") as f:
        unsup_landmarks = pickle.load(f)
        unsup_transcript = pickle.load(f)

    d = path.split(pickle_fn)[0]

    unsup_bounds = {}
    for utt in unsup_landmarks:
        unsup_bounds[utt] = intervals_to_landmarks(unsup_landmarks[utt])

    # Set options
    options_dict_fn = path.join(d, "options_dict.pkl")
    print "Reading:", options_dict_fn
    with open(options_dict_fn, "rb") as f:
        options_dict = pickle.load(f)
    options_dict["init_am_n_iter"] = 0
    options_dict["segment_n_iter"] = 0
    options_dict["seed_bounds"] = unsup_bounds
    options_dict["seed_assignments"] = unsup_transcript
    if data_dir is not None:
        options_dict["data_dir"] = data_dir

    # Perform dummy segmentation
    if options_dict["fb_type"] in ["unigram", "bigram"]:
        # Bigram segmenter
        segmenter = bigram_segment.segment(options_dict, suppress_pickling=True)
    else:
        # Unigram segmenter
        segmenter = segment.segment(options_dict, suppress_pickling=True)

    return segmenter


def analyze_segmentation(unsup_landmarks, unsup_transcript,
        gt_bounds, gt_labels, interval_mapping_func=intervals_to_seq):
    
    analysis = {}  # return values are added to this dictionary

    print "Converting true boundaries to intervals"
    gt_intervals = {}
    for utt in gt_bounds:
        gt_intervals[utt] = landmarks_to_intervals(gt_bounds[utt])

    unsup_transcript_seq = {}
    for utt in unsup_transcript:
        # print utt
        # print unsup_landmarks[utt]
        # print gt_intervals[utt]
        # print gt_labels[utt]
        unsup_transcript_seq[utt] = interval_mapping_func(
            unsup_landmarks[utt], gt_intervals[utt],
            gt_labels[utt]
            )

    print "Mapping clusters to sequences"
    labels_pred = []
    labels_true = []
    for utt in unsup_transcript:
        indices = np.where(np.array(unsup_transcript[utt]) != -1)[0]
        labels_pred.extend([unsup_transcript[utt][i] for i in indices])
        labels_true.extend([" ".join(unsup_transcript_seq[utt][i]) for i in indices])
    one_to_one, cluster_to_label_map = cluster_analysis.one_to_one_mapping(labels_true, labels_pred)
    cluster_to_label_map_many = cluster_analysis.many_to_one_mapping(labels_true, labels_pred)
    analysis["one_to_one"] = one_to_one
    analysis["cluster_to_label_map"] = cluster_to_label_map
    analysis["cluster_to_label_map_many"] = cluster_to_label_map_many
    analysis["labels_true"] = labels_true
    analysis["labels_pred"] = labels_pred

    return analysis


def ned(s1, s2):
    dp_errors = dp_align.dp_align(s1, s2, ins_penalty=1, del_penalty=1, sub_penalty=1)
    ued = dp_errors.get_levenshtein()
    return float(ued) / max(len(s1), len(s2))


def clusters_ned(labels_true, labels_pred):
    """
    Calculate the normalized edit distance (NED) for each cluster and overall.

    Return
    ------
    per_cluster_ned, ned, n_pairs : dict, float, int
        The per-cluster NED (normalized by the number of pairs in the cluster)
        and the overall NED.
    """
    per_cluster_ned = {}
    neds = []
    for i_cluster in sorted(set(labels_pred))[4:]:
        cluster_indices = list(np.where(labels_pred == i_cluster)[0])
        cluster_neds = []
        for i, j in itertools.combinations(cluster_indices, 2):
            cluster_neds.append(ned(labels_true[i].split(" "), labels_true[j].split(" ")))
        if len(cluster_neds) != 0:
            per_cluster_ned[i_cluster] = sum(cluster_neds) / len(cluster_neds)
            neds.extend(cluster_neds)
    if len(neds) == 0:
        return per_cluster_ned, np.inf, len(neds)
    return per_cluster_ned, sum(neds)/len(neds), len(neds)


def score_boundaries(ref, seg, tolerance=0):
    """
    Calculate precision, recall, F-score for the utterance bounadries.

    Parameters
    ----------
    ref : list of vector of bool
        The ground truth reference.
    seg : list of vector of bool
        The segmentation hypothesis. Every vector in the list are in the format
        of the rows of the `Utterances.boundaries`.
    tolerance : int
        The number of slices with which a boundary might differ but still be
        regarded as correct.

    Return
    ------
    output : (float, float, float)
        Precision, recall, F-score.
    """
    n_boundaries_ref = 0
    n_boundaries_seg = 0
    n_boundaries_correct = 0
    for i_boundary, boundary_ref in enumerate(ref):
        assert boundary_ref[-1]  # check if last boundary is True
        assert seg[i_boundary][-1]
        boundary_ref = boundary_ref[:-1]  # last boundary is always True, don't want to count this
        boundary_seg = seg[i_boundary][:-1]

        boundary_ref = list(np.nonzero(boundary_ref)[0])
        boundary_seg = list(np.nonzero(boundary_seg)[0])
        n_boundaries_ref += len(boundary_ref)
        n_boundaries_seg += len(boundary_seg)

        for i_seg in boundary_seg:
            for i, i_ref in enumerate(boundary_ref):
                if abs(i_seg - i_ref) <= tolerance:
                    n_boundaries_correct += 1
                    boundary_ref.pop(i)
                    break

    # # Temp
    # print "n_boundaries_correct", n_boundaries_correct
    # print "n_boundaries_seg", n_boundaries_seg
    # print "n_boundaries_ref", n_boundaries_ref

    # precision = float(n_boundaries_correct)/n_boundaries_seg
    # recall = float(n_boundaries_correct)/n_boundaries_ref
    # if precision + recall != 0:
    #     f = 2*precision*recall / (precision + recall)
    # else:
    #     f = -np.inf

    return precision_recall_f(n_boundaries_correct, n_boundaries_seg, n_boundaries_ref)


def precision_recall_f(n_correct, n_pred, n_true):
    precision = float(n_correct)/n_pred
    recall = float(n_correct)/n_true
    if precision + recall != 0:
        f = 2*precision*recall / (precision + recall)
    else:
        f = -np.inf
    # print "n_boundaries_correct", n_correct
    # print "n_boundaries_seg", n_pred
    # print "n_boundaries_ref", n_true
    return precision, recall, f


def get_word_boundary_scores(gt_phone_bounds, gt_word_bounds, seg_bounds):
    """
    Segmentation boundaries within one phone from a ground truth word boundary
    is taken as correct.
    """
    n_boundaries_correct = 0
    n_boundaries_ref = 0
    n_boundaries_seg = 0
    for utt in gt_word_bounds:
        cur_gt_word_bounds = gt_word_bounds[utt][:-1]  # last boundary is always correct, shouldn't count
        cur_gt_phone_bounds = gt_phone_bounds[utt]
        seg = seg_bounds[utt][:-1]
        n_boundaries_seg += len(seg)
        n_boundaries_ref += len(cur_gt_word_bounds)
        word_bound_intervals = []
        for word_bound in cur_gt_word_bounds:
            i_word_bound = cur_gt_phone_bounds.index(word_bound)
            phone_bound = cur_gt_phone_bounds[max(0, i_word_bound - 1): i_word_bound + 2]
            word_bound_intervals.append((phone_bound[0], phone_bound[-1]))
        for seg_bound in [i[1] for i in seg]:
            for i_gt_bound, (start, end) in enumerate(word_bound_intervals):
                if start <= seg_bound <= end:
                    n_boundaries_correct += 1
                    word_bound_intervals.pop(i_gt_bound)  # can't re-use this boundary
                    break
    # precision = float(n_boundaries_correct)/n_boundaries_seg
    # recall = float(n_boundaries_correct)/n_boundaries_ref
    # if precision + recall != 0:
    #     f = 2*precision*recall / (precision + recall)
    # else:
    #     f = -np.inf
    return precision_recall_f(n_boundaries_correct, n_boundaries_seg, n_boundaries_ref)


def get_word_token_scores(gt_phone_bounds, gt_word_bounds, seg_bounds,
        tolerance=None):
    """
    If tolerance is provided, this is the number of frames with which a
    boundary can differ from the ground truth word boundary, and this is used
    instead of `gt_phone_bounds`.
    """
    n_boundaries_correct = 0
    n_boundaries_ref = 0
    n_boundaries_seg = 0
    for utt in gt_word_bounds:
        cur_gt_word_bounds = gt_word_bounds[utt]
        cur_gt_phone_bounds = gt_phone_bounds[utt]
        seg = seg_bounds[utt]
        # print "cur_gt_word_bounds:", cur_gt_word_bounds
        # print "cur_gt_phone_bounds:", cur_gt_phone_bounds
        # print "seg:", [i[1] for i in seg]

        # Build list of ((word_start_lower, word_start_upper), (word_end_lower, word_end_upper))
        word_bound_intervals = []
        for word_start, word_end in landmarks_to_intervals(cur_gt_word_bounds):
            i_word_start_bound = 0 if word_start == 0 else cur_gt_phone_bounds.index(word_start)
            i_word_end_bound = cur_gt_phone_bounds.index(word_end)
            if tolerance is None:
                phone_start_bound = (
                    [0] if word_start == 0 else cur_gt_phone_bounds[max(0,
                    i_word_start_bound - 1): i_word_start_bound + 2]
                    )
                phone_end_bound = cur_gt_phone_bounds[max(0, i_word_end_bound - 1): i_word_end_bound + 2]
                word_bound_intervals.append((
                    (phone_start_bound[0], phone_start_bound[-1]),
                    (phone_end_bound[0], phone_end_bound[-1])
                    ))
            else:
                word_bound_intervals.append((
                    (max(0, word_start - tolerance), word_start + tolerance),
                    (word_end - tolerance, word_end + tolerance)
                    ))
        # print "word_bound_intervals", word_bound_intervals

        n_boundaries_seg += len(seg)
        n_boundaries_ref += len(word_bound_intervals)

        for seg_start, seg_end in seg:
            # print seg_start, seg_end
            for i_gt_word, (word_start_interval, word_end_interval) in enumerate(word_bound_intervals):
                word_start_lower, word_start_upper = word_start_interval
                word_end_lower, word_end_upper = word_end_interval

                if (word_start_lower <= seg_start <= word_start_upper and
                        word_end_lower <= seg_end <= word_end_upper):
                    n_boundaries_correct += 1
                    word_bound_intervals.pop(i_gt_word)  # can't re-use this token
                    # print "correct"
                    break
    # print "n_boundaries_correct", n_boundaries_correct
    # print "n_boundaries_seg", n_boundaries_seg
    # print "n_boundaries_ref", n_boundaries_ref
    return precision_recall_f(n_boundaries_correct, n_boundaries_seg, n_boundaries_ref)


def get_boundary_scores(gt_bounds, seg_bounds, tolerance):
    ref_boundaries = []
    seg_boundaries = []
    for utt in gt_bounds:
        ref = landmarks_to_intervals(gt_bounds[utt])
        seg = seg_bounds[utt]

        n = max(ref[-1][1], seg[-1][1])
        ref_bound = np.zeros(n, dtype=bool)
        seg_bound = np.zeros(n, dtype=bool)
        for i_bound_start, i_bound_end in ref:
            if i_bound_start > 0:
                ref_bound[i_bound_start - 1] = True
            ref_bound[i_bound_end - 1] = True
        for i_bound_start, i_bound_end in seg:
            if i_bound_start > 0:
                seg_bound[i_bound_start - 1] = True
            seg_bound[i_bound_end - 1] = True
        ref_bound[-1] = True
        seg_bound[-1] = True
        ref_boundaries.append(ref_bound)
        seg_boundaries.append(seg_bound)
    return score_boundaries(ref_boundaries, seg_boundaries, tolerance)


def segment_eval(pickle_fn, level, suppress_plot=False):

    print datetime.now()
    result_dict = {}

    # Read segmentation
    print "Reading:", pickle_fn
    with open(pickle_fn, "rb") as f:
        unsup_landmarks = pickle.load(f)
        unsup_transcript = pickle.load(f)
        am_init_record = pickle.load(f)
        segmenter_record = pickle.load(f)
        unsup_landmark_indices = pickle.load(f)

    d = path.split(pickle_fn)[0]
    data_dir = path.join("data", path.normpath(path.split(d)[0]).replace("models" + os.sep, ""))

    options_dict_fn = path.join(d, "options_dict.pkl")
    print "Reading:", options_dict_fn
    with open(options_dict_fn, "rb") as f:
        options_dict = pickle.load(f)
    print "Options:", options_dict

    if level == "phone":
        gt_fn = path.join(data_dir, "phone_gt.pkl")
        interval_mapping_func = intervals_to_seq
    elif level == "word":
        gt_fn = path.join(data_dir, "word_gt.pkl")
        interval_mapping_func = intervals_to_max_overlap
    else:
        assert False

    print "Reading:", gt_fn
    with open(gt_fn, "rb") as f:
        gt_bounds = pickle.load(f)
        gt_labels = pickle.load(f)

    analysis = analyze_segmentation(
        unsup_landmarks, unsup_transcript, gt_bounds, gt_labels, interval_mapping_func
        )
    one_to_one = analysis["one_to_one"]
    cluster_to_label_map = analysis["cluster_to_label_map"]
    cluster_to_label_map_many = analysis["cluster_to_label_map_many"]
    labels_true = analysis["labels_true"]
    labels_pred = analysis["labels_pred"]

    # Create unsupervised transcription using ground truth labels
    print "Creating mapped unsupervised transcription"
    unsup_transcript_mapped = {}
    unsup_transcript_mapped_many = {}
    for utt in unsup_transcript:
        unsup_transcript_mapped[utt] = [
            cluster_to_label_map[i].split(" ") if i in cluster_to_label_map else
            ["unk"] for i in unsup_transcript[utt]
            ]
        unsup_transcript_mapped_many[utt] = [
            cluster_to_label_map_many[i].split(" ") for i in unsup_transcript[utt] if i != -1
            ]

    unmapped_labels = list(set(labels_true).difference(set(cluster_to_label_map.values())))
    unmapped_clusters = list(set(labels_pred).difference(set(cluster_to_label_map.keys())))

    print "Calculating NED"
    clusters_ned_, ned_, n_ned_pairs = clusters_ned(labels_true, labels_pred)

    print datetime.now()
    print
    print "Clusters:"
    clusters = cluster_analysis.analyse_clusters(labels_true, labels_pred)

    # Additional cluster analysis
    gender_purity = 0
    speaker_purity = 0
    for cluster in clusters:
        i_cluster = cluster["id"]
        lengths = []
        genders = []
        speakers = []
        n_landmarks_crossed_cluster = []
        for utt in unsup_landmark_indices:
            if i_cluster in unsup_transcript[utt]:
                speaker = utt[10:14] if utt.startswith("nchlt-tso-") else utt[:3]
                for i_cut in np.where(unsup_transcript[utt] == i_cluster)[0]:
                    start, stop = unsup_landmarks[utt][i_cut]
                    lengths.append(float(stop - start)/100.0)
                    start, stop = unsup_landmark_indices[utt][i_cut]
                    n_landmarks_crossed_cluster.append(stop - start)
                    genders.append(
                        buckeye_speaker_genders[speaker] if speaker in
                        buckeye_speaker_genders else speaker[-1]
                        )  # Tsonga speakers has gender as last character in ID
                    speakers.append(speaker)
        cluster["lengths"] = lengths
        cluster["genders"] = genders
        cluster["speakers"] = speakers
        cluster["n_landmarks_crossed"] = n_landmarks_crossed_cluster
        gender_purity += max(Counter(genders).values())
        speaker_purity += max(Counter(speakers).values())
    gender_purity = float(gender_purity)/len(labels_pred)
    speaker_purity = float(speaker_purity)/len(labels_pred)

    # Print the biggest clusters
    n_biggest = 20
    n_tokens_covered = 0
    i_cluster_count = 1
    sizes, _ = cluster_model_present.get_sizes_purities(clusters)
    biggest_clusters = list(np.argsort(sizes)[-n_biggest:])  # http://stackoverflow.com/questions/16878715/how-to-find-the-index-of-n-largest-elements-in-a-list-or-np-array-python
    biggest_clusters.reverse()
    n_clusters_covered_90 = None
    print "-"*79
    for i_cluster in biggest_clusters:

        # Raw cluster statistics
        cluster = clusters[i_cluster]
        lengths = cluster["lengths"]
        genders = cluster["genders"]
        speakers = cluster["speakers"]
        n_landmarks_crossed_cluster = cluster["n_landmarks_crossed"]
        print "Cluster " + str(i_cluster) + " (rank: " + str(i_cluster_count) + ")"
        if i_cluster in cluster_to_label_map:
            print "Mapped to: '" + cluster_to_label_map[i_cluster] + "'"   
        if i_cluster in cluster_to_label_map_many:
            print "Many-mapped to: '" + cluster_to_label_map_many[i_cluster] + "'"   
        for label in cluster["counts"]:
            print '"' + label + '":', cluster["counts"][label]
        print "Size:", cluster["size"]
        print "Purity:", cluster["purity"]
        if i_cluster in clusters_ned_:
            print "Cluster NED:", clusters_ned_[i_cluster]

        print "Mean length:", np.mean(lengths)
        print "Std. length:", np.std(lengths)
        print "Mean landmarks crossed:", np.mean(n_landmarks_crossed_cluster)
        print "Std. landmarks crossed:", np.std(n_landmarks_crossed_cluster)
        print "Genders:", Counter(genders)
        print "Speakers:", Counter(speakers)

        # Tokens covered statistics
        n_tokens_covered += cluster["size"]
        prop_tokens_covered = n_tokens_covered*100./len(labels_pred)
        if n_clusters_covered_90 is None and prop_tokens_covered > 90.:
            n_clusters_covered_90 = i_cluster_count
        print "Tokens covered (%):", prop_tokens_covered
        print "-"*79

        i_cluster_count += 1

    # Get landmarks crossed for whole dataset
    n_landmarks_crossed = []
    for utt in unsup_landmark_indices:
        for start, stop in unsup_landmark_indices[utt]:
            n_landmarks_crossed.append(stop - start)

    # Singleton cluster analysis
    n_singleton_landmarks_crossed = []
    n_singleton_unmapped = 0
    n_singleton = 0
    for cluster in clusters:
        if cluster["size"] == 1:
            n_singleton += 1
            n_singleton_landmarks_crossed.extend(cluster["n_landmarks_crossed"])
            if not cluster["id"] in cluster_to_label_map:
                n_singleton_unmapped += 1

    print "No. clusters:", len(clusters)
    result_dict["n_clusters"] = len(clusters)
    print "No. unmapped clusters:", len(unmapped_clusters)
    result_dict["n_unmapped_clusters"] = len(unmapped_clusters)
    print "No. labels:", len(set(labels_true))
    print "No. unmapped labels:", len(unmapped_labels)
    print "Mean landmarks crossed:", np.mean(n_landmarks_crossed)
    result_dict["mean_n_landmarks_crossed"] = np.mean(n_landmarks_crossed)
    print "Std. landmarks crossed:", np.std(n_landmarks_crossed)
    print
    print "No. singleton clusters:", n_singleton
    result_dict["n_singleton"] = n_singleton
    if n_singleton > 0:
        print "No. unmapped singleton clusters:", n_singleton_unmapped
        result_dict["n_singleton_unmapped"] = n_singleton_unmapped
        print "Mean landmarks crossed singleton clusters:", np.mean(n_singleton_landmarks_crossed)
        result_dict["mean_n_singleton_landmarks_crossed"] = np.mean(n_singleton_landmarks_crossed)
        print "Std. landmarks crossed singleton clusters:", np.std(n_singleton_landmarks_crossed)
    print
    avg_purity = cluster_analysis.purity(labels_true, labels_pred)
    print "Clustering average purity:", avg_purity
    result_dict["avg_purity"] = avg_purity
    print "Clustering one-to-one accuracy:", one_to_one
    result_dict["one_to_one"] = one_to_one
    print "NED:", ned_, "(" + str(n_ned_pairs) + " pairs)"
    result_dict["ned"] = ned_
    result_dict["n_ned_pairs"] = n_ned_pairs
    print "Speaker purity:", speaker_purity
    result_dict["speaker_purity"] = speaker_purity
    print "Gender purity:", gender_purity
    result_dict["gender_purity"] = gender_purity
    print

    # Calculate unsupervised PER
    dp_error = dp_align.DPError()
    dp_error_many = dp_align.DPError()
    for utt in unsup_transcript_mapped:
        flattened_unsup = [val for sublist in unsup_transcript_mapped[utt] for val in sublist]
        cur_dp_error = dp_align.dp_align(gt_labels[utt], flattened_unsup)
        # print cur_dp_error
        # if cur_dp_error.n_ins > 0:
        #     print utt
        #     print unsup_transcript[utt]
        #     print flattened_unsup
        #     print gt_labels[utt]
        #     assert False
        dp_error = dp_error + cur_dp_error
        flattened_unsup_many = [val for sublist in unsup_transcript_mapped_many[utt] for val in sublist]
        cur_dp_error_many = dp_align.dp_align(gt_labels[utt], flattened_unsup_many)
        dp_error_many = dp_error_many + cur_dp_error_many
    print "No. utterances:", len(gt_labels)
    result_dict["n_utterances"] = len(gt_labels)
    print "No. tokens:", len(labels_pred)
    result_dict["n_tokens"] = len(labels_pred)
    print "Error counts:", dp_error
    result_dict["dp_error"] = {
        "n_del": dp_error.n_del,
        "n_ins": dp_error.n_ins,
        "n_sub": dp_error.n_sub,
        "n_match": dp_error.n_match,
        "n_total": dp_error.n_total,
        }
    print "Errors:", result_dict["dp_error"]
    er = dp_error.get_wer()
    if level == "phone":
        print "uPER:", er
    elif level == "word":
        print "uWER:", er
    result_dict["uWER"] = er
    print "Accuracy:", dp_error.get_accuracy()
    print "Error counts many-to-one:", dp_error_many
    result_dict["dp_error_many"] = {
        "n_del": dp_error_many.n_del,
        "n_ins": dp_error_many.n_ins,
        "n_sub": dp_error_many.n_sub,
        "n_match": dp_error_many.n_match,
        "n_total": dp_error_many.n_total,
        }
    er = dp_error_many.get_wer()
    if level == "phone":
        print "uPER_many:", er
    elif level == "word":
        print "uWER_many:", er
    result_dict["uWER_many"] = er
    print "Accuracy_many:", dp_error_many.get_accuracy()

    # Get model statistics
    total_time = 0
    if not am_init_record is None and len(am_init_record["log_marg"]) > 0:
        print 
        print "Initial acoustic model sampling final log P(X, z) =", am_init_record["log_marg"][-1]
        total_time += sum(am_init_record["sample_time"])
    if not segmenter_record is None and len(segmenter_record["log_prob_z"]) > 0:
        print
        print "Final log P(z) =", segmenter_record["log_prob_z"][-1]
        print "Final log P(X|z) =", segmenter_record["log_prob_X_given_z"][-1]
        print "Final log P(X, z) =", segmenter_record["log_marg"][-1]
        print "Final log [P(X, z)*length] =", segmenter_record["log_marg*length"][-1]
        result_dict["log_marg*length"] = segmenter_record["log_marg*length"][-1]
        total_time += sum(segmenter_record["sample_time"])
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)
    print "Total time:", "%d:%02d:%02d" % (h, m, s)

    # Calculate boundary scores
    # print
    # print datetime.now()
    # if level == "word":
    #     gt_phone_fn = path.join(data_dir, "phone_gt.pkl")
    #     with open(gt_phone_fn, "rb") as f:
    #         gt_phone_bounds = pickle.load(f)
    #         gt_phone_labels = pickle.load(f)
    #     n_boundaries_correct = 0
    #     n_boundaries_ref = 0
    #     n_boundaries_seg = 0
    #     for utt in gt_bounds:
    #         cur_gt_word_bounds = gt_bounds[utt][:-1]  # last boundary is always correct, shouldn't count
    #         cur_gt_phone_bounds = gt_phone_bounds[utt]
    #         seg = unsup_landmarks[utt][:-1]
    #         # print cur_gt_word_bounds
    #         # print cur_gt_phone_bounds
    #         # print [i[1] for i in seg]
    #         n_boundaries_seg += len(seg)
    #         n_boundaries_ref += len(cur_gt_word_bounds)
    #         word_bound_intervals = []
    #         for word_bound in cur_gt_word_bounds:
    #             i_word_bound = cur_gt_phone_bounds.index(word_bound)
    #             phone_bound = cur_gt_phone_bounds[max(0, i_word_bound - 1): i_word_bound + 2]
    #             word_bound_intervals.append((phone_bound[0], phone_bound[-1]))
    #             # word_bound_intervals.append((word_bound, word_bound))  # no tolerance
    #             # word_bound_intervals.append((word_bound - 2, word_bound + 2))  # 2 frame tolerance
    #         # print "word_bound_intervals", word_bound_intervals
    #         for seg_bound in [i[1] for i in seg]:
    #             for i_gt_bound, (start, end) in enumerate(word_bound_intervals):
    #                 if start <= seg_bound <= end:
    #                     n_boundaries_correct += 1
    #                     word_bound_intervals.pop(i_gt_bound)
    #                     break
    #     # print "n_boundaries_correct", n_boundaries_correct
    #     # print "n_boundaries_seg", n_boundaries_seg
    #     # print "n_boundaries_ref", n_boundaries_ref
    #     precision = float(n_boundaries_correct)/n_boundaries_seg
    #     recall = float(n_boundaries_correct)/n_boundaries_ref
    #     if precision + recall != 0:
    #         f = 2*precision*recall / (precision + recall)
    #     else:
    #         f = -np.inf
    #     print ("tolerance = one phone: P = " + str(precision) +
    #         ", R = " + str(recall) + ", F = " + str(f)
    #         )
    # ref_boundaries = []
    # seg_boundaries = []
    # for utt in gt_bounds:
    #     ref = landmarks_to_intervals(gt_bounds[utt])
    #     seg = unsup_landmarks[utt]

    #     n = max(ref[-1][1], seg[-1][1])
    #     ref_bound = np.zeros(n, dtype=bool)
    #     seg_bound = np.zeros(n, dtype=bool)
    #     for i_bound_start, i_bound_end in ref:
    #         if i_bound_start > 0:
    #             ref_bound[i_bound_start - 1] = True
    #         ref_bound[i_bound_end - 1] = True
    #     for i_bound_start, i_bound_end in seg:
    #         if i_bound_start > 0:
    #             seg_bound[i_bound_start - 1] = True
    #         seg_bound[i_bound_end - 1] = True
    #     ref_bound[-1] = True
    #     seg_bound[-1] = True
    #     ref_boundaries.append(ref_bound)
    #     seg_boundaries.append(seg_bound)
    # for tolerance in [0, 2, 4]:
    #     precision, recall, f = score_boundaries(ref_boundaries, seg_boundaries, tolerance)
    #     print (
    #         "tolerance = " + str(tolerance) + ": P = " + str(precision) +
    #         ", R = " + str(recall) + ", F = " + str(f)
    #         )
    # print datetime.now()

    # Calculate boundary scores
    print
    print datetime.now()
    if level == "word":
        print "Word boundaries:"
        gt_phone_fn = path.join(data_dir, "phone_gt.pkl")
        with open(gt_phone_fn, "rb") as f:
            gt_phone_bounds = pickle.load(f)
            gt_phone_labels = pickle.load(f)
        precision, recall, f = get_word_boundary_scores(gt_phone_bounds, gt_bounds, unsup_landmarks)
        print ("tolerance = one phone: P = " + str(precision) +
            ", R = " + str(recall) + ", F = " + str(f)
            )
        result_dict["word_bounds_onephone"] = (precision, recall, f)
    for tolerance in [0, 2, 4]:
        precision, recall, f = get_boundary_scores(gt_bounds, unsup_landmarks, tolerance)
        print (
            "tolerance = " + str(tolerance) + ": P = " + str(precision) +
            ", R = " + str(recall) + ", F = " + str(f)
            )
        if tolerance == 2:
            result_dict["word_bounds_2"] = (precision, recall, f)
    if level == "word":
        print "Token scores:"
        precision, recall, f = get_word_token_scores(gt_phone_bounds, gt_bounds, unsup_landmarks)
        print ("tolerance = one phone: P = " + str(precision) +
            ", R = " + str(recall) + ", F = " + str(f)
            )
        result_dict["word_token_onephone"] = (precision, recall, f)
        for tolerance in [0, 2, 4]:
            precision, recall, f = get_word_token_scores(
                gt_phone_bounds, gt_bounds, unsup_landmarks, tolerance=tolerance
                )
            print (
                "tolerance = " + str(tolerance) + ": P = " + str(precision) +
                ", R = " + str(recall) + ", F = " + str(f)
                )
            if tolerance == 2:
                result_dict["word_token_2"] = (precision, recall, f)

        # Also print phone boundary scores when evaluating words
        print "Phone boundaries:"
        for tolerance in [0, 2, 4]:
            precision, recall, f = get_boundary_scores(gt_phone_bounds, unsup_landmarks, tolerance)
            print (
                "tolerance = " + str(tolerance) + ": P = " + str(precision) +
                ", R = " + str(recall) + ", F = " + str(f)
                )
            if tolerance == 2:
                result_dict["phone_bounds_2"] = (precision, recall, f)
    print datetime.now()

    if not segmenter_record is None and not suppress_plot:
        cluster_model_present.plot_log_marg_components(segmenter_record)

    # Frequency and counts for Zipf plot
    if not suppress_plot:
        counter = Counter(labels_pred)
        most_common = counter.most_common()
        frequency = [i[1] for i in most_common]
        rank = np.arange(1.0, len(frequency) + 1.0)
        plt.figure()
        ax = plt.subplot(111)
        ax.scatter(rank, frequency, edgecolors="none", marker="o", s=12)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("Rank of cluster")
        ax.set_ylabel("Count of cluster")
        plt.grid()
        plt.gca().set_axisbelow(True)

    if not suppress_plot:
        plt.show()

    return result_dict


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    result_dict = segment_eval(args.segment_fn, args.level)

    if args.write_dict:
        result_dict_fn = path.join(path.split(args.segment_fn)[0], "segment_eval_dict.pkl")
        print("Writing: " + result_dict_fn)
        with open(result_dict_fn, "wb") as f:
            pickle.dump(result_dict, f, -1)


if __name__ == "__main__":
    main()
