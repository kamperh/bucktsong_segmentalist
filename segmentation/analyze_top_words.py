#!/usr/bin/env python

"""
Analyze the most frequent non-function words and their corresponding clusters.

There is much overlap between this script and `segment_eval.py`.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from collections import Counter
from datetime import datetime
from os import path
import argparse
import cPickle as pickle
import numpy as np
import os
import sys

from segment_eval import (intervals_to_max_overlap, analyze_segmentation,
    buckeye_speaker_genders, clusters_ned)
from utils import cluster_analysis, cluster_model_present


# Obtained from:
# http://www.sequencepublishing.com/academic.html
# http://www.sequencepublishing.com/cgi-bin/download.cgi?efw
# cat *.txt | grep -v "^\/" | grep -v "\%" | sed 's/ /\n/g' | sed -e's/\,//g' \
# -e 's/\.//g' | sort -u
# function_words = [
#     "a", "able", "aboard", "about", "above", "absent", "according",
#     "accordingly", "across", "after", "against", "ahead", "albeit", "all",
#     "along", "alongside", "although", "amid", "amidst", "among", "amongst",
#     "amount", "an", "and", "another", "anti", "any", "anybody", "anyone",
#     "anything", "around", "as", "aside", "astraddle", "astride", "at", "away",
#     "bar", "barring", "be", "because", "before", "behind", "below", "beneath",
#     "beside", "besides", "better", "between", "beyond", "bit", "both", "but",
#     "by", "can", "certain", "circa", "close", "concerning", "consequently",
#     "considering", "could", "couple", "dare", "deal", "despite", "down", "due",
#     "during", "each", "either", "enough", "etc", "every", "everybody",
#     "everyone", "everything", "except", "excepting", "excluding", "failing",
#     "few", "fewer", "fifth", "following", "for", "four", "fourth", "from",
#     "front", "given", "good", "great", "had", "half", "have", "he", "heaps",
#     "hence", "her", "hers", "herself", "him", "himself", "his", "however", "i",
#     "if", "in", "including", "inside", "instead", "into", "it", "its",
#     "itself", "keeping", "lack", "less", "like", "little", "loads", "lots",
#     "majority", "many", "masses", "may", "me", "might", "mine", "minority",
#     "minus", "more", "most", "much", "must", "my", "myself", "near", "need",
#     "neither", "nevertheless", "next", "no", "no_one", "nobody", "none", "nor",
#     "nothing", "notwithstanding", "number", "numbers", "of", "off", "on",
#     "once", "one", "onto", "opposite", "or", "other", "ought", "our", "ours",
#     "ourselves", "out", "outside", "over", "part", "past", "pending", "per",
#     "pertaining", "place", "plenty", "plethora", "plus", "quantities",
#     "quantity", "quarter", "regarding", "remainder", "respecting", "rest",
#     "round", "save", "saving", "several", "shall", "she", "should", "similar",
#     "since", "so", "some", "somebody", "someone", "something", "spite", "such",
#     "than", "thanks", "that", "the", "their", "theirs", "them", "themselves",
#     "then", "thence", "therefore", "these", "they", "third", "this", "tho'",
#     "those", "though", "three", "through", "throughout", "thru", "thus",
#     "till", "time", "to", "tons", "top", "toward", "towards", "two", "under",
#     "underneath", "unless", "unlike", "until", "unto", "up", "upon", "us",
#     "used", "various", "versus", "via", "view", "wanting", "we", "what",
#     "whatever", "when", "whence", "whenever", "where", "whereas", "wherever",
#     "whether", "which", "whichever", "while", "whilst", "who", "whoever",
#     "whole", "whom", "whomever", "whose", "will", "with", "within", "without",
#     "would", "yet", "you", "your", "yours", "yourself", "yourselves"
#     ]

# Obtained from:
# http://myweb.tiscali.co.uk/wordscape/museum/funcword.html
function_words = [
    "a", "about", "above", "after", "again", "ago", "all", "almost", "along",
    "already", "also", "although", "always", "am", "among", "an", "and",
    "another", "any", "anybody", "anything", "anywhere", "are", "arent",
    "around", "as", "at", "back", "else", "be", "been", "before", "being",
    "below", "beneath", "beside", "between", "beyond", "billion", "billionth",
    "both", "each", "but", "by", "can", "cant", "could", "couldnt", "did",
    "didnt", "do", "does", "doesnt", "doing", "done", "dont", "down", "during",
    "eight", "eighteen", "eighteenth", "eighth", "eightieth", "eighty",
    "either", "eleven", "eleventh", "enough", "even", "ever", "every",
    "everybody", "everyone", "everything", "everywhere", "except", "far",
    "few", "fewer", "fifteen", "fifteenth", "fifth", "fiftieth", "fifty",
    "first", "five", "for", "fortieth", "forty", "four", "fourteen",
    "fourteenth", "fourth", "hundred", "from", "get", "gets", "getting", "got",
    "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed",
    "hell", "hence", "her", "here", "hers", "herself", "hes", "him", "himself",
    "his", "hither", "how", "however", "near", "hundredth", "i", "id", "if",
    "ill", "im", "in", "into", "is", "ive", "isnt", "it", "its", "itself",
    "just", "last", "less", "many", "me", "may", "might", "million",
    "millionth", "mine", "more", "most", "much", "must", "mustnt", "my",
    "myself", "nearby", "nearly", "neither", "never", "next", "nine",
    "nineteen", "nineteenth", "ninetieth", "ninety", "ninth", "no", "nobody",
    "none", "noone", "nothing", "nor", "not", "now", "nowhere", "of", "off",
    "often", "on", "or", "once", "one", "only", "other", "others", "ought",
    "oughtnt", "our", "ours", "ourselves", "out", "over", "quite", "rather",
    "round", "second", "seven", "seventeen", "seventeenth", "seventh",
    "seventieth", "seventy", "shall", "shant", "shed", "she", "shell", "shes",
    "should", "shouldnt", "since", "six", "sixteen", "sixteenth", "sixth",
    "sixtieth", "sixty", "so", "some", "somebody", "someone", "something",
    "sometimes", "somewhere", "soon", "still", "such", "ten", "tenth", "than",
    "that", "thats", "the", "their", "theirs", "them", "themselves", "these",
    "then", "thence", "there", "therefore", "they", "theyd", "theyll",
    "theyre", "third", "thirteen", "thirteenth", "thirtieth", "thirty", "this",
    "thither", "those", "though", "thousand", "thousandth", "three", "thrice",
    "through", "thus", "till", "to", "towards", "today", "tomorrow", "too",
    "twelfth", "twelve", "twentieth", "twenty", "twice", "two", "under",
    "underneath", "unless", "until", "up", "us", "very", "when", "was",
    "wasnt", "we", "wed", "well", "were", "werent", "weve", "what", "whence",
    "where", "whereas", "which", "while", "whither", "who", "whom", "whose",
    "why", "will", "with", "within", "without", "wont", "would", "wouldnt",
    "yes", "yesterday", "yet", "you", "your", "youd", "youll", "youre",
    "yours", "yourself", "yourselves", "youve"
    ]

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


def analyze_most_frequent(pickle_fn):

    print datetime.now()

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

    gt_fn = path.join(data_dir, "word_gt.pkl")
    interval_mapping_func = intervals_to_max_overlap
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

    # Analyze ground truth labels
    print
    print "Ground truth labels:"
    gt_labels_list = []
    for key in gt_labels:
        gt_labels_list.extend(gt_labels[key])
    gt_labels_count = Counter(gt_labels_list)
    n_types = 0
    n_tokens = 0
    n_function_types = 0
    n_function_tokens = 0
    most_common_non_func = Counter()
    for label, count in gt_labels_count.most_common():
        n_types += 1
        n_tokens += count
        if label in function_words:
            n_function_types += 1
            n_function_tokens += count
        else:
            most_common_non_func[label] = count
    print "No. types:", n_types
    print "No. tokens:", n_tokens
    print "No. function types:", n_function_types
    print "No. function tokens:", n_function_tokens
    print "30 most common non-function words:", most_common_non_func.most_common(30)

    clusters = cluster_analysis.analyse_clusters(labels_true, labels_pred)
    sizes, _ = cluster_model_present.get_sizes_purities(clusters)
    biggest_clusters = list(np.argsort(sizes))
    biggest_clusters.reverse()

    print 
    print "Clusters mapped to most common non-function words:"
    print "-"*79
    for gt_label, gt_count in most_common_non_func.most_common(30):
        print "'" + gt_label + "',", "count:", gt_count
        n_tokens = 0
        for i_cluster in biggest_clusters:
            if cluster_to_label_map_many[i_cluster] == gt_label:
                cluster = clusters[i_cluster]
                print (
                    "cluster: " + str(i_cluster) +
                    ", rank: " + str(biggest_clusters.index(i_cluster)) +
                    ", purity: " + str(cluster["purity"]) +
                    ", size: " + str(cluster["size"]) +
                    ", no. target tokens: " + str(cluster["counts"][gt_label])
                    )
                n_tokens += cluster["counts"][gt_label]
        print "No. mapped tokens:", n_tokens
        print "-"*79

        # true count, cluster id, cluster rank, cluster purity, number of target tokens in cluster

    # print
    # print "Clusters:"
    # clusters_ned_, ned_, n_ned_pairs = clusters_ned(labels_true, labels_pred)
    # clusters = cluster_analysis.analyse_clusters(labels_true, labels_pred)

    # # Additional cluster analysis
    # gender_purity = 0
    # speaker_purity = 0
    # for cluster in clusters:
    #     i_cluster = cluster["id"]
    #     lengths = []
    #     genders = []
    #     speakers = []
    #     n_landmarks_crossed_cluster = []
    #     for utt in unsup_landmark_indices:
    #         if i_cluster in unsup_transcript[utt]:
    #             speaker = utt[10:14] if utt.startswith("nchlt-tso-") else utt[:3]
    #             for i_cut in np.where(unsup_transcript[utt] == i_cluster)[0]:
    #                 start, stop = unsup_landmarks[utt][i_cut]
    #                 lengths.append(float(stop - start)/100.0)
    #                 start, stop = unsup_landmark_indices[utt][i_cut]
    #                 n_landmarks_crossed_cluster.append(stop - start)
    #                 genders.append(
    #                     buckeye_speaker_genders[speaker] if speaker in
    #                     buckeye_speaker_genders else speaker[-1]
    #                     )  # Tsonga speakers has gender as last character in ID
    #                 speakers.append(speaker)
    #     cluster["lengths"] = lengths
    #     cluster["genders"] = genders
    #     cluster["speakers"] = speakers
    #     cluster["n_landmarks_crossed"] = n_landmarks_crossed_cluster
    #     gender_purity += max(Counter(genders).values())
    #     speaker_purity += max(Counter(speakers).values())
    # gender_purity = float(gender_purity)/len(labels_pred)
    # speaker_purity = float(speaker_purity)/len(labels_pred)

    # # Print the biggest clusters
    # i_cluster_count = 1
    # sizes, _ = cluster_model_present.get_sizes_purities(clusters)
    # biggest_clusters = list(np.argsort(sizes))
    # biggest_clusters.reverse()
    # print "-"*79
    # for i_cluster in biggest_clusters:

    #     if (i_cluster not in cluster_to_label_map_many or
    #             cluster_to_label_map_many[i_cluster] in function_words):
    #         continue

    #     # Raw cluster statistics
    #     cluster = clusters[i_cluster]
    #     lengths = cluster["lengths"]
    #     genders = cluster["genders"]
    #     speakers = cluster["speakers"]
    #     n_landmarks_crossed_cluster = cluster["n_landmarks_crossed"]
    #     print "Cluster " + str(i_cluster) + " (rank: " + str(i_cluster_count) + ")"
    #     if i_cluster in cluster_to_label_map:
    #         print "Mapped to: '" + cluster_to_label_map[i_cluster] + "'"   
    #     if i_cluster in cluster_to_label_map_many:
    #         print "Many-mapped to: '" + cluster_to_label_map_many[i_cluster] + "'"   
    #     for label in cluster["counts"]:
    #         print '"' + label + '":', cluster["counts"][label]
    #     print "Size:", cluster["size"]
    #     print "Purity:", cluster["purity"]
    #     if i_cluster in clusters_ned_:
    #         print "Cluster NED:", clusters_ned_[i_cluster]

    #     print "Mean length:", np.mean(lengths)
    #     print "Std. length:", np.std(lengths)
    #     print "Mean landmarks crossed:", np.mean(n_landmarks_crossed_cluster)
    #     print "Std. landmarks crossed:", np.std(n_landmarks_crossed_cluster)
    #     print "Genders:", Counter(genders)
    #     print "Speakers:", Counter(speakers)

    #     # Tokens covered statistics
    #     print "-"*79

    #     i_cluster_count += 1

    #     if i_cluster_count == 20:
    #         break

    # for i in cluster_to_label_map_many:
    #     if cluster_to_label_map_many[i] == "yknow":
    #         print i, clusters[i]["size"]


    # print Counter(gt_labels_list)

    # unmapped_labels = list(set(labels_true).difference(set(cluster_to_label_map.values())))
    # unmapped_clusters = list(set(labels_pred).difference(set(cluster_to_label_map.keys())))

    print datetime.now()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    analyze_most_frequent(args.segment_fn)


if __name__ == "__main__":
    main()
