#!/usr/bin/env python

"""
Unsupervised unigram acoustic word segmentation of Buckeye and NCHLT Tsonga.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015-2016
"""

from datetime import datetime
from os import path
import argparse
import cPickle as pickle
import hashlib
import logging
import numpy as np
import os
import random
import sys

sys.path.append(path.join("..", "src", "segmentalist"))

from segmentalist import fbgmm
from segmentalist import gaussian_components_fixedvar
from segmentalist import unigram_acoustic_wordseg
import segment_eval

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                           DEFAULT TRAINING OPTIONS                          #
#-----------------------------------------------------------------------------#

default_options_dict = {
    "data_dir": None, #"data/devpart1/mfcc.unsup_syl",
    "model_dir": "models",
    "am_alpha": 1.0,
    "am_K": "0.2landmarks",  # can either be an absolute number or "0.2landmarks", indicating a proportion
    "m_0": "zero",
    "k_0": 0.05,
    "S_0_scale": 0.001,  # 0.001
    "init_am_n_iter": 15,  # initial iterations of acoustic model sampling; can also be "kmeans"
    "init_am_assignments": "rand",  # "rand" or "one-by-one"
    "p_boundary_init": 0.25,  # probability of initial boundaries
    "beta_sent_boundary": -1.,
    "lms": 1.,  # language model scaling factor
    "wip": 0.,  # word insertion penalty
    "n_slices_min": 0,
    "n_slices_max": 6,
    "min_duration": 0,  # minimum number of frames in segment
    "covariance_type": "fixed",     # can be "fixed", "diag" or "full" (if last
                                    # two, then additional hyper-parameters are
                                    # required)
    "fb_type": "standard",  # "standard" or "viterbi"
    "segment_n_iter": 15,
    "anneal_schedule": "step",  # None, "linear" or "step"
    "anneal_start_temp_inv": 0.01,  # some start at 0.1 (see Goldwater PhD, 2007, p. 70)
    "anneal_end_temp_inv": 1.,  # standard to end at 1.0 (see Goldwater PhD, 2007, p. 70)
    "n_anneal_steps": 3,    # if -1, then use all `segment_n_iter` iterations
                            # for annealing; if "step" annealing, then this is
                            # the number of steps
    "anneal_gibbs_am": False,  # whether the inside Gibbs loop of the acoustic model should be annealed
    "seed_bounds": None,  # filename of pickle of seed boundary dict
    "seed_assignments": None,  # filename of pickle of seed assignments dict
    "time_power_term": 1,   # with 1.2 instead of 1, we get less words (prefer longer words)
    "rnd_seed": 42,
    "log_to_file": False,
    }


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])#, add_help=False)
    parser.add_argument(
        "--data_dir", type=str, help="data directory (default: %(default)s)",
        default=default_options_dict["data_dir"]
        )
    parser.add_argument(
        "--am_K", type=str, help="default: %(default)s",
        default=default_options_dict["am_K"]
        )
    parser.add_argument(
        "--S_0_scale", type=float, help="default: %(default)s",
        default=default_options_dict["S_0_scale"]
        )
    parser.add_argument(
        "--wip", type=float, help="default: %(default)s",
        default=default_options_dict["wip"]
        )
    parser.add_argument(
        "--p_boundary_init", type=float, help="default: %(default)s",
        default=default_options_dict["p_boundary_init"]
        )    
    parser.add_argument(
        "--lms", type=float, help="default: %(default)s",
        default=default_options_dict["lms"]
        )
    parser.add_argument(
        "--time_power_term", type=float, help="default: %(default)s",
        default=default_options_dict["time_power_term"]
        )   
    parser.add_argument(
        "--rnd_seed", type=int, help="default: %(default)s",
        default=default_options_dict["rnd_seed"]
        )
    parser.add_argument(
        "--min_duration", type=int, help="default: %(default)s",
        default=default_options_dict["min_duration"]
        )
    parser.add_argument(
        "--segment_n_iter", type=int, help="default: %(default)s",
        default=default_options_dict["segment_n_iter"]
        )
    parser.add_argument(
        "--log_to_file", help="whether the output should be logged to file", action="store_true"
        )
    parser.add_argument(
        "--eval", help="evaluate output from segmentation", action="store_true"
        )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


def eval_pickle(pickle_fn):
    logger.info("Evaluating: " + pickle_fn)
    output_fn = path.join(path.split(pickle_fn)[0], "segment_eval.txt")
    logger.info("Writing: " + output_fn)
    with open(output_fn, "w") as f:
        sys.stdout = f
        result_dict = segment_eval.segment_eval(pickle_fn, "word", suppress_plot=True)
    sys.stdout = sys.__stdout__
    result_dict_fn = path.join(path.split(pickle_fn)[0], "segment_eval_dict.pkl")
    logger.info("Writing: " + result_dict_fn)
    with open(result_dict_fn, "wb") as f:
        pickle.dump(result_dict, f, -1)


#-----------------------------------------------------------------------------#
#                            SEGMENTATION FUNCTIONS                           #
#-----------------------------------------------------------------------------#

def segment(options_dict, suppress_pickling=False):
    """
    Segment and save the results of unigram aoustic word segmentation.

    The `suppress_pickling` parameter is useful for initializing a model from
    previous output.
    """

    logger.info(datetime.now())

    random.seed(options_dict["rnd_seed"])
    np.random.seed(options_dict["rnd_seed"])

    # Set output pickle filename
    hasher = hashlib.md5(repr(sorted(options_dict.items())).encode("ascii"))
    hash_str = hasher.hexdigest()[:10]
    model_dir = path.join(options_dict["model_dir"], hash_str)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    options_dict_fn = path.join(model_dir, "options_dict.pkl")
    if not suppress_pickling:
        if "log_to_file" in options_dict and options_dict["log_to_file"]:
            log_fn = path.join(model_dir, "log")
            print "Log:", log_fn
            with open(log_fn, "w") as f:
                f.write("")
            logging.basicConfig(filename=log_fn, level=logging.INFO)

        logger.info("Writing: " + options_dict_fn)
        with open(options_dict_fn, "wb") as f:
            pickle.dump(options_dict, f, -1)
    logger.info("Options: " + str(options_dict))

    logger.info("Reading from directory: " + options_dict["data_dir"])
    landmarks_dict_fn = path.join(options_dict["data_dir"], "landmarks.pkl")
    dense_embeddings_fn = path.join(options_dict["data_dir"], "dense_embeddings.npz")
    vec_ids_dict_fn = path.join(options_dict["data_dir"], "vec_ids.pkl")
    durations_dict_fn = path.join(options_dict["data_dir"], "durations.pkl")
    word_gt_dict_fn = path.join(options_dict["data_dir"], "word_gt.pkl")
    # phone_gt_dict_fn = path.join(data_dir, "phone_gt.pkl")
    with open(landmarks_dict_fn, "rb") as f:
        landmarks_dict = pickle.load(f)
    dense_embeddings = dict(np.load(dense_embeddings_fn))
    with open(vec_ids_dict_fn, "rb") as f:
        vec_ids_dict = pickle.load(f)
    with open(durations_dict_fn, "rb") as f:
        durations_dict = pickle.load(f)
    with open(word_gt_dict_fn, "rb") as f:
        word_gt_bound_dict = pickle.load(f)
        word_gt_label_dict = pickle.load(f)
    logger.info("No. of utterances: " + str(len(landmarks_dict)))

    n_landmarks = sum([len(i) for i in landmarks_dict.values()])
    logger.info("No. of landmarks: " + str(n_landmarks))
    if "landmarks" in str(options_dict["am_K"]):
        # The number of components are set as a proportion of landmarks
        proportion = float(options_dict["am_K"].replace("landmarks", ""))
        am_K = int(np.floor(proportion * n_landmarks))
    else:
        am_K = int(options_dict["am_K"])
    logger.info("am_K: " + str(am_K))

    D = dense_embeddings[dense_embeddings.keys()[0]].shape[1]
    logger.info("Embedding dimensionality: " + str(D))

    logger.info(datetime.now())
    logger.info("Normalizing embeddings")
    n_embeds = 0
    for utt in dense_embeddings:
        for i in range(dense_embeddings[utt].shape[0]):
            n_embeds += 1
            cur_embed = dense_embeddings[utt][i, :]
            norm = np.linalg.norm(cur_embed)
            assert norm != 0.
            dense_embeddings[utt][i, :] = cur_embed / np.linalg.norm(cur_embed)
    logger.info("No. of embeddings: " + str(n_embeds))

    try:
        # Try to open seed values as pickled files

        # Seeding labels requires boundaries to also be seeded
        if options_dict["seed_bounds"] is not None or options_dict["seed_assignments"] is not None:
                bounds_pkl_fn = (
                    options_dict["seed_bounds"] if options_dict["seed_bounds"] is not
                    None else options_dict["seed_assignments"]
                    )
                logger.info("Reading: " + bounds_pkl_fn)
                with open(bounds_pkl_fn, "rb") as f:
                    seed_bounds = pickle.load(f)
            # options_dict["p_boundary_init"] = -1
        else:
            seed_bounds = None

        if options_dict["seed_assignments"] is not None:
            logger.info("Reading: " + options_dict["seed_assignments"])
            with open(options_dict["seed_assignments"], "rb") as f:
                seed_bounds = pickle.load(f)
                seed_assignments = pickle.load(f)
        else:
            seed_assignments = None

    except TypeError:
        # Dictionaries are provided, not filenames of pickles
        seed_bounds = options_dict["seed_bounds"]
        seed_assignments = options_dict["seed_assignments"]

        # Seeding labels requires boundaries to also be seeded
        assert seed_assignments is None or seed_bounds is not None  

    # print seed_bounds[seed_bounds.keys()[0]]
    # print seed_assignments[seed_assignments.keys()[0]]
    # assert False

    # Setup model hyper-parameters
    logger.info("Setting up model")
    am_class = fbgmm.FBGMM
    if options_dict["m_0"] == "zero":
        m_0 = np.zeros(D)
    else:
        assert False, "invalid `m_0`"
    if options_dict["covariance_type"] == "full":
        assert False, "to-do"
        S_0 = options_dict["S_0_scale"]*np.eye(D) * (options_dict["v_0"] - D - 1.0)
        am_param_prior = niw.NIW(m_0, k_0, v_0, S_0)
    elif options_dict["covariance_type"] == "diag":
        assert False, "to-do"
        S_0 = options_dict["S_0_scale"]*np.ones(D) * (options_dict["v_0"] - 2)
        am_param_prior = niw.NIW(m_0, options_dict["k_0"], options_dict["v_0"], S_0)
    elif options_dict["covariance_type"] == "fixed":
        S_0 = options_dict["S_0_scale"]*np.ones(D)
        am_param_prior = gaussian_components_fixedvar.FixedVarPrior(S_0, m_0, S_0/options_dict["k_0"])
    else:
        assert False, "invalid `covariance_type`"

    # Setup model
    segmenter = unigram_acoustic_wordseg.UnigramAcousticWordseg(
        am_class, options_dict["am_alpha"], am_K,
        am_param_prior, dense_embeddings, vec_ids_dict, durations_dict,
        landmarks_dict, seed_boundaries_dict=seed_bounds,
        seed_assignments_dict=seed_assignments,
        covariance_type=options_dict["covariance_type"],
        p_boundary_init=options_dict["p_boundary_init"],
        beta_sent_boundary=options_dict["beta_sent_boundary"],
        lms=options_dict["lms"], wip=options_dict["wip"],
        fb_type=options_dict["fb_type"],
        n_slices_min=options_dict["n_slices_min"],
        n_slices_max=options_dict["n_slices_max"],
        min_duration=options_dict["min_duration"],
        init_am_assignments=options_dict["init_am_assignments"],
        time_power_term=options_dict["time_power_term"]
        )

    # Write mapping if seed was provided
    try:
        seedlabel_to_cluster = segmenter.seed_to_cluster
    except AttributeError:
        seedlabel_to_cluster = None
    if not suppress_pickling and seedlabel_to_cluster:
        mapping_fn = path.join(model_dir, "seedlabel_to_cluster.pkl")
        logger.info("Writing: " + mapping_fn)
        with open(mapping_fn, "wb") as f:
            pickle.dump(seedlabel_to_cluster, f, -1)

    # Initialize acoustic model by training for a few iterations
    if options_dict["init_am_n_iter"] == "kmeans":

        from sklearn.cluster import KMeans

        logger.info(datetime.now())
        logger.info("Performing initial acoustic model iterations using k-means clustering")
        indices = np.where(segmenter.acoustic_model.components.assignments != -1)[0]
        embeddings = segmenter.acoustic_model.components.X[indices, :]
        kmeans = KMeans(n_clusters=am_K, n_init=1)
        kmeans.fit(embeddings)
        labels_pred = kmeans.labels_
        for i_embed in indices:
            segmenter.acoustic_model.components.del_item(i_embed)
        for i in np.argsort(labels_pred):
            i_embed = indices[i]
            assignment = labels_pred[i]
            segmenter.acoustic_model.components.add_item(i_embed, assignment)
        # for i_embed, assignment in zip(indices, labels_pred):
        #     segmenter.acoustic_model.components.del_item()
        #     print i_embed, assignment
        # print embeddings.shape
        logger.info(datetime.now())

    elif options_dict["init_am_n_iter"] > 0:
        logger.info("Performing initial acoustic model iterations")
        am_init_record = segmenter.acoustic_model.gibbs_sample(
            options_dict["init_am_n_iter"], consider_unassigned=False,
            anneal_schedule=None
            )

    # Perform segmentation
    if options_dict["segment_n_iter"] > 0:
        segmenter_record = segmenter.gibbs_sample(
            options_dict["segment_n_iter"],
            anneal_schedule=options_dict["anneal_schedule"],
            anneal_start_temp_inv=options_dict["anneal_start_temp_inv"],
            anneal_end_temp_inv=options_dict["anneal_end_temp_inv"],
            n_anneal_steps=options_dict["n_anneal_steps"],
            anneal_gibbs_am=options_dict["anneal_gibbs_am"]
            )

    # Obtain clusters and landmarks (frame indices)
    unsup_transcript = {}
    unsup_landmarks = {}
    unsup_landmark_indices = {}
    for i_utt in xrange(segmenter.utterances.D):
        utt = segmenter.ids_to_utterance_labels[i_utt]
        unsup_transcript[utt] = segmenter.get_unsup_transcript_i(i_utt)
        if -1 in unsup_transcript[utt]:
            logger.warning(
                "Unassigned cuts in: " + utt + " (transcript: " + str(unsup_transcript[utt]) + ")"
                )
        unsup_landmarks[utt] = segmenter.utterances.get_segmented_landmarks(i_utt)
        unsup_landmark_indices[utt] = segmenter.utterances.get_segmented_landmark_indices(i_utt)

    # Write output
    if not suppress_pickling:
        output_fn = path.join(model_dir, "segment.pkl")
        logger.info("Writing: " + output_fn)
        with open(output_fn, "wb") as f:
            pickle.dump(unsup_landmarks, f, -1)
            pickle.dump(unsup_transcript, f, -1)
            if options_dict["init_am_n_iter"] != "kmeans" and options_dict["init_am_n_iter"] > 0:
                pickle.dump(am_init_record, f, -1)
            else:
                pickle.dump(None, f, -1)
            if options_dict["segment_n_iter"] > 0:
                pickle.dump(segmenter_record, f, -1)
            else:
                pickle.dump(None, f, -1)
            pickle.dump(unsup_landmark_indices, f, -1)
        logger.info(datetime.now())
        return output_fn

    logger.info(datetime.now())
    return segmenter


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    # Set options
    options_dict = default_options_dict.copy()
    options_dict["data_dir"] = path.normpath(args.data_dir)
    options_dict["am_K"] = args.am_K
    options_dict["S_0_scale"] = args.S_0_scale
    options_dict["wip"] = args.wip
    options_dict["lms"] = args.lms
    options_dict["p_boundary_init"] = args.p_boundary_init
    options_dict["time_power_term"] = args.time_power_term
    options_dict["rnd_seed"] = args.rnd_seed
    options_dict["min_duration"] = args.min_duration
    options_dict["segment_n_iter"] = args.segment_n_iter
    options_dict["log_to_file"] = args.log_to_file
    options_dict["model_dir"] = path.join(
        options_dict["model_dir"], options_dict["data_dir"].replace("data" + os.sep, "")
        )

    if "log_to_file" not in options_dict or not options_dict["log_to_file"]:
        logging.basicConfig(level=logging.INFO)

    # Temp
    # options_dict["am_K"] = None
    # options_dict["seed_bounds"] = path.join(options_dict["data_dir"], "word_gt.pkl")
    # options_dict["seed_assignments"] = path.join(options_dict["data_dir"], "word_gt.pkl")

    pickle_fn = segment(options_dict)

    if args.eval:
        eval_pickle(pickle_fn)
        logger.info(datetime.now())


if __name__ == "__main__":
    main()
