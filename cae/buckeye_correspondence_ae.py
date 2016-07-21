#!/usr/bin/env python

"""
Train a stacked dAE on the Buckeye corpus.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2015
"""

from os import path
import argparse
import sys
sys.path.append("../src/speech_correspondence/speech_correspondence")
import train_correspondence_ae


# Get dataset
def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("dataset", type=str, help="dataset label", choices=["zs", "devpart1", "tsonga"])
    parser.add_argument("--utd_tag", type=str, help="utd_label")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

args = check_argv()
dataset = args.dataset

dataset_basename = path.join(dataset, dataset + "_utd")
if args.utd_tag:
    dataset_basename += "_" + args.utd_tag

# Network parameter settings
parameter_dict = {
    "dataset_npy_fn_x": "../features/wordpair_aligns/" + dataset_basename + ".word1.npy",
    "dataset_npy_fn_y": "../features/wordpair_aligns/" + dataset_basename + ".word2.npy",
    "models_basedir": "models/" + dataset + "/",
    "dim_input": 39,
    "layer_spec_str": "[100] * 9",  # "layer_spec_str": "100,100,100,100,100,100,100,13,100",
    "dae_corruption": 0,  # these dae parameters specify which pretrained model to use
    "dae_max_epochs": 5,
    "max_epochs": 120,  # 120
    "batch_size": 2048,
    "learning_rate": 0.032,
    "start_from_scratch": False,  # do not initialize from other model, but start from scratch
    "reverse": True,  # do pairs both ways
    }

# Train network
train_correspondence_ae.train(parameter_dict)
