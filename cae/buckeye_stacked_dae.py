#!/usr/bin/env python

"""
Train a stacked dAE on the Buckeye corpus.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2015
"""

import argparse
import sys
sys.path.append("../../src/speech_correspondence/speech_correspondence")
import train_stacked_dae


# Get dataset
def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("dataset", type=str, help="dataset label", choices=["zs", "devpart1", "tsonga"])
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()
args = check_argv()
dataset = args.dataset

# Network parameter settings
parameter_dict = {
    "dataset_npy_fn": "../features/subsets/" + dataset + "/" + dataset + ".mfcc.cmvn_dd.npy",
    "models_basedir": "models/" + dataset + "/",
    "dim_input": 39,
    "layer_spec_str": "[100] * 9",  # "layer_spec_str": "100,100,100,100,100,100,100,13,100",
    "corruption": 0,
    "max_epochs": 5,
    "batch_size": 2048,
    "learning_rate": 0.002,
    }

# Train network
train_stacked_dae.train(parameter_dict)
