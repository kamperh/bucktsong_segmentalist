Feature Extraction
==================


MFCCs
-----

Move to the MFCC feature extraction directory:

    cd mfcc

Edit the `buckeye_wavs` and `tsonga_wavs` variables at the start of
`get_raw_htk.py` to reflect your paths.

Extract the MFCCs with CMVN and deltas and delta-delta features for the whole
Buckeye corpus (this can take a while):

    ./get_raw_htk.py buckeye
    ./get_segments_scp.py buckeye
    ./get_cmvn_dd_htk.py buckeye
    mkdir buckeye/numpy
    ./write_htk_npz.py buckeye/cmvn_dd buckeye/numpy/buckeye.mfcc.cmvn_dd.npz

Extract the MFCCs with CMVN and deltas and delta-delta features for the Tsonga
corpus (this can take a while):

    ./get_raw_htk.py tsonga
    ./get_segments_scp.py tsonga
    ./get_cmvn_dd_htk.py tsonga
    mkdir tsonga/numpy
    ./write_htk_npz.py tsonga/cmvn_dd tsonga/numpy/tsonga.mfcc.cmvn_dd.npz



Extract and analyze subsets
---------------------------

Move to the subset directory:

    cd subsets

Create Buckeye subsets:

    mkdir devpart1
    ./get_subset_npz.py ../mfcc/buckeye/numpy/buckeye.mfcc.cmvn_dd.npz \
        ../data/devpart1_speakers.list devpart1/devpart1.mfcc.cmvn_dd.npz
    ./npz_to_npy.py devpart1/devpart1.mfcc.cmvn_dd.npz devpart1/devpart1.mfcc.cmvn_dd.npy

    mkdir devpart2
    ./get_subset_npz.py ../mfcc/buckeye/numpy/buckeye.mfcc.cmvn_dd.npz \
        ../data/devpart2_speakers.list devpart2/devpart2.mfcc.cmvn_dd.npz
    ./npz_to_npy.py devpart2/devpart2.mfcc.cmvn_dd.npz devpart2/devpart2.mfcc.cmvn_dd.npy
    
    mkdir zs
    ./get_subset_npz.py ../mfcc/buckeye/numpy/buckeye.mfcc.cmvn_dd.npz \
        ../data/zs_speakers.list zs/zs.mfcc.cmvn_dd.npz
    ./npz_to_npy.py zs/zs.mfcc.cmvn_dd.npz zs/zs.mfcc.cmvn_dd.npy

    mkdir tsonga
    ln -s ../../mfcc/tsonga/numpy/tsonga.mfcc.cmvn_dd.npz tsonga
    ./npz_to_npy.py tsonga/tsonga.mfcc.cmvn_dd.npz tsonga/tsonga.mfcc.cmvn_dd.npy

    mkdir tsongdev
    ./get_subset_npz.py ../mfcc/tsonga/numpy/tsonga.mfcc.cmvn_dd.npz \
        ../data/tsongdev_speakers.list tsongdev/tsongdev.mfcc.cmvn_dd.npz

    mkdir tsongtest
    ./get_subset_npz.py ../mfcc/tsonga/numpy/tsonga.mfcc.cmvn_dd.npz \
        ../data/tsongtest_speakers.list tsongtest/tsongtest.mfcc.cmvn_dd.npz

Analyze speaker lists:

    ./analyze_buckeye_speaker_list.py ../data/devpart1_speakers.list
    ./analyze_buckeye_speaker_list.py ../data/zs_speakers.list

Analyze the lengths, means and variances in a given npz file:

    ./analyze_buckeye_npz.py zs/zs.mfcc.cmvn_dd.npz



Word-pair extraction
--------------------

These steps are required if you are planning to train the cAE. Move to the
word-pair directory:

    cd wordpairs

Get the features for the word pairs discovered only in the respective subset
portions:

    # Buckeye, pairs from Aren
    mkdir buckeye
    ./strip_nonvad_from_pairs.py \
        ../mfcc/buckeye/lists/segments.list \
        ../data/buckeye.fdlps.0.93.pairs \
        buckeye/buckeye_utd_pairs.list
    ./get_terms_from_pairs.py buckeye/buckeye_utd_pairs.list buckeye/buckeye_utd_terms.list
    ./get_segments_from_npz.py \
        ../mfcc/buckeye/numpy/buckeye.mfcc.cmvn_dd.npz \
        buckeye/buckeye_utd_terms.list buckeye/buckeye_utd_terms.mfcc.cmvn_dd.npz

    # ZeroSpeech, pairs from Aren
    mkdir zs
    ./strip_nonvad_from_pairs.py \
        ../mfcc/buckeye/lists/segments.list \
        ../data/zs_buckeye.fdlps.0.93.pairs \
        zs/zs_utd_pairs.list
    ./get_terms_from_pairs.py zs/zs_utd_pairs.list zs/zs_utd_terms.list
    ./get_segments_from_npz.py \
        ../subsets/zs/zs.mfcc.cmvn_dd.npz \
        zs/zs_utd_terms.list zs/zs_utd_terms.mfcc.cmvn_dd.npz

    # Devpart1
    mkdir devpart1
    ./strip_nonvad_from_pairs.py \
        ../mfcc/buckeye/lists/segments.list \
        ../data/buckeye.fdlps.0.93.pairs buckeye/buckeye_utd_pairs.list
    ./get_pairs_for_speakers.py \
        buckeye/buckeye_utd_pairs.list \
        ../data/devpart1_speakers.list devpart1/devpart1_utd_pairs.list
    ./get_terms_from_pairs.py devpart1/devpart1_utd_pairs.list \
        devpart1/devpart1_utd_terms.list
    ./get_segments_from_npz.py \
        ../subsets/devpart1/devpart1.mfcc.cmvn_dd.npz \
        devpart1/devpart1_utd_terms.list devpart1/devpart1_utd_terms.mfcc.cmvn_dd.npz

    # Tsonga
    mkdir tsonga
    ./strip_nonvad_from_pairs.py \
        ../mfcc/tsonga/lists/segments.list \
        ../data/zs_tsonga.fdlps.0.925.pairs.v0 \
        tsonga/tsonga_utd_pairs.list
    ./get_terms_from_pairs.py tsonga/tsonga_utd_pairs.list tsonga/tsonga_utd_terms.list
    ./get_segments_from_npz.py \
        ../subsets/tsonga/tsonga.mfcc.cmvn_dd.npz \
        tsonga/tsonga_utd_terms.list tsonga/tsonga_utd_terms.mfcc.cmvn_dd.npz

Get the same-different terms and subsets:

    ./get_samediff.py buckeye
    ./get_samediff.py tsonga

    # Buckeye
    ./get_segments_from_npz.py \
        ../mfcc/buckeye/numpy/buckeye.mfcc.cmvn_dd.npz \
        buckeye/buckeye_samediff_terms.list \
        buckeye/buckeye_samediff_terms.mfcc.cmvn_dd.npz

    # Devpart1
    ./get_terms_for_speakers.py \
        buckeye/buckeye_samediff_terms.list \
        ../data/devpart1_speakers.list \
        devpart1/devpart1_samediff_terms.list
    ./get_segments_from_npz.py \
        ../subsets/devpart1/devpart1.mfcc.cmvn_dd.npz \
        devpart1/devpart1_samediff_terms.list \
        devpart1/devpart1_samediff_terms.mfcc.cmvn_dd.npz

    # Devpart2
    mkdir devpart2
    ./get_terms_for_speakers.py \
        buckeye/buckeye_samediff_terms.list \
        ../data/devpart2_speakers.list \
        devpart2/devpart2_samediff_terms.list
    ./get_segments_from_npz.py \
        ../subsets/devpart2/devpart2.mfcc.cmvn_dd.npz \
        devpart2/devpart2_samediff_terms.list \
        devpart2/devpart2_samediff_terms.mfcc.cmvn_dd.npz

    # ZeroSpeech
    ./get_terms_for_speakers.py \
        buckeye/buckeye_samediff_terms.list \
        ../data/zs_speakers.list \
        zs/zs_samediff_terms.list
    ./get_segments_from_npz.py \
        ../subsets/zs/zs.mfcc.cmvn_dd.npz \
        zs/zs_samediff_terms.list \
        zs/zs_samediff_terms.mfcc.cmvn_dd.npz

    # Tsonga
    ./get_segments_from_npz.py \
        ../subsets/tsonga/tsonga.mfcc.cmvn_dd.npz \
        tsonga/tsonga_samediff_terms.list \
        tsonga/tsonga_samediff_terms.mfcc.cmvn_dd.npz

    # Tsongdev
    mkdir tsongdev
    ./get_terms_for_speakers.py \
        tsonga/tsonga_samediff_terms.list \
        ../data/tsongdev_speakers.list \
        tsongdev/tsongdev_samediff_terms.list
    ./get_segments_from_npz.py \
        ../subsets/tsongdev/tsongdev.mfcc.cmvn_dd.npz \
        tsongdev/tsongdev_samediff_terms.list \
        tsongdev/tsongdev_samediff_terms.mfcc.cmvn_dd.npz

    # Tsongtest
    mkdir tsongtest
    ./get_terms_for_speakers.py \
        tsonga/tsonga_samediff_terms.list \
        ../data/tsongtest_speakers.list \
        tsongtest/tsongtest_samediff_terms.list
    ./get_segments_from_npz.py \
        ../subsets/tsongtest/tsongtest.mfcc.cmvn_dd.npz \
        tsongtest/tsongtest_samediff_terms.list \
        tsongtest/tsongtest_samediff_terms.mfcc.cmvn_dd.npz


Word-pair alignment
-------------------

These steps are required if you are planning to train the cAE. Move to the
directory where word alignment is performed:

    cd wordpair_aligns

Process and get frame alignments for word pairs discovered only in the
respective subset portions:

    # Devpart1
    mkdir devpart1
    ./get_npz_keys.py \
        ../wordpairs/devpart1/devpart1_utd_terms.mfcc.cmvn_dd.npz devpart1/devpart1_utd_keys.list
    ./get_pairs_list.py \
        ../wordpairs/devpart1/devpart1_utd_pairs.list \
        devpart1/devpart1_utd_keys.list \
        devpart1/devpart1_utd_pairs_keys.list
    ../../src/speech_dtw/utils/calculate_dtw_paths.py --input_fmt npz \
        devpart1/devpart1_utd_pairs_keys.list \
        ../wordpairs/devpart1/devpart1_utd_terms.mfcc.cmvn_dd.npz \
        devpart1/devpart1_utd_pairs_paths.pkl
    ./get_frames_from_paths.py \
        ../wordpairs/devpart1/devpart1_utd_terms.mfcc.cmvn_dd.npz \
        devpart1/devpart1_utd_pairs_keys.list \
        devpart1/devpart1_utd_pairs_paths.pkl \
        devpart1/devpart1_utd.word1.npy \
        devpart1/devpart1_utd.word2.npy

    # ZeroSpeech, pairs from Aren
    mkdir zs
    ./get_npz_keys.py \
        ../wordpairs/zs/zs_utd_terms.mfcc.cmvn_dd.npz zs/zs_utd_keys.list
    ./get_pairs_list.py \
        ../wordpairs/zs/zs_utd_pairs.list \
        zs/zs_utd_keys.list \
        zs/zs_utd_pairs_keys.list
    ../../src/speech_dtw/utils/calculate_dtw_paths.py --input_fmt npz \
        zs/zs_utd_pairs_keys.list \
        ../wordpairs/zs/zs_utd_terms.mfcc.cmvn_dd.npz \
        zs/zs_utd_pairs_paths.pkl
    ./get_frames_from_paths.py \
        ../wordpairs/zs/zs_utd_terms.mfcc.cmvn_dd.npz \
        zs/zs_utd_pairs_keys.list \
        zs/zs_utd_pairs_paths.pkl \
        zs/zs_utd.word1.npy \
        zs/zs_utd.word2.npy

    # Tsonga
    mkdir tsonga
    ./get_npz_keys.py \
        ../wordpairs/tsonga/tsonga_utd_terms.mfcc.cmvn_dd.npz tsonga/tsonga_utd_keys.list
    ./get_pairs_list.py \
        ../wordpairs/tsonga/tsonga_utd_pairs.list \
        tsonga/tsonga_utd_keys.list \
        tsonga/tsonga_utd_pairs_keys.list
    ../../src/speech_dtw/utils/calculate_dtw_paths.py --input_fmt npz \
        tsonga/tsonga_utd_pairs_keys.list \
        ../wordpairs/tsonga/tsonga_utd_terms.mfcc.cmvn_dd.npz \
        tsonga/tsonga_utd_pairs_paths.pkl
    ./get_frames_from_paths.py \
        ../wordpairs/tsonga/tsonga_utd_terms.mfcc.cmvn_dd.npz \
        tsonga/tsonga_utd_pairs_keys.list \
        tsonga/tsonga_utd_pairs_paths.pkl \
        tsonga/tsonga_utd.word1.npy \
        tsonga/tsonga_utd.word2.npy



Buckeye set definitions
-----------------------

Buckeye is divided into a number of sets based on the speakers:

- sample: s2801a, s2801b, s2802a, s2802b, s2803a, s3701a, s3701b, s3702a,
  s3702b, s3703a, s3703b.
- devpart1: s02, s04, s05, s08, s12, s16, s03, s06, s10, s11, s13, s38.
- devpart2: s18, s17, s37, s39, s19, s22, s40, s34.
- ZS: s20, s25, s27, s01, s26, s31, s29, s23, s24, s32, s33, s30.
- testpart2: s07, s14, s09, s21, s36, s35, s15, s28.
- dev: devpart1 + devpart2.
- test: ZS + testpart2. 
