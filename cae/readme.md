Correspondence Autoencoder
==========================

Training
--------

Pretrain and train a cAE on a Buckeye portion, and encode a test set (the
scripts themselves should be edited to change training options):

    # Devpart1
    ./buckeye_stacked_dae.py devpart1
    ./buckeye_correspondence_ae.py devpart1
    ../../src/speech_correspondence/speech_correspondence/encode.py \
        --use_layer -2 \
        ../features/wordpairs/devpart1/devpart1_samediff_terms.mfcc.cmvn_dd.npz \
        models/devpart1/100x9.batch_size2048.corruption0.max_epochs5/correspondence_ae.devpart1_utd.max_epochs120.reverseTrue.pkl
    ../../src/speech_correspondence/speech_correspondence/encode.py \
        --use_layer -2 \
        ../features/subsets/devpart1/devpart1.mfcc.cmvn_dd.npz \
        models/devpart1/100x9.batch_size2048.corruption0.max_epochs5/correspondence_ae.devpart1_utd.max_epochs120.reverseTrue.pkl

    # ZeroSpeech
    ./buckeye_stacked_dae.py zs
    ./buckeye_correspondence_ae.py zs
    ../../src/speech_correspondence/speech_correspondence/encode.py \
        --use_layer -2 \
        ../features/wordpairs/zs/zs_samediff_terms.mfcc.cmvn_dd.npz \
        models/zs/100-100-100-100-100-100-100-13-100.batch_size2048.corruption0.max_epochs5/correspondence_ae.zs_utd.max_epochs120.reverseTrue.pkl
    ../../src/speech_correspondence/speech_correspondence/encode.py \
        --use_layer -2 \
        ../features/subsets/zs/zs.mfcc.cmvn_dd.npz \
        models/zs/100-100-100-100-100-100-100-13-100.batch_size2048.corruption0.max_epochs5/correspondence_ae.zs_utd.max_epochs120.reverseTrue.pkl

    # Tsonga
    ./buckeye_stacked_dae.py tsonga
    ./buckeye_correspondence_ae.py tsonga
    ../../src/speech_correspondence/speech_correspondence/encode.py \
        --use_layer -2 \
        ../features/wordpairs/tsonga/tsonga_samediff_terms.mfcc.cmvn_dd.npz \
        models/tsonga/100-100-100-100-100-100-100-13-100.batch_size2048.corruption0.max_epochs5/correspondence_ae.tsonga_utd.max_epochs120.reverseTrue.pkl
    ../../src/speech_correspondence/speech_correspondence/encode.py \
        --use_layer -2 \
        ../features/subsets/tsonga/tsonga.mfcc.cmvn_dd.npz \
        models/tsonga/100-100-100-100-100-100-100-13-100.batch_size2048.corruption0.max_epochs5/correspondence_ae.tsonga_utd.max_epochs120.reverseTrue.pkl
