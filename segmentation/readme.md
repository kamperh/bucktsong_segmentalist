Unsupervised Segmentation and Clustering of Buckeye English and NCHLT Xitsonga
==============================================================================

Data pre-processing
-------------------

Format the Buckeye and Tsonga data into the input format used by
`segmentalist`:

    # Downsampled MFCCs
    ./get_data_downsample.py devpart1 unsup_syl mfcc 10
    ./get_data_downsample.py zs unsup_syl mfcc 10
    ./get_data_downsample.py tsonga unsup_syl mfcc 10

    # Downsampled cAE
    ./get_data_downsample.py devpart1 unsup_syl cae.d_13 10
    ./get_data_downsample.py zs unsup_syl cae.d_13 10
    ./get_data_downsample.py tsonga unsup_syl cae.d_13 10

Get a subset of the data for a particular speaker:

    ./get_data_speaker.py data/devpart1/mfcc.n_10.unsup_syl/ s38
    ./get_data_speaker.py data/tsonga/mfcc.n_10.unsup_syl/ 001m

Get data for all the speakers (calls `get_data_speaker.py` repeatedly):

    # MFCC
    ./get_data_sd.py devpart1 unsup_syl mfcc 10
    ./get_data_sd.py zs unsup_syl mfcc 10
    ./get_data_sd.py tsonga unsup_syl mfcc 10

    # cAE
    ./get_data_sd.py devpart1 unsup_syl cae.d_13 10
    ./get_data_sd.py zs unsup_syl cae.d_13 10
    ./get_data_sd.py tsonga unsup_syl cae.d_13 10



Unigram single-speaker segmentation and evaluation
--------------------------------------------------

Perform unsupervised acoustic segmentation for specific speakers and evaluate:

    # Downsampled Devpart1 s38 MFCCs
    ./segment.py --data_dir data/devpart1/mfcc.n_10.unsup_syl/s38
    ./segment_eval.py --level phone \
        models/devpart1/mfcc.n_10.unsup_syl/s38/bb55d977f7/segment.pkl
    ./segment_eval.py \
        models/devpart1/mfcc.n_10.unsup_syl/s38/bb55d977f7/segment.pkl




