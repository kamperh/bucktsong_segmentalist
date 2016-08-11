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

Perform unsupervised acoustic segmentation for specific speaker using MFCC
embeddings and evaluate:

    # Downsampled Devpart1 s38 MFCCs
    ./segment.py --data_dir data/devpart1/mfcc.n_10.unsup_syl/s38 \
        --min_duration 25
    ./segment_eval.py \
        models/devpart1/mfcc.n_10.unsup_syl/s38/38f9526370/segment.pkl

This took around 20 minutes on my computer. The final word error rate for this
case (part of the output of `./segment_eval.py`):

    No. utterances: 1782
    No. tokens: 4517
    Error counts: H = 1321, D = 2904, S = 3121, I = 75, N = 7346
    Errors: {'n_ins': 75, 'n_sub': 3121, 'n_del': 2904, 'n_total': 7346, 'n_match': 1321}
    uWER: 0.830383882385
    Accuracy: 0.169616117615
    Error counts many-to-one: H = 2373, D = 2902, S = 2071, I = 73, N = 7346
    uWER_many: 0.686904437789
    Accuracy_many: 0.313095562211

This is the WER mentioned in [Kamper PhD, Section
5.3.37](http://www.kamperh.com/papers/kamper_phd2016.pdf).

Perform segmentation for Xitsonga speaker using MFCC embeddings:

    ./segment.py --data_dir data/tsonga/mfcc.n_10.unsup_syl/001m/ \
        --min_duration 25
    ./segment_eval.py \
        models/tsonga/mfcc.n_10.unsup_syl/001m/1bc6fa7178/segment.pkl

Producing output:

    No. utterances: 138
    No. tokens: 665
    Error counts: H = 271, D = 123, S = 284, I = 110, N = 678
    Errors: {'n_ins': 110, 'n_sub': 284, 'n_del': 123, 'n_total': 678, 'n_match': 271}
    uWER: 0.762536873156
    Accuracy: 0.237463126844
    Error counts many-to-one: H = 292, D = 116, S = 270, I = 103, N = 678
    uWER_many: 0.721238938053
    Accuracy_many: 0.278761061947

Perform segmentation for Xitsonga speaker using cAE embeddings:

    ./segment.py --data_dir data/tsonga/cae.d_13.n_10.unsup_syl/001m/ \
        --S_0_scale 0.0001 --min_duration 25
    ./segment_eval.py \
        models/tsonga/cae.d_13.n_10.unsup_syl/001m/31a47a2465/segment.pkl

Producing output:

    No. utterances: 138
    No. tokens: 594
    Error counts: H = 249, D = 160, S = 269, I = 76, N = 678
    Errors: {'n_ins': 76, 'n_sub': 269, 'n_del': 160, 'n_total': 678, 'n_match': 249}
    uWER: 0.744837758112
    Accuracy: 0.255162241888
    Error counts many-to-one: H = 267, D = 149, S = 262, I = 65, N = 678
    uWER_many: 0.702064896755
    Accuracy_many: 0.297935103245



Analysis
--------

Plot embeddings:

    ./plot_embeddings.py \
        models/devpart1/mfcc.n_10.unsup_syl/s38/38f9526370/segment.pkl

Get the wav file for a specific cluster:

    ./get_cluster_wav.py \
        models/devpart1/mfcc.n_10.unsup_syl/s38/38f9526370/segment.pkl PT795

Analyze the top non-function words:

    ./analyze_top_words.py \
        models/devpart1/mfcc.n_10.unsup_syl/s38/38f9526370/segment.pkl

Plot a heatmap of phones in the biggest clusters:

    ./plot_clusters_prons.py \
        models/devpart1/mfcc.n_10.unsup_syl/s38/38f9526370/segment.pkl
    ./plot_clusters_prons.py \
        --clusters 323,252,1112,647,746,1087,395,867,46,1429,110,442,1079,1118,367,1435,255,147,1190,607,1163,1395,878,103,1328,61,1356,1280,684,1077 \
        models/devpart1/mfcc.n_10.unsup_syl/s38/38f9526370/segment.pkl

Plot a heatmap of the words in the biggest clusters:

    ./plot_clusters_words.py \
        models/devpart1/mfcc.n_10.unsup_syl/s38/38f9526370/segment.pkl

Print or plot some mappings:

    ./print_mappings.py \
        models/devpart1/mfcc.n_10.unsup_syl/s38/38f9526370/segment.pkl
    ./plot_mappings.py --n_true_tokens_min 4 \
        models/devpart1/mfcc.n_10.unsup_syl/s38/38f9526370/segment.pkl
