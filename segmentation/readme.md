Unsupervised Segmentation and Clustering of Buckeye English and NCHLT Xitsonga
==============================================================================

Overview
--------
Some of the results obtained below might be slightly different from that
obtained in [Kamper et al., 2016](http://arxiv.org/abs/1606.06950) because of
different initializations and small changes that I made. The general trends
should stay the same, though.



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



Examples of analysis scripts
----------------------------
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



Unigram speaker-dependent segmentation and evaluation
-----------------------------------------------------
The approach is to run `./segment.py` for each speaker independently. A
separate job is spawned for each speaker, so it is recommended that the steps
below are executed on a multi-core machine (preferably with at least 24 cores
for each of the speakers). The `stdbuf -oL` commands below just makes sure that
standard output gets flushed (so the outputs and log files are written as the
programs run). All options should be set in `./segment.py` or via the
command-line.

Perform segmentation on the different corpora and different feature
representations:

    # Devpart1
    stdbuf -oL ./spawn_segment_sd.py data/devpart1/mfcc.n_10.unsup_syl
    ./spawn_segment_sd_eval.py \
        models/devpart1/mfcc.n_10.unsup_syl/sd_1a34acf3f8/models.txt

    # ZeroSpeech
    stdbuf -oL ./spawn_segment_sd.py data/zs/mfcc.n_10.unsup_syl \
        --min_duration 25
    ./spawn_segment_sd_eval.py \
        models/zs/mfcc.n_10.unsup_syl/sd_79c7ecee12/models.txt

    # Xitsonga
    stdbuf -oL ./spawn_segment_sd.py --min_duration 25 --S_0_scale 0.0001 \
        data/tsonga/cae.d_13.n_10.unsup_syl
    ./spawn_segment_sd_eval.py \
        models/tsonga/cae.d_13.n_10.unsup_syl/sd_ce0a493211/models.txt

The first of these (applied to Devpart1) should give the following results:

    Avg. clustering average purity: 0.466451538841
    Std. clustering average purity: 0.0324127351051
    Avg. clustering one-to-one accuracy: 0.232250125334
    Std. clustering one-to-one accuracy: 0.0283745260209
    NED: 0.772528496666
    Errors: H = 19329, D = 14161, S = 56191, I = 9551, N = 89681
    No. of utterances: 24401
    No. of tokens: 85071
    uWER: 0.890969101593
    uWER_many: 0.681259129582

This corresponds (approximately) to the scores of the English1 BayesSeg-MFCC
system given in Tables 2 and 3 of
[Kamper et al., 2016](http://arxiv.org/abs/1606.06950).

The second of these (applied to ZeroSpeech) should give the following results:

    Avg. clustering average purity: 0.560622578439
    Std. clustering average purity: 0.0279629436698
    Avg. clustering one-to-one accuracy: 0.320508515728
    Std. clustering one-to-one accuracy: 0.0244399112599
    NED: 0.689065600198
    Errors: H = 12870, D = 30133, S = 26540, I = 920, N = 69543
    No. of utterances: 21498
    No. of tokens: 40330
    uWER: 0.828163869836
    uWER_many: 0.684094732755

This corresponds (approximately) to the scores of the English2
BayesSegMinDur-MFCC system given in Tables 2 and 3 of
[Kamper et al., 2016](http://arxiv.org/abs/1606.06950).

The last of the above results (on Xitsonga) should give:

    Avg. clustering average purity: 0.524288019617
    Std. clustering average purity: 0.0276663141808
    Avg. clustering one-to-one accuracy: 0.41867441981
    Std. clustering one-to-one accuracy: 0.042845486258
    NED: 0.875654602583
    Errors: H = 6704, D = 5346, S = 7798, I = 1999, N = 19848
    No. of utterances: 4058
    No. of tokens: 16501
    uWER: 0.7629484079
    uWER_many: 0.691555824264

This corresponds (approximately) to the scores of the Xitsonga
BayesSegMinDur-cAE  results given in Tables 2 and 4 of
[Kamper et al., 2016](http://arxiv.org/abs/1606.06950).

To evaluate any of the above models using the tools from the [Zero Resource
Speech Challenge](http://www.zerospeech.com), the decoded output from the above
systems need to be converted to the appropriate format, and can then be
evaluated:

    ./segment_sd_to_zs.py \
        models/zs/mfcc.n_10.unsup_syl/sd_79c7ecee12/models.txt
    cd ../../src/tde/english_dir/
    ln -s \
        ../../../bucktsong_segmentalist/segmentation/models/zs/mfcc.n_10.unsup_syl/sd_79c7ecee12/classes.txt \
        sd_79c7ecee12.classes.txt
    ./english_eval2 -j 5 sd_79c7ecee12.classes.txt sd_79c7ecee12
    cd -
    
In the `tde/sd_79c7ecee12/` directory, various files will then be created
given the various scores. For example, `tde/sd_79c7ecee12/nlp` contains:

    -------------------------------------
    NLP total
    #folds:    12
    #samples:  11153
    -------------------------------------
    measure    mean   std    min    max  
    ---------  -----  -----  -----  -----
    NED        0.556  0.024  0.498  0.591
    coverage   1.063  0.004  1.059  1.072
    -------------------------------------

This matches the NED scores of the English BayesSegMinDur-MFCC model in Figure
4 and Table 7 in [Kamper et al., 2016](http://arxiv.org/abs/1606.06950).



Unigram speaker-independent segmentation and evaluation
-------------------------------------------------------
Here I give an example only on the ZeroSpeech subset; results on Devpart1 and
Xitsonga can be generated in a similar way.

Speaker-indepedent segmentation and evaluation:

    # ZeroSpeech
    ./segment.py --data_dir data/zs/mfcc.n_10.unsup_syl --min_duration 25
    ./segment_eval.py \
        models/zs/mfcc.n_10.unsup_syl/26792eb71b/segment.pkl > \
        models/zs/mfcc.n_10.unsup_syl/26792eb71b/segment_eval.txt

Evaluate using the ZeroSpeech tools:

    ./segment_to_zs.py models/zs/mfcc.n_10.unsup_syl/26792eb71b/segment.pkl
    cd ../../src/tde/english_dir/
    ln -s \
        ../../../bucktsong_segmentalist/segmentation/models/zs/mfcc.n_10.unsup_syl/26792eb71b/classes.txt \
        26792eb71b.classes.txt
    ./english_eval2 -j 5 26792eb71b.classes.txt 26792eb71b
    cd -



Bigram single-speaker segmentation and evaluation
-------------------------------------------------
Again, set additional hyperparameters within `segment.py`:

    ./bigram_segment.py --min_duration 25 data/devpart1/mfcc.n_10.unsup_syl/s38
    ./segment_eval.py \
        models/devpart1/mfcc.n_10.unsup_syl/s38/.../segment.pkl



Sweeping options for models
---------------------------
The scripts below take a list of parameters in which can then be swept. This is
useful for parameter optimization on development data.

Sweep speaker-dependent segmentation options:

    stdbuf -oL ./spawn_sweep_segment.py \
        --sd
        --S_0_scale 0.001,0.0005,0.0001 \
        --am_K 1000,1500,2000,3000 \
        --serial data/devpart1/mfcc.n_10.unsup_syl > models/1.txt
    ./spawn_sweep_segment_eval.py models/1.txt

Sweep speaker-independent segmentation options:

    stdbuf -oL ./spawn_sweep_segment.py \
        --S_0_scale 0.001,0.0001 \
        --am_K 0.05landmarks,0.1landmarks,0.2landmarks,0.3landmarks \
        --serial data/devpart1/mfcc.n_10.unsup_syl > models/2.txt
    ./spawn_sweep_segment_eval.py models/2.txt

Sweeping options for single-speaker bigram model (note the directory which
points to the specific speaker):

    stdbuf -oL ./spawn_sweep_segment.py \
        --bigram \
        --S_0_scale 0.0001,0.001,0.002,0.005 \
        --lms 1.0,5.0,10.0,20.0,30.0,40.050.0 \
        --intrp_lambda 0.1 \
        data/devpart1/mfcc.n_10.unsup_syl/s38 > models/devpart1/s38_1.txt
