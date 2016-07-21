Recipe: Segmentation and Clustering of Buckeye English and NCHLT Xitsonga
=========================================================================

Overview
--------

This is a recipe for unsupervised segmentation and clustering of subsets of the
Buckeye English and NCHLT Xitsonga corpora. Details of the approach is given in
[Kamper et al., 2016](http://arxiv.org/abs/1606.06950):

- H. Kamper, A. Jansen, and S. J. Goldwater, "A segmental framework for
  fully-unsupervised large-vocabulary speech recognition," *arXiv preprint
  arXiv:1606.06950*, 2016.

Please cite this paper if you use this code.

The recipe below makes use of the separate
[segmentalist](https://github.com/kamperh/segmentalist/) package which performs
the actual unsupervised segmentation and clustering and was developed together
with this recipe.



Disclaimer
----------

The code provided here is not pretty. But I believe that research should be
reproducible, and I hope that this repository is sufficient to make this
possible for the paper mentioned above. I provide no guarantees with the code,
but please let me know if you have any problems, find bugs or have general
comments.



Preliminaries
-------------

Install all the standalone dependencies (see Dependencies section below). Then
clone the required GitHub repositories into `../src/` as follows:

    mkdir ../src/
    git clone https://github.com/kamperh/segmentalist.git ../src/segmentalist/
    git clone https://github.com/kamperh/speech_correspondence.git \
        ../src/speech_correspondence/
    git clone https://github.com/kamperh/speech_dtw.git ../src/speech_dtw/
    git clone https://github.com/bootphon/tde.git ../src/tde

For both `segmentalist` and `speech_dtw`, you need to run `make` to build. Unit
tests can be performed by running `make test`. See the readmes for more
details.

The `speech_correspondence` and `speech_dtw` repositories are only necessary if
you plan to do correspondence autoencoder (cAE) feature extraction. This
repository uses the Theano and Pylearn2 dependencies, which is unnecessary if
cAE features will not be used. The `tde` repository is only necessary if you
plan to also calculate the evaluation metrics from the Zero Resource Speech
Challenge 2015; without `tde` you will not be able to calculate the metrics in
Section 4.5 of [Kamper et al., 2016](http://arxiv.org/abs/1606.06950), but you
will still be able to calculate the other metrics in the paper.



Feature extraction
------------------

Some preprocessed resources are given in `features/data/`. Extract MFCC
features by running the steps in [features/readme.md](features/readme.md). Some
steps are optional depending on whether you intend to train a cAE (see below).



Correspondence autoencoder features (optional)
----------------------------------------------

In [Kamper et al., 2016](http://arxiv.org/abs/1606.06950) we compare both MFCCs
and correspondence autoencoder (cAE) features as input to our system. It is not
necessary to perform the steps below if you are happy with using MFCCs. The cAE
was first introduced in this paper:

- H. Kamper, M. Elsner, A. Jansen, and S. J. Goldwater, "Unsupervised neural
  network based feature extraction using weak top-down constraints," in Proc.
  ICASSP, 2015.

The cAE is trained on word pairs discovered using an unsupervised term
discovery (UTD) system (based on the code available
[here](https://github.com/arenjansen/ZRTools)). This UTD system does not form
part of the repository here. Instead, the output word pairs discovered by the
UTD system are provided as part of the repository in the following files:

- English pairs: `features/data/buckeye.fdlps.0.93.pairs`
- Xitsonga pairs: `features/data/zs_tsonga.fdlps.0.925.pairs.v0`

The MFCC features for these pairs were extracted as part of feature extraction
(previous section). To train the cAE, run the steps in
[cae/readme.md](cae/readme.md).



Unsupervised syllable boundary detection
----------------------------------------




Acoustic word embeddings through downsampling
---------------------------------------------




Segmentalist: Unsupervised segmentation and clustering
------------------------------------------------------



Dependencies
------------

Standalone packages:

- [Python](https://www.python.org/)
- [Cython](http://cython.org/): Used by the `segmentalist` and `speech_dtw`
  repositories below.
- [HTK](http://htk.eng.cam.ac.uk/): Used for MFCC feature extraction.
- [Theano](http://deeplearning.net/software/theano/): Required by the
  `speech_correspondence` repository below.
- [Pylearn2](http://deeplearning.net/software/pylearn2/): Required by the
  `speech_correspondence` repository below.

Repositories from GitHub:

- [segmentalist](https://github.com/kamperh/segmentalist/): This is the main
  segmentation software developed as part of this project. Should be cloned
  into the directory `../src/segmentalist/`, done in the Preliminary section
  above.
- [speech_correspondence](https://github.com/kamperh/speech_correspondence/):
  Used for correspondence autoencoder feature extraction.  Should be cloned
  into the directory `../src/speech_correspondence/`, as done in the
  Preliminary section above.
- [speech_dtw](https://github.com/kamperh/speech_dtw/): Used for correspondence
  autoencoder feature extraction.  Should be cloned into the directory
  `../src/speech_dtw/`, as done in the Preliminary section above.
- [tde](https://github.com/bootphon/tde/): The Zero Resource Speech Challenge
  evaluation tools. Should be cloned into the directory `tde/`, as done in the
  Preliminary section above.



Contributors
------------

- [Herman Kamper](http://www.kamperh.com/)
- [Aren Jansen](http://www.clsp.jhu.edu/~ajansen/)
- [Sharon Goldwater](http://homepages.inf.ed.ac.uk/sgwater/)
