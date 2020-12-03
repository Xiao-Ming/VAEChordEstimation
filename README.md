# Semi-supervised Neural Chord Estimation Based on a Variational Autoencoder with Latent Chord Labels and Features

Links to the original paper: [IEEE](https://ieeexplore.ieee.org/document/9246301), [arxiv](https://arxiv.org/abs/2005.07091) 

## Quick navigations to some important codes

If you are interested in implementation details of the presented model, you may want to see the following scripts:

- `net_chordrec.py` -- The definition of the chord estimation model.
- `net_generative.py` -- The proposed VAE is defined as the `UnifiedModel` class.
- `training.py` -- Codes for training loops.
- `const.py`  --  Some hyper-parameters are specified.
- `Experiment_supervised_training.py` -- Performs cross-validation experiments for supervised training (`976+0` in Fig.3). 
- `Experiment_vae_training.py` -- Performs cross-validation experiment for the proposed semi-supervised training, which uses part of annotated songs as unsupervised data (left half of Fig.3). 
- `Experiment_vae_training_with_unsupervised_data.py` -- Performs cross-validation experiment for the proposed semi-supervised training, which uses the non-annotated data as unsupervised data (right half of Fig.3).


## Training dataset

The dataset for training our model is stored in `dataset` folder, which includes annotated data pairs of 1217 songs, and non-annotated data of 700 songs.
`chordlab` folder stores the ground-truth chord labels of each annotated songs. `chroma` folder stores the 36-dimension feature sequences extracted from each song, using a DNN extractor proposed in "Automatic Audio Chord Recognition with MIDI-Traind Deep Feature and BLSTM-CRF Sequence Decoding Model" (see [this repository](https://github.com/Xiao-Ming/ChordRecognitionMIDITrainedExtractor)).


## Dependencies
The experiments were performed on Python 3.6 and the following libraries were used:

- Chainer 7.0.0
- Cupy 7.0.0
- librosa 0.7.0
- mir_eval 0.5

It is ok to use the later versions of those libraries since (as we know) currently there are no major changes in API from the above versions. 

## How to estimate chord from raw audio files?

Code for estimating chord from raw audio is not provided in this repository.

## How to reproduce the experiments?

Run the script in `scripts` folder under the root directory. When experiment is finished, the chord estimation results can be found at corresponding 'estimated' folders.
Example: `source scripts\experiment_semisup_fulldata.sh`

- `experiment_supervised.sh` -- the three different training methods for `976+0` in Fig.3.
- `experiment_proposed_vae.sh` -- `VAE_MR_SSL` and `VAE_MR_SL` experiments for the left part of Fig.3.
- `experiment_nomarkov_vae.sh` -- `VAE_UN_SSL` and `VAE_UN_SL` experiments for the left part of Fig.3.
- `experiment_semisupervised_vae.sh` -- `VAE_MR_SSL` experiments for `976+700` in Fig.3.

