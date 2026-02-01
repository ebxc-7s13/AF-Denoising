# AF-Denoising
A Self-Supervised Wavelet Residual Network for Label-Free Autofluorescent Image Denoising
WRTPNet Ablation Study: Self-Supervised Denoising Benchmarks

This repository contains the complete codebase for reproducing the ablation study of WRTPNet (Wavelet Residual Transformer Pyramid Network) and benchmarking it against various self-supervised denoising methods.

Benchmark Methods

This repository implements and benchmarks the following methods:

Noise2Void (N2V)

Neighbor2Neighbor (Ne2Ne)

Self2Self (Dropout-based inference)

Noise2Same (Noise2Self logic adapted)

DIP (Deep Image Prior)

CBM3D (Conventional non-learning baseline)

WRTPNet Variants (Baseline, Attentive, Adaptive Masking)

Structure

main.py: Entry point. Orchestrates data splitting, noise injection, training loops for all models, and result logging.

engine.py: Contains training loops (train_noise2void, train_ne2ne, etc.) and inference logic.

models.py: PyTorch implementations of UNet and WRTPNet variants.

utils.py: Configuration, metrics (PSNR, SSIM, FSIM, VIF, FOM), data loading, and noise generation.

requirements.txt: Python dependencies.

Installation

pip install -r requirements.txt


Usage

Configure Paths: Open utils.py and modify Config.INPUT_DIR to point to your dataset directory.

Run Benchmark:

python main.py


Output: Results, images, and CSV logs will be saved to the directory specified in Config.OUTPUT_DIR.

Reproducibility

The script iterates over three random seeds (42, 100, 123) and three noise types (PoissonGauss, Gaussian, Poisson) to ensure robust statistical evaluation.
