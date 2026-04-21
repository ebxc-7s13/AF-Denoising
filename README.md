# FASCANet: Frequency-Aware Spatial Cross-Attention Denoising

A Noise-Supervised Wavelet-Domain Denoising Network for Label-Free Oral Cancer Screening from Autofluorescent Images.

## Overview

FASCANet is a noise-supervised framework designed specifically for the low-SNR regime of clinical Autofluorescence Imaging (AFI). It processes images in the wavelet domain to separate low-frequency structural components from high-frequency textures, ensuring that diagnostically critical metabolic signatures are not blurred during noise suppression.

## Architecture

The FASCANet architecture is defined by three primary technical pillars:

* **Daubechies-2 (db2) Wavelet Decomposition**: Uses a single-level 2D discrete wavelet transform to provide a piecewise-linear frequency decomposition. This significantly reduces reconstruction artifacts compared to the simpler Haar basis.
* **Spatial Cross-Attention (SCA)**: A bidirectional coupling mechanism between frequency branches. It generates full-resolution per-pixel attention maps to exchange edge cues and structural context every two residual blocks.
* **Wavelet-Domain Residual Learning**: The network predicts band-specific corrections (residuals) rather than full image synthesis, preserving the weak endogenous fluorophore signals required for cancer screening.

## Benchmark Methods

This repository evaluates FASCANet against the following established denoising baselines:

* **FASCANet (Proposed)**: db2-based wavelet residual network with SCA.
* **Noise2Void (N2V)**: Self-supervised blind-spot masking.
* **Neighbor2Neighbor (Ne2Ne)**: Subsampling-based self-supervision.
* **Self2Self**: Dropout-based ensemble inference.
* **Noise2Same**: Blind denoising via self-supervision.
* **DIP (Deep Image Prior)**: Optimization-based reconstruction.
* **SwinConv**: SCUnet inspired arcehtecture.
* **CBM3D**: Traditional collaborative filtering.

## Project Structure

* **main.py**: Entry point for benchmarking. Orchestrates dataset splitting, noise injection, and evaluation across random seeds.
* **training.py**: The training engine. Contains optimized loops for FASCANet and all baselines.
* **models.py**: Architecture definitions (FASCANet implementation, SCA modules, and UNet).
* **utils.py**: Utilities for configuration, clinical metrics (PSNR, SSIM, FSIM, VIF, MS-SSIM, FOM), and noise generation.

## Installation

```pip install torch torchvision numpy opencv-python scikit-image PyWavelets pandas scipy```

## Configuration

*Update paths and hardware settings in utils.py:

* Config.INPUT_DIR: Point to your AFI dataset.

* Config.OUTPUT_DIR: Define your results directory.

## Execution

Run the full benchmarking and ablation suite:

```python main.py```


## Clinical Metrics

The framework evaluates perceptual and structural fidelity using:

* PSNR/SSIM: Standard signal fidelity.

* FSIM: Feature Similarity Index.

* VIF: Visual Information Fidelity.

* MS-SSIM: Multi-Scale Structural Similarity.

* FOM: Pratt’s Figure of Merit for edge preservation.

* Redox Ratio: Verification of $FAD / [NADH + FAD]$ preservation.

## Citation

If you utilize this framework, cite the original research:
FASCANet: Frequency-Aware Spatial Cross-Attention Denoising for Label-Free Oral Cancer Screening from Autofluorescent Images.
