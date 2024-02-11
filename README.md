# LPT Lensing Comparison

The goal of this repo is to provide an open-source and easy-to-use testbed for comparing full-field weak lensing constraints under an LPT gravity model, lognormal convergence model, and gaussian convergence model. 

A guiding principle of this repo is that everything should be runnable within a few hours on colab.


## Description of fiducial forward model

We are adopting the following setting:
- **patch size**: 5 x 5 deg^2
- **pixel size**: 5 arcmin^2 (which gives us an lmax of around 3000)
- **galaxy number density**: 27 gals/arcmin^2
- **galaxy ellipticity dispersion**: 0.3

![image](https://github.com/EiffL/LPTLensingComparison/assets/861591/24994aeb-87fd-4644-8005-249cd6fbf1c4)

![image](https://github.com/EiffL/LPTLensingComparison/assets/861591/8a068d7e-8df4-4ac8-81e5-00efccd70111)

### Accuracy of LPT model
<a href="https://colab.research.google.com/github/EiffL/LPTLensingComparison/blob/main/notebooks/LPTLensingAccuracyTest.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

We check that our LPT model has sufficient resolution to reproduce the linear theory angular power spectra (i.e. without halofit).
![image](https://github.com/EiffL/LPTLensingComparison/assets/861591/05e19df0-db65-4527-a93e-455f78d66726)

We obtain this match to theory using the following lightcone volume and resolution:
- **lightcone volume**: 400 x 400 x 4000 Mpc/h
- **lightcone resolution**: 300 x 300 x 256 voxels

### Inference
<a href="https://colab.research.google.com/github/EiffL/LPTLensingComparison/blob/main/notebooks/Inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Given that we have a matching theory for our numerical simulator, we can easily compute the 2pt posterior of a simulated map, as well as the full-field posterior by HMC.

Preliminary (unconverged) results:
![image](https://github.com/EiffL/LPTLensingComparison/assets/861591/aad2834d-4252-4e31-94ab-13e91b56fb1f)

## Reproducing results from the litterature

### [Porqueres et al. 2023](https://arxiv.org/abs/2304.04785)
<a href="https://colab.research.google.com/github/EiffL/LPTLensingComparison/blob/main/notebooks/AccuracyTest_Porqueres2023.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Here, we tune our simulator to match the setting of the series of papers led by Natalia Porqueres:
- **Field-level inference of cosmic shear with intrinsic alignments and baryons** https://arxiv.org/abs/2304.04785
- **Lifting weak lensing degeneracies with a field-based likelihood** https://arxiv.org/abs/2108.04825

The main simulation setting in this series of papers uses the following setup:
- **patch size**: 16 x 16 deg^2
- **pixel size**: 15 arcmin^2 (which gives us an lmax of around 1300)
- **galaxy number density**: 30 gals/arcmin^2
- **galaxy ellipticity dispersion**: 0.3
- **lightcone volume**: 1 x 1 x 4.5 Gpc/h
- **lightcone resolution**: 64 x 64 x 128 voxels
  
![image](https://github.com/EiffL/LPTLensingComparison/assets/861591/68ca4572-40bb-4117-90be-52f19c2f81ea)
![image](https://github.com/EiffL/LPTLensingComparison/assets/861591/92f20cf1-c32f-4923-b781-63bde07aa0c3)

We find that this resolution is not quite sufficient for the LPT model to converge to expected linear theory, but note that is a detail, mostly irrelevant to the story of this line of papers.
![image](https://github.com/EiffL/LPTLensingComparison/assets/861591/0a3ddf15-54f9-481a-afef-0290ec490392)

