# LPT Lensing Comparison

The goal of this repo is to provide an open-source and easy-to-use testbed for comparing full-field weak lensing constraints under an LPT gravity model, lognormal convergence model, and gaussian convergence model. 

A guiding principle of this repo is that everything should be runnable within a few hours on colab.

## Description of forward model

We are adopting the following setting:
- patch size: 5 x 5 deg^2
- pixel size: 5 arcmin^2 (which gives us an lmax of around 3000)
- galaxy number density: 27 gals/arcmin^2

![image](https://github.com/EiffL/LPTLensingComparison/assets/861591/24994aeb-87fd-4644-8005-249cd6fbf1c4)

![image](https://github.com/EiffL/LPTLensingComparison/assets/861591/8a068d7e-8df4-4ac8-81e5-00efccd70111)

### Accuracy of LPT model
<a href="https://colab.research.google.com/github/EiffL/LPTLensingComparison/blob/main/notebooks/LPTLensingAccuracyTest.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

We check that our LPT model has sufficient resolution to reproduce the linear theory angular power spectra.
![image](https://github.com/EiffL/LPTLensingComparison/assets/861591/05e19df0-db65-4527-a93e-455f78d66726)
