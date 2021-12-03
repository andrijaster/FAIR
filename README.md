# FAIR

Fair Adversarial Instance Re-weighting (Petrovic et al. [https://arxiv.org/abs/2011.07495](https://arxiv.org/abs/2011.07495))

## Overview

Here we provide the implementation of a Fair Adversarial Instance Re-weighting (FAIR) algotithm in PyTorch, along with a minimal execution examples (on the aif360 dataset). Besides four different variants of FAIR (FAIR-scalar, FAIR-betaREP, FAIR-betaSF and FAIR-Bernolli) we provided three different baselines: 
- Learning Unbiased Representations via Mutual Information backpropagation [https://arxiv.org/abs/2003.06430](https://arxiv.org/abs/2003.06430)
- Conditional Learning of Fair Representations [https://arxiv.org/abs/1910.07162](https://arxiv.org/abs/1910.07162)
- Fair Adversial Discriminative model [http://mlg.eng.cam.ac.uk/adrian/AAAI2019_OneNetworkAdversarialFairness.pdf](http://mlg.eng.cam.ac.uk/adrian/AAAI2019_OneNetworkAdversarialFairness.pdf)

The repository is organised as follows:
- `data/` contains the folders where results and trained models are going to be stored;
- `src/models` contains the implementation of the FAIR models and baselines;
- `src/utilities.py` contains a utilitise for model training and evaluation;
- `main` contains script for starting training and evaluation


## Dependencies

The script has been tested running under Python 3..2, with the following packages installed along with the dependencies that are packed in `requirements.txt`. 

In addition, CUDA 11.2 is used.

### Conda

Conda environment is available at the `environment.yml`.

## Starting

The code can be started with the 
```
python3 main.py 
```

## Reference  

If you make advantage of the FAIR model in your research, please cite the following in your manuscript:


```
@article{petrovic2020fair,
  title={FAIR: Fair Adversarial Instance Re-weighting},
  author={Petrovi{\'c}, Andrija and Nikoli{\'c}, Mladen and Radovanovi{\'c}, Sandro and Deliba{\v{s}}i{\'c}, Boris and Jovanovi{\'c}, Milo{\v{s}}},
  journal={arXiv preprint arXiv:2011.07495},
  year={2020}
}
```

## LICENSE
MIT


