
Valda is a Python package for data valuation in machine learning. If you are interested in 

- analyzing the contribution of individual training examples to the final classification performance, or 
- identifying some noisy examples in the training set, 

you may be interested in the functions provided by this package.


The current version (v0.1.5) supports five different data valuation methods 

- Leave-one-out (LOO), 
- Data Shapley with the TMC algorithm (TMC-Shapley) from [Ghorbani and Zou (2019)](https://proceedings.mlr.press/v97/ghorbani19c.html), 
- Beta Shapley from [Kwon and Zou (2022)](https://arxiv.org/abs/2110.14049)
- Class-wise Shapley (CS-Shapley) from [Schoch et al. (2022)](https://arxiv.org/abs/2211.06800)
- Influence Function (IF) from [Koh and Liang (2017)](https://arxiv.org/abs/1703.04730)
  - IF only works with the classifiers built with PyTorch, because it requires gradient computation.


It supports all the classifiers from Sklearn for valuation, and also user-defined classifier using PyTorch. 

## Tutorial

Please checkout a simple tutorial on [Google Colab](https://colab.research.google.com/drive/1agsMNqZan-3RnJLQtBGATRHHWYMe7C9H?usp=sharing), for how to use this package. 


## Installation

For running it on local machines, Valda requires Python 3.6+. It is available through `pip`

```
pip install valda
```
or you can download from the UVa ILP group github via [this link](https://github.com/uvanlp/valda).

## Contributors

- [Yangfeng Ji](https://yangfengji.net/)
- [Stephanie Schoch](https://stephanieschoch.com/)
