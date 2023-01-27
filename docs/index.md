
Valda is a Python package for data valuation in machine learning. If you are interested in 

- analyzing the contribution of individual training examples to the final classification performance, or 
- identifying some noisy examples in the training set, 

you may be interested in the functions provided by this package.


The current version (v0.1.5) supports five different data valuation methods 

- Leave-one-out (LOO), 
- Data Shapley with the TMC algorithm (TMC-Shapley), 
- Beta Shapley, 
- Class-wise Shapley (CS-Shapley), and 
- Influence Function


It also support all the classifiers from Sklearn for valuation, and also user-defined classifier using PyTorch. 

Please checkout a simple tutorial on [Google Colab](https://colab.research.google.com/drive/1agsMNqZan-3RnJLQtBGATRHHWYMe7C9H?usp=sharing), for how to use this package. 


## Setup

You should be running Python 3.6+ to use this package. Valda is available through `pip`

```
pip install valda
```
or you can download from the UVa ILP group github via [this link](https://github.com/uvanlp/valda).

## Contributors

- [Yangfeng Ji](https://yangfengji.net/)
- [Stephanie Schoch](https://stephanieschoch.com/)
