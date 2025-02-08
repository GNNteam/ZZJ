## DKGCN/MKGCN
This is the PyTorch implementation for MKGCN proposed in the paper [Multi-view knowledge graph convolutional networks for recommendation](https://www.sciencedirect.com/science/article/abs/pii/S1568494624014078), Applied Soft Computing, 2025.

### 1. Note on datasets and files
Due to the large size of datasets Last.FM, Book-Crossing and MovieLens20M., we did not upload the source files directly. Please download through other reliable channels. The dataset will pass through data_loader.py to read.

Before executing the codes, make sure that the data set and all.py files are properly placed in the project root directory.

### 2. Running environment
We develope our codes in the following environment:

  Python version 3.8.19
 
  torch==2.0.1+cu118
  
  numpy==1.24.3
  
  psutil==6.0.0

  tensorboardX==2.6.2.2

  scikit-learn==1.3.2

### 3. How to run the codes
For each specified data set path:

-Training:

``` Python
  python train.py 
```
-Testing:

``` Python
  python KGCN.py 
```
### 4. About configurable arguments


The model parameters are configured in KGCN using the parse.py script.


### 5. Citing our paper

```
@article{WANG2025112633,
title = {Multi-view knowledge graph convolutional networks for recommendation},
journal = {Applied Soft Computing},
volume = {169},
pages = {112633},
year = {2025},
doi = {https://doi.org/10.1016/j.asoc.2024.112633}
}
```

