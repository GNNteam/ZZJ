## DKGCN/MKGCN
This is the PyTorch implementation for MKGCN proposed in the paper [Multi-view knowledge graph convolutional networks for recommendation](https://www.sciencedirect.com/science/article/abs/pii/S1568494624014078), Applied Soft Computing, 2024.

![Uploading 1.png…]()

### 1. Note on datasets and files
Due to the large size of datasets Last.FM, Book-Crossing and MovieLens20M., we did not upload the source files directly. Please download through other reliable channels.

Before executing the codes, make sure that the data set and all.py files are properly placed in the project root directory.

### 2. Running environment
We develope our codes in the following environment:

  Python version 3.9.12
  torch==1.12.0+cu113
  numpy==1.21.5
  tqdm==4.64.0

### 3. How to run the codes
For each specified data set path:

' ''
  python main.py 
' ''



