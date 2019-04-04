# Convolutional-Neural-Networks-for-Text-Classification
Implementation of CNN for Text Classification

# Getting Started
This repository consists of the implementation of the paper https://arxiv.org/pdf/1408.5882.pdf. I have written code seperately for each of the datasets so that the results can be viewed separately.

This model is tested as of now for 3 datasets and instructions to run models for them are given below.

# Changes made
I have implemented the architectures as described in the paper but with the following changes.

1. I used glove 42B pretrained vectors instead of google's pretrained vectors.

2. Used Adam as optimizer as opposed to SGD mentioned in the paper.

# Packages required

1. python 3.6
2. tensorflow>=1.9.0
3. keras>=2.2.0
4. Glove pre trained vectors(glove.42B.300d.zip)(Download,Extract and keep in Models folder)
5. Nltk 

# How to run

1. Clone this repository

```
git clone https://github.com/avinashsai/Convolutional-Neural-Networks-for-Text-Classification.git
```
2. Change the directory

```
cd Models
```

3. Run model for individual datasets using

```
python rt_polarity.py

python sub_obj.py

python trec.py

```
