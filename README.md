# Math and Architectures of Deep Learning

Python code (in the form of Jupyter ipython notebooks) to support the book "**Math and Architectures of Deep Learning**".

Code contributors: **Ananya Ashok, Sujay Narumanchi, Devashish Shankar, Krishnendu Chaudhury**.

This repository contains the example code - mostly in Numpy and PyTorch - corresponding to
the theoretical topics introduced in the book. The code listings are organized in chapters
that correspond to the main book.

## Installation
1. Clone the repository: `git clone https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython.git`
2. Create virtual environment: `virtualenv venv --python=python3` (you may need to do  `pip install virtualenv` first)
3. Activate virtual environment: `source venv/bin/activate` 
4. Change directory: `cd mathematical-methods-in-deep-learning-ipython`
5. Install dependencies: `pip install -r requirements.txt`
6. Navigate to the python directory: `cd python`
7. Start jupyter: `jupyter notebook`

This will redirect you to a browser window with the ipython notebooks 

Note: Ensure to use Python3 to run the notebooks

## Table of Contents

* Chapter 2:
  * [2.2 Intro to Vectors](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.2-vector-numpy-pytorch-intro.ipynb)
  * [2.4 Intro to Matrices, Tensors and Images](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.4-matrix-numpy-pytorch-intro.ipynb)
  * [2.7 Basic Vector and Matrix operations](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.7-transpose-dot-matmul-numpy-pytorch.ipynb)
  * [2.12.5 Solving an overdetermined system using pseudo inverse](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.12.5-overdet-numpy.ipynb)
  * [2.13 Eigenvalues and Eigenvectors](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.13-eig-numpy.ipynb)
  * [2.14 Rotation Matrices](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.14-rotation-numpy.ipynb)
  * [2.15 Matrix Diagonalization](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.15-mat-diagonalization-numpy.ipynb)
  * [2.16 Spectral Decomposition of a Symmetric Matrix](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.16-spectral-decomp-numpy.ipynb)
  * [2.17 Finding the axes of a hyper-ellipse](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch2/2.17-hyper-ellipse-numpy.ipynb)
  
  
* Chapter 3
  * [3.4 Common code for chapter 3](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch3/3.4-common.ipynb)
  * [3.4.1 Gradient Descent](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch3/3.4.1-gradients-numpy-pytorch.ipynb)
  * [3.4.2 Non-linear Models](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch3/3.4.2-gradients-nonlinear-numpy-pytorch.ipynb)
  * [3.4.3 A Linear Model for the cat-brain](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch3/3.4.3-gradients-catbrain-numpy-pytorch.ipynb)
  
* Chapter 4
  * [4.3.2 Common code for chapter 4](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.3.2-common.ipynb)
  * [4.3.2 PCA on synthetic correlated data](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.3.2-pca-numpy.ipynb)
  * [4.3.2 PCA on synthetic uncorrelated data](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.3.2-pca-uncorrelated-numpy.ipynb)
  * [4.3.3 PCA on synthetic correlated non-linear data](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.3.3-pca-nonlinear-numpy.ipynb)
  * [4.4.4 Linear system solving via SVD](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.4.4-svd-linear-system-numpy.ipynb)
  * [4.4.5 PCA computation via SVD](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.4.5-svd-pca-numpy.ipynb)
  * [4.5.3 LSA on a toy dataset](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.5.3-svd-lsa-toy-dataset-numpy.ipynb)
  * [4.5.4 LSA/SVD on a 500 × 3 dataset](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch4/4.5.4-svd-lsa-numpy.ipynb)

* Chapter 5
  * [5.9.1 Uniform Random Distribution](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch5/5.9.1-uniform-random-distribution.ipynb)
  * [5.9.2 Normal Distribution](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch5/5.9.2-normal-distribution.ipynb)
  * [5.9.3 Binomial Distribution](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch5/5.9.3-binomial-distribution.ipynb)
  * [5.9.4 Multinomial Distribution](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch5/5.9.4-multinomial-distribution.ipynb)
  * [5.9.5 Bernoulli Distribution](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch5/5.9.5-bernoulli-distribution.ipynb)
  
* Chapter 6
  * [6.2.2 Entropy](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch6/6.2.2-entropy-gaussian.ipynb)
  * [6.3.1 Cross Entropy](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch6/6.3.1-cross-entropy.ipynb)
  * [6.4.2 KL Divergence](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch6/6.4.2-kullback-leibler-divergence.ipynb)
  * [6.7.1 Model Parameter Estimation](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch6/6.7.1-model-parameter-estimation.ipynb)
  * [6.8.2 Gaussian Mixture Modelling](https://nbviewer.jupyter.org/github/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch6/6.8.2-gaussian-mixture-models.ipynb)

* Chapter 7
  * [7.2.3 Perceptron](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch7/7.2.3-perceptron.ipynb)
  * [7.2.4 Modeling logic gates with perceptrons](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch7/7.2.4-modeling-logic-gates-with-perceptrons.ipynb)
  * [7.4.3 Approximating surfaces with perceptrons](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch7/7.4.3-approximating-surfaces-with-perceptrons.ipynb)

* Chapter 8
  * [8.4 Forward and Backward pass](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch8/8.4-forward-and-backward-pass.ipynb)
  * [8.5 Training a neural network](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch8/8.5-training-a-neural-network.ipynb)

* Chapter 9
  * [9.1 Loss Functions](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch9/9.1-loss-functions.ipynb)
  * [9.2.3 Stochastic Gradient Descent](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch9/9.2.3-stochastic-gradient-descent.ipynb)
  * [9.2.4 Momentum](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch9/9.2.4-momentum.ipynb)

* Chapter 10
  * [10.3.1 2D Convolution for image smoothing](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch10/10.3.1-2dconv-image-smoothing.ipynb)
  * [10.3.2 2D Convolution for edge detection](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch10/10.3.2-2dconv-edge-detection.ipynb)
  * [10.4.1 3D Convolution for motion detection](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch10/10.4.1-3dconv-motion-detection.ipynb)
  * [10.5.3 Transpose Convolution for upsampling](https://github.com/krishnonwork/mathematical-methods-in-deep-learning-ipython/blob/master/python/ch10/10.5.3-transpose-conv-upsampling.ipynb)
