# Guide to Teaching the Fundamentals of Convolutional Neural Networks

This repository is a quick guide to teaching the fundamentals of convolutional neural networks from basic neural network and perceptron design up to convolutional and pooling layers. It also includes bits of residual learning and a quick overview of many popular networks that have been used in transfer learning problems recently. The key libraries used in building these programs are [Pytorch](https://pytorch.org/) and [Numpy](https://numpy.org/).

### Background Information:
Several key concepts are expected to be understood when going through this tutorial. Mathematically, as is expected with any machine learning and deep learning adventures, you must have at least a brief understanding of:
- Linear Algebra
- Partial Derivatives
- Statistical Distributions

In addition to these mathematical backgrounds, it is also important to have a decent understanding of programming. In this tutorial we make use of python and its very robust library support. Specifically, we use:
- Pytorch
- Numpy
- Matplotlib
- Jupyter Notebooks - While jupyter notebooks are nice, they are mearly used as a teaching tool. **Note: Github has issues showing all of the markdown (especially LaTeX formatting) in Jupyter Notebook sometimes. Because of this, we recommend forking or cloning the repo and firing it up on your local machine instead to see all of the markdown.** This notebook was also built using python virtual environments so installing different variants would not pollute the namespace.

Finally, as deep learning has grown and expanded, it is hoped that a basic understanding of some machine learning techniques are understood. Upon writing this tutorial for a class, the previously taught concepts include:
- Logistic Regression
- PCA Dimensionality Reduction
- Linear Regression
- Loss Function Convexivity
- Cross Entropy

These are not necessarily utilized in the lessons here, but they are background knowledge the class had before going into this tutorial.


### Key Concepts:
Overall, this tutorial covers:
- Forward Pass of a Perceptron
- Backpropagation for a Fully Connected Neural Network
- Convolutional Filters
- Convolutional Layers
- Strides
- Transfer Learning
- Pooling Layers
- ConvNets as Feature Extractors
- Cross-Correlation vs. Convolutions
- Various types of ConvNets
- Residuals

### Step-By-Step:
1. On a very high level, go over the purpose of a ConvNet (and NNs as a whole) and show the documentation PyTorch has provided. Have students build a simple NN from the PyTorch example, following step-by-step, and prove to them they can do it on Day 1. It is ok if its not clear, it is more just a way to show them where they are going. This is done in [Getting Started Notebook](Getting&#32;Started.ipynb)
    - Make sure to introduce them to PyTorch and have them start on the [60 Minute Blitz Tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html). It will help them build confidence and while it might not all make sense, the docs over there are incredibly good at explaining what is happening.
    - Also, if avaliable, assign the simple videos from the [Coursera Course](https://www.coursera.org/learn/convolutional-neural-networks) as this will help them think in terms of convolutions. It won't really connect everything until step 5, but it will make the transition easy.
2. Explain how the Forward Pass works in a fully connected neural network. This is done in [Backprop and Convolution vs Cross-Correlation](Backprop&#32;and&#32;Convolution&#32;vs&#32;Cross-Correlation.ipynb). It is ok if you don't get through this entire notebook. Instead emphasize the simplicity of using linear algebra and doing matrix multiplcation through the system. 
3. Explain Backpropagation. It is a bit complex, and since it requires partial derivatives and backtracking through the forward pass, make sure they have a solid understanding of the forward pass. If you want, use the exercise question at the bottom to work through the forward pass before getting to this step. These equations were pulled from [this textbook](https://www.amazon.com/Fundamentals-Computational-Intelligence-Evolutionary-Computation/dp/1119214343). 
4. After backprop, explain what a convolution is using [Origins of Convolutions & Kernels](Origins&#32;of&#32;Convolutions&#32;&&#32;Kernels.pdf). This should get them thinking about the idea of applying it to a Neural Network. Slowly introduce that this idea can be applied in regards to Neural Networks.
    - Have them do [Quiz 1](Quiz&#32;1.ipynb) More General Questions. **DO NOT DO SPECIFIC CONVNET QUESTIONS YET**. You can have them continously change the inputs, initial weights, or labels for Q2 if you want as well, the output is given from the code at the bottom 
    - They should be through the first 9 tutorials in the PyTorch Tutorials. This would take you up to, but not including, Tensorflow: Static Graphs. 
5. Introduce the Convolutional Neural Network. No Jupyter Notebook page was developed for this because there are countless articles online. The one chosen to explain it to the class in this situation was [this one](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53). Go through it slowly and simply name the networks at the bottom.
    - Hopefully at this point, they've gotten through the Coursera videos to the point where convolutions make sense and the transition isn't too bad from fully connected to ConvNets. 
6. Go over some simple activation function differences in [Activations](Activations.ipynb). These are very basic and just a PyTorch implementation. This does not need to be a standalone component unless you want to get into the math. Otherwise, it is just important that these ideas and terms have been discussed. 
7. Go over a couple examples of various CNNs. You don't need to explain in detail since they haven't gotten things like residuals quite yet, but show how these networks can be applied to various datasets in [Pytorch through transfer learning](https://pytorch.org/docs/stable/torchvision/models.html). [Here](https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d) is a good article covering 10 different ConvNets.
    - Finish up the second part of [Quiz 1](Quiz&#32;1.ipynb), Specific ConvNet Questions.
8. Finally, cover the newer concepts you want to, whether it be dynamic kernels, residuals, LSTM-FCNs, etc. In this class, we made it residuals so that we could work towards ResNet as a basis for ConvNets to work on during the final projects. There is obviously a lot of room to grow, and hopefully by this point they have worked through the PyTorch tutorials and can begin messing around with them. 

Spend as much time as necessary on each step, and don't be afraid to go back to a subject multiple times. Sometimes it takes it, and while its not difficult to understand once you've done it, first learning it can be a pain. 

### Exercises:
- [Quiz 1](Quiz&#32;1.ipynb) is used to evaluate everything from 1-4. If something isn't clicking, please re-iterate the point. The main exercise is a simple perceptron forward pass and backprop walkthrough.
- [Coursera Course](https://www.coursera.org/learn/convolutional-neural-networks). Very good at explaining convolutions and how feature extraction can work. Used as a supplementary tool.
- [PyTorch tutorial](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html). This is really good at getting them up and running. They should be doing this a lesson or two each day (takes maybe 5-10 minutes to read through and execute the code) just to stay up to date.
