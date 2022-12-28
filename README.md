# Convolutional Neural Network (CNN) for Image Classification
This repository contains the implementation of CNN for MNIST Image classification.
Following will be the detailed article to understand the CNN architecture and working better, followed with brief explanation of the implementation.
The code is built on [Kaggle](https://www.kaggle.com/code/nikitgoku/cnn-mnist-classification-pytorch-implementatio)

# Contents
1. Introduction
2. Inside Convolutional Neural Network

# Introduction
Convolutional Neural Network have become the most vividly used algorithm in Computer Vision and other implementation with many noted researches following the CNN architecture. Introduced during the 1980s, popularized in early 2010s, CNN attempted to make deeper and complicated networks thereby achieving higher accuracy. Convolutional Neural Network aka 'ConvNet' consist of multiple layers to process and extract features from data which has a known grid-like topology (images), because of which CNN is mostly used for image processing and object detection.
This seems overwhelming, but convolution is nothing but element wise multiplication of two matrices.

 i. Take two matrices
    
 ii. Multiply them elementwise [Hadamard Product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)#:~:text=In%20mathematics%2C%20the%20Hadamard%20product,elements%20i%2C%20j%20of%20the)
   
 iii. Sum the elements together.
  
Now, why matrices? Just to drop off a simple explanation, images are nothing but matrix of pixel values. A grayscale image has only one matrix but a coloured image (ex. RGB) has 3 matrices stacked over each other with each matrix representing specific colour palette, combining to form an image.

![image](https://user-images.githubusercontent.com/114753615/209846362-f2ff504b-7b89-4c89-8d08-71a700939a3b.png)

The multiple layers contained in a CNN represent different actions with there specific tasks consolidated for processing images.

# Inside Convolutional Neural Network
The CNN is trained to understand the image better, where it is successful in grabbing the spatial dependencies and work on blurring, sharpening and edge detection. CNN works in a way that given an image matrix, use a small matrix called the kernel or the convolution matrix, perform matrix multiplication to fit an image by reducing the number of parameters. The objective of this process is to flatten a given matrix into a vector wherein the vector contains all the high-level features of the image.
## Convolutional Layer
  ### The Kernel
  As seen in the image, we shift the kernel from left-right and top-bottom along the original image, at each step we perform the matrix multiplication between kernel and the portion of image (seen in the yellow highlighted area). 
  The convolved area is the result of adding the elements during the multiplication.
![Kernal Operation](https://user-images.githubusercontent.com/114753615/209853202-f10693bc-1315-485b-a072-7ab6063ba939.gif)
  ### The Stride
  The kernel moves across the image with a certain stride value![Convolution Operation](https://user-images.githubusercontent.com/114753615/209854699-f2815813-7599-4580-a2a2-5e166ceba79c.gif)
  In case of multi-channel image (e.g., RGB), kernel follows the same depth as of the image with n kernels for n depth.
  ### Padding
  Padding is required for dimensionality reduction with which we can either increase or decrease the dimension depending on the layers required demonstrating adaptability in the architecture. Padding has a big role in extracting high-level and low-level features. As seen in the image, neutral values are added across the border of the image in order to help with dimensionality reduction.
![Stride operation](https://user-images.githubusercontent.com/114753615/209855944-e5ea4b27-e47a-4248-8397-b3a91334b160.gif)
  ## Activation
  Activation is used for effective training, where the negative values are mapped to zero thereby maintaining the positive values. This is used to carry identified features to be carried to the next layer
  ### Pooling
  These three methods are associated with the convolutional layer, however this layer is followed by a Pooling Layer which is responsible for extracting dominant feature and reducing spatial resolution. Pooling layer acts as a transition layer between two contiguous convolutional layers.

## Fully-Connected Layer
The convolution layer and the pooling layer forms a single layer in a multi-layer neural network where next comes the fully connected layer aka classification layer. Similar to the multi-layer perceptron, this layer outputs k dimensional vector used for image classification where k represents the number of classes being predicted. Images are classified using a Softmax Classification Technique.
![image](https://user-images.githubusercontent.com/114753615/209862696-4bb2f10e-6887-45dd-8848-4b3c508c35f8.png)


# References
[An Introduction to Convolutional Neural Networks by Keiron O'Shea, Ryan Nash](https://arxiv.org/abs/1511.08458)

[Densely Connected Convolutional Networks by Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger](https://arxiv.org/abs/1608.06993)

[Convolutions with OpenCV and Python by Adrian Rosebrock](https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/)

