# Digital Image Processing - IUST - Fall 2019
The assignments of Digital Image Processing course taught by [Dr. Mohammadi](http://webpages.iust.ac.ir/mrmohammadi/index.html) at [Iran Science and Technology University](http://iust.ac.ir)

## Contents
You can find explanation in each folder. In this section, there is just summary of each work folder:

### HW01
In this folder, following topics have been covered:
* An idea of using computer vision in real life
* Impact of Shutter Speed on image quality
* HIstograrm Equalization and its effects

### HW02
In this work, following topics have been covered:
* Padding of images
* Separable Kernels and the corresponding speed up
* Computing norm of vectors on paper

### HW03
In this work, following topics have been covered:
* Explanation of frequency spectrum (DFT and DCT) for stripped images in different directions
* Analysis of Squared Cumulative Sum of Sorted brightness and frequency spectrum of two images
* Reducing noise of stripped image using DCT and DFT
* Obtaining 2D DCT from 1D FFT on paper and comparing to the `cv2` functions

### HW04 [WIP]
In this folder, following topics have been covered:
* Difference between DCT and Wavelet
* Computing coefficients of Haar wavelet
* Decomposing a signal using Haar wavelet
* Reducing Noise using Haar wavelet

### HW05
In this work, following topics have been covered:
* Implementation of Canny Edge Detector from scratch and comparing its result with `cv2`
* Improvement of Hough Transform to use gradient directions in voting step [WIP]
* CamScanner clone:
	* Finding Lines using LineSegmentDetector and HoughLines
	* Finding Corners of a rectangle by finding intersection of lines
	* Finding Homography between two images
	* WarpPerspective the found rectangle to image domain corners.

### HW06
In this work, following topics have been covered:
* Ellipse Specific Fitting paper:
	* Summary of it
	* Implementing its approach
	* Showing result on images of a circle and ellipse
* Computing number of operation using RANSAC approach
* Implementation of Ellipse Specific Fitting with RANSAC and demonstrating results

### HW07
In this work, following topics have been covered:
* Comparison and summarization of AAM and CLM papers [WIP]
* Detecting Face Landmarks using `dlib` and `cv2`

### HW08
In this work, following topics have been covered:
* Compressing a image with a few unique values using Huffman coding
* Compare different results of JPEG compression on image of uniform distribution of constant of half constant using RMS and compression Ratio
* Implementation of JPEG
	* Patch
	* Shift
	* DCT
	* Quantization
	* Zig-Zag
	* Binary
* Summary of Multiresolution Segmentation Algorithm

### HW09
In this work, following topics have been covered:
* Determining Morphological Operation and Structural Element type for few instances
* Drawing result of some morph operations on few images
* Hit or Miss operation
* Comparison of Local Binary Pattern and Soft Local Binary Pattern
* Train and evaluation for Farsi Handwritten Digit Recognition using:
	* Shape and Texture for Feature Extraction
	* SVM, kNN, Trees, etc for Classification
	* Confusion Matrix for evaluation
	* Comparison

### HW10
In this work, following topics have been covered:
* Benefits of CNNs over MLPs for image processing
* Benefits of Pooling Operation
* Role of non-linear activation function
* Comparison of number of parameters in GoogLeNet (22 layers) with AlexNet (8 layers)
* Summary of Xception model
	* LeNet
    * AlexNet
    * VGG
    * ResNet
    * Inception (GoogleNet)
    * Xception
* Train a Keras model on CIFAR10 library and report results
	* Normalization
	* Learning rate decay
	* ResNet101V2
	* Cutout regularization
	* Results

### HW11
In this work, following topics have been covered:
* Definition and compare Semantic Segmentation, Object Detection and Instance Segmentation
* Compare RCNN, Fast RCNN and Faster RCNN
	* RCNN
	* Fast RCNN
	* Faster RCNN
* Template Matching for reading car plates [WIP]


### HW12
In this work, following topics have been covered:
* Some info about SSD300 and VGG19 in term of number of parameters
* Computation of AR25 and AR50 for some ground truth and predicted anchor boxes
* Training a model similar to SSD300 for object detection [WIP]

### HW13
In this work, following topics have been covered:
* Dilated Convolutions, use cases and cons and pros
* Summary of MarkRCNN paper and example of a public implementation of it
* Computing number of parameters in each layer for two particular networks

### HW14
In this work, following topics have been covered:
* Summary of CycleGAN paper
* Comparison of PCA and Autoencoder
* Training a GAN on FashionMNIST using Keras

### Project [WIP]
In this work, Self Supervised Learning for Image Representation has been discussed. Finally, Feature Decoupling method has been adopted but for a discrete amount of rotations (only 0, 45, 90, 180) as the supervision signal on MNIST dataset.
The code and report is not currently available in this repository and will be added as soon as possible (my teammates need to help!).

## Acknowledgment
I have used many contents from different websites which I have not referenced! I have bookmarked all those references and I will update the slides or at least provide the links at each folder.<br>
This course was the only course that I enjoyed during studying my MSc.

## Course Content
Slides can be shared upon request (only in Farsi, not English). Please send me an email (available on [nikronic.github.io](nikronic.github.io) or if you could not find it, contact me by creating an issue.
Videos are are not available.

## Citation
- [ ] Add citation
- [ ] create zenodo page
