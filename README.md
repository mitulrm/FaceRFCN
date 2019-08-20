# Keras-RFCN
RFCN implement based on Keras&amp;Tensorflow

This is an implementation of [FaceRFCN - Detecting Faces Using Region-based Fully Convolutional Networks](https://arxiv.org/abs/1709.05256) on Python 3, Keras, and TensorFlow. The model generates bounding boxes for each face in the image. It's based on Feature Pyramid Network (FPN) and a [ResNet50](https://arxiv.org/abs/1512.03385) or ResNet101 backbone.

We took (https://github.com/parap1uie-s/Keras-RFCN) as our base code and implemented position sensitive average pooling and Online Hard Example Mining as described in FaceRFCN paper.

This repository includes model code, training and testing code for WiderFace dataset.
