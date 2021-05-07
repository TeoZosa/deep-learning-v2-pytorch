Deep Learning v2 PyTorch
==============================
![CI](https://github.com/TeoZosa/deep-learning-v2-pytorch/workflows/CI/badge.svg)
![codecov](https://codecov.io/gh/TeoZosa/deep-learning-v2-pytorch/branch/master/graph/badge.svg?token=3HF21UWY82)
![License](https://img.shields.io/github/license/TeoZosa/deep-learning-v2-pytorch?style=plastic)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/deep-learning-v2-pytorch?style=plastic)
![PyPI](https://img.shields.io/pypi/v/deep-learning-v2-pytorch?color=informational&style=plastic)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![powered by semgrep](https://img.shields.io/badge/powered%20by-semgrep-1B2F3D?labelColor=lightgrey&link=https://semgrep.dev/&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAA0AAAAOCAYAAAD0f5bSAAAABmJLR0QA/gD+AP+cH+QUAAAACXBIWXMAAA3XAAAN1wFCKJt4AAAAB3RJTUUH5AYMEy0l8dkqrQAAAvFJREFUKBUB5gIZ/QEAAP8BAAAAAAMG6AD9+hn/GzA//wD//wAAAAD+AAAAAgABAQDl0MEBAwbmAf36GQAAAAAAAQEC9QH//gv/Gi1GFQEC+OoAAAAAAAAAAAABAQAA//8AAAAAAAAAAAD//ggX5tO66gID9AEBFSRxAgYLzRQAAADpAAAAAP7+/gDl0cMPAAAA+wAAAPkbLz39AgICAAAAAAAAAAAs+vU12AEbLz4bAAAA5P8AAAAA//4A5NDDEwEBAO///wABAQEAAP//ABwcMD7hAQEBAAAAAAAAAAAaAgAAAOAAAAAAAQEBAOXRwxUAAADw//8AAgAAAAD//wAAAAAA5OXRwhcAAQEAAAAAAAAAAOICAAAABP3+/gDjzsAT//8A7gAAAAEAAAD+AAAA/wAAAAAAAAAA//8A7ePOwA/+/v4AAAAABAIAAAAAAAAAAAAAAO8AAAABAAAAAAAAAAIAAAABAAAAAAAAAAgAAAD/AAAA8wAAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAA8AAAAEAAAA/gAAAP8AAAADAAAA/gAAAP8AAAAAAAAAAAAAAAACAAAAAAAAAAAAAAAAAAAA7wAAAPsAAAARAAAABAAAAP4AAAAAAAAAAgAAABYAAAAAAAAAAAIAAAD8AwICAB0yQP78/v4GAAAA/wAAAPAAAAD9AAAA/wAAAPr9//8aHTJA6AICAgAAAAD8AgAAADIAAAAAAP//AB4wPvgAAAARAQEA/gEBAP4BAQABAAAAGB0vPeIA//8AAAAAAAAAABAC+vUz1QAAAA8AAAAAAwMDABwwPu3//wAe//8AAv//ABAcMD7lAwMDAAAAAAAAAAAG+vU0+QEBAvUB//4L/xotRhUBAvjqAAAAAAAAAAAAAQEAAP//AAAAAAAAAAAA//4IF+bTuuoCA/QBAQAA/wEAAAAAAwboAP36Gf8bMD//AP//AAAAAP4AAAACAAEBAOXQwQEDBuYB/foZAAAAAAD4I6qbK3+1zQAAAABJRU5ErkJggg==)](https://semgrep.dev/)
[![Dependabot](https://api.dependabot.com/badges/status?host=github&repo=TeoZosa/deep-learning-v2-pytorch)](https://dependabot.com/)


---

**Documentation**: [https://deep-learning-v2-pytorch.readthedocs.io](https://deep-learning-v2-pytorch.readthedocs.io)

**Source Code**: [https://github.com/TeoZosa/deep-learning-v2-pytorch](https://github.com/TeoZosa/deep-learning-v2-pytorch)

---

Overview
--------
Fork of [udacity/deep-learning-v2-pytorch](https://github.com/udacity/deep-learning-v2-pytorch)

------------

Table of Contents

<!-- toc -->

  * [Tutorials](#tutorials)
    + [Introduction to Neural Networks](#introduction-to-neural-networks)
    + [Convolutional Neural Networks](#convolutional-neural-networks)
    + [Recurrent Neural Networks](#recurrent-neural-networks)
    + [Generative Adversarial Networks](#generative-adversarial-networks)
    + [Deploying a Model (with AWS SageMaker)](#deploying-a-model-with-aws-sagemaker)
    + [Projects](#projects)
    + [Elective Material](#elective-material)
- [Development](#development)
  * [Package and Dependencies Installation](#package-and-dependencies-installation)
  * [Testing](#testing)
  * [Code Quality](#code-quality)
    + [Automate via Git Pre-Commit Hooks](#automate-via-git-pre-commit-hooks)
  * [Documentation](#documentation)
- [Summary](#summary)
- [Further Reading](#further-reading)
- [Legal](#legal)
  * [License](#license)
  * [Credits](#credits)

<!-- tocstop -->

Tutorials
---------

### Introduction to Neural Networks

* [Introduction to Neural Networks](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-neural-networks): Learn how to implement gradient descent and apply it to predicting patterns in student admissions data.
* [Sentiment Analysis with NumPy](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-analysis-network): [Andrew Trask](http://iamtrask.github.io/) leads you through building a sentiment analysis model, predicting if some text is positive or negative.
* [Introduction to PyTorch](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-to-pytorch): Learn how to build neural networks in PyTorch and use pre-trained networks for state-of-the-art image classifiers.

### Convolutional Neural Networks

* [Convolutional Neural Networks](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/convolutional-neural-networks): Visualize the output of layers that make up a CNN. Learn how to define and train a CNN for classifying [MNIST data](https://en.wikipedia.org/wiki/MNIST_database), a handwritten digit database that is notorious in the fields of machine and deep learning. Also, define and train a CNN for classifying images in the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
* [Transfer Learning](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/transfer-learning). In practice, most people don't train their own networks on huge datasets; they use **pre-trained** networks such as VGGnet. Here you'll use VGGnet to help classify images of flowers without training an end-to-end network from scratch.
* [Weight Initialization](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/weight-initialization): Explore how initializing network weights affects performance.
* [Autoencoders](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/autoencoder): Build models for image compression and de-noising, using feedforward and convolutional networks in PyTorch.
* [Style Transfer](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/style-transfer): Extract style and content features from images, using a pre-trained network. Implement style transfer according to the paper, [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) by Gatys et. al. Define appropriate losses for iteratively creating a target, style-transferred image of your own design!

### Recurrent Neural Networks

* [Intro to Recurrent Networks (Time series & Character-level RNN)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/recurrent-neural-networks): Recurrent neural networks are able to use information about the sequence of data, such as the sequence of characters in text; learn how to implement these in PyTorch for a variety of tasks.
* [Embeddings (Word2Vec)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/word2vec-embeddings): Implement the Word2Vec model to find semantic representations of words for use in natural language processing.
* [Sentiment Analysis RNN](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-rnn): Implement a recurrent neural network that can predict if the text of a moview review is positive or negative.
* [Attention](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/attention): Implement attention and apply it to annotation vectors.

### Generative Adversarial Networks

* [Generative Adversarial Network on MNIST](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/gan-mnist): Train a simple generative adversarial network on the MNIST dataset.
* [Batch Normalization](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/batch-norm): Learn how to improve training rates and network stability with batch normalizations.
* [Deep Convolutional GAN (DCGAN)](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/dcgan-svhn): Implement a DCGAN to generate new images based on the Street View House Numbers (SVHN) dataset.
* [CycleGAN](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/cycle-gan): Implement a CycleGAN that is designed to learn from unpaired and unlabeled data; use trained generators to transform images from summer to winter and vice versa.

### Deploying a Model (with AWS SageMaker)

* [All exercise and project notebooks](https://github.com/udacity/sagemaker-deployment) for the lessons on model deployment can be found in the linked, Github repo. Learn to deploy pre-trained models using AWS SageMaker.

### Projects

* [Predicting Bike-Sharing Patterns](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-bikesharing): Implement a neural network in NumPy to predict bike rentals.
* [Dog Breed Classifier](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification): Build a convolutional neural network with PyTorch to classify any image (even an image of a face) as a specific dog breed.
* [TV Script Generation](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-tv-script-generation): Train a recurrent neural network to generate scripts in the style of dialogue from Seinfeld.
* [Face Generation](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-face-generation): Use a DCGAN on the CelebA dataset to generate images of new and realistic human faces.

### Elective Material

* [Intro to TensorFlow](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/tensorflow/intro-to-tensorflow): Starting building neural networks with TensorFlow.
* [Keras](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/keras): Learn to build neural networks and convolutional neural networks with Keras.


Development
===========

> 📝 **Note**  
>  For convenience, many of the below processes are abstracted away
>  and encapsulated in single [Make](https://www.gnu.org/software/make/) targets.


> 🔥 **Tip**  
>  Invoking `make` without any arguments will display
>  auto-generated documentation on available commands.

Package and Dependencies Installation
--------------------------------------

Make sure you have Python 3.6+ and [`poetry`](https://python-poetry.org/)
installed and configured.

To install the package and all dev dependencies, run:
```shell script
make provision_environment
```

> 🔥 **Tip**  
>  Invoking the above without `poetry` installed will emit a
>  helpful error message letting you know how you can install poetry.

Testing
------------

We use [`tox`](https://tox.readthedocs.io/en/latest/) for our test automation framework
and [`pytest`](https://pytest.readthedocs.io/) for our testing framework.

To invoke the tests, run:

```shell script
make test
```

Run [mutation tests](https://opensource.com/article/20/7/mutmut-python) to validate test suite robustness (Optional):

```shell script
make test-mutations
```

> 📝 **Note**  
>  Test time scales with the complexity of the codebase. Results are cached
>  in `.mutmut-cache`, so once you get past the initial [cold start problem](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)),
>  subsequent mutation test runs will be much faster; new mutations will only
>  be applied to modified code paths.

Code Quality
------------

We are using [`pre-commit`](https://pre-commit.com/) for our code quality
static analysis automation and management framework.

To invoke the analyses and auto-formatting over all version-controlled files, run:

```shell script
make lint
```

> 🚨 **Danger**  
>  CI will fail if either testing or code quality fail,
>  so it is recommended to automatically run the above locally
>  prior to every commit that is pushed.

### Automate via Git Pre-Commit Hooks

To automatically run code quality validation on every commit (over to-be-committed
files), run:

```shell script
make install-pre-commit-hooks
```

> ⚠️ Warning  
>  This will prevent commits if any single pre-commit hook fails
>  (unless it is allowed to fail)
>  or a file is modified by an auto-formatting job;
>  in the latter case, you may simply repeat the commit and it should pass.

Documentation
--------------

```shell script
make docs-clean docs-html
```

> 📝 **Note**  
>  For faster feedback loops, this will attempt to automatically open the newly
>  built documentation static HTML in your browser.

---

Legal
=====

License
-------

Deep Learning v2 PyTorch is licensed under the Apache License, Version 2.0.
See [LICENSE](./LICENSE) for the full license text.


Credits
-------

This project was generated from
[`@TeoZosa`'s](https://github.com/TeoZosa)
[`cookiecutter-cruft-poetry-tox-pre-commit-ci-cd`](https://github.com/TeoZosa/cookiecutter-cruft-poetry-tox-pre-commit-ci-cd)
template.
