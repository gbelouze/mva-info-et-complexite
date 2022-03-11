# Kili  Data Challenge

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Code for the Kili challenge of https://challengedata.ens.fr/, 2022 edition

## Overview

### Challenge context
As machine learning models go into production, practioners often realize that the data matters more than the model. Until now, AI research focus was more on models than data. However, rather than spending time on building bigger architectures, testing or building new fancy models, it is often a better use of time to iterate on the datasets. Andrew Ng sparked this model-centric to data-centric AI paradigm shift a few months ago (see this youtube video). Kili embodies this shift, by providing engineers the tools to work on the data. Iterating on a dataset can mean : cleaning label errors, cleaning domain errors, pre processing the data, identifying hidden sub-classification, carefully augmenting the data on which the model performs worse, generating new data, sourcing new data, etc... To our knowledge, this challenge is the first NLP data centric competition, after the first computer vision challenge this summer. Organizations are starting to pick up this movement, by using tools like Kili Technology to iterate and understand their dataset better, which help achieve the expected business performance.

### Challenge goals
The goal of the challenge is to predict the sentiment (positive or negative) of movie reviews. The interest of the challenge lies in the training pipeline being kept fixed. You won't be able to choose the model to use, or have to create complex ensembles of models that add no real value. Instead, you will have to select, augment, clean, generate, source, etc… a training dataset starting from a given dataset. Actually you will be allowed to give anything to the model. To allow you to iterate fast on your experiments, we provide you with the training script, which uses a rather simple model, fastText. Your submissions (ie, the training dataset) will be used to train the model on our servers, which will then be tested on a hidden test set of movie reviews. We reveal a small fraction of this test set (10 texts) to give a sense of the test data distribution.

## Usage

We advise to create a python virtual environment. Then in this venv run

```bash
pip install -e .
```

This additionnaly installs a command line interface `kili`. See available commands with

```bash
kili --help
```

## Contribute

Follow the steps from [CONTRIBUTING.md](CONTRIBUTING.md)
