#!/bin/bash

#################
# MNIST
#################

BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

TRAIN_IMAGES_FILE="${BASE_URL}/train-images-idx3-ubyte.gz"
TRAIN_LABELS_FILE="${BASE_URL}/train-labels-idx1-ubyte.gz"
TEST_IMAGES_FILE="${BASE_URL}/t10k-images-idx3-ubyte.gz"
TEST_LABELS_FILE="${BASE_URL}/t10k-labels-idx1-ubyte.gz"

wget $TRAIN_IMAGES_FILE
wget $TRAIN_LABELS_FILE
wget $TEST_IMAGES_FILE
wget $TEST_LABELS_FILE

BASE_URL="https://storage.googleapis.com/learnjs-data/model-builder"

MNIST_IMAGES_SPRITE_PATH="${BASE_URL}/mnist_images.png"
MNIST_LABELS_PATH="${BASE_URL}/mnist_labels_uint8"

wget MNIST_IMAGES_SPRITE_PATH
wget MNIST_LABELS_PATH

# wget https://storage.googleapis.com/learnjs-data/speech-commands/speech-commands-data-v0.02-browser.tar.gz
# tar -czvf speech-commands-data-v0.02-browser.tar.gz
