/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs'

import { MnistDatasetCore } from './dataCore'

// Hyperparameters.
const LEARNING_RATE = 0.1
const BATCH_SIZE = 64
const TRAIN_STEPS = 100

// Data constants.
const IMAGE_SIZE = 28
const LABELS_SIZE = 10
const optimizer = tf.train.sgd(LEARNING_RATE)

// Variables that we want to optimize
const conv1OutputDepth = 8
const conv1Weights = tf.variable(tf.randomNormal([5, 5, 1, conv1OutputDepth], 0, 0.1))

const conv2InputDepth = conv1OutputDepth
const conv2OutputDepth = 16
const conv2Weights = tf.variable(
    tf.randomNormal([5, 5, conv2InputDepth, conv2OutputDepth], 0, 0.1))

const fullyConnectedWeights = tf.variable(
    tf.randomNormal([7 * 7 * conv2OutputDepth, LABELS_SIZE], 0,
        1 / Math.sqrt(7 * 7 * conv2OutputDepth)))
const fullyConnectedBias = tf.variable(tf.zeros([LABELS_SIZE]))

// Loss function
const loss = (labels: tf.Tensor, ys: tf.Tensor): tf.Scalar => {
    return tf.losses.softmaxCrossEntropy(labels, ys).mean()
}

// Our actual model
export const model = (inputXs: tf.Tensor): tf.Tensor => {
    const xs = inputXs.as4D(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    const strides = 2
    const pad = 0

    // Conv 1
    const layer1 = tf.tidy(() => {
        return xs.conv2d(conv1Weights as tf.Tensor4D, 1, 'same')
            .relu()
            .maxPool([2, 2], strides, pad)
    })

    // Conv 2
    const layer2 = tf.tidy(() => {
        return layer1.conv2d(conv2Weights as tf.Tensor4D, 1, 'same')
            .relu()
            .maxPool([2, 2], strides, pad)
    })

    // Final layer
    return layer2.as2D(-1, fullyConnectedWeights.shape[0])
        .matMul(fullyConnectedWeights as tf.Tensor)
        .add(fullyConnectedBias)
}

// Train the model.
export const train = async (data: MnistDatasetCore, log: Function): Promise<void> => {
    const returnCost = true

    for (let i = 0; i < TRAIN_STEPS; i++) {
        const cost = optimizer.minimize(() => {
            const batch = data.nextTrainBatch(BATCH_SIZE)
            const _labels = batch.labels as tf.Tensor
            const _xs = batch.xs as tf.Tensor
            return loss(_labels, model(_xs))
        }, returnCost)

        log(i, cost?.dataSync())
        await tf.nextFrame()
    }
}

// Predict the digit number from a batch of input images.
export const predict = (x: tf.Tensor): tf.Tensor => {
    const pred = tf.tidy(() => {
        const axis = 1
        return model(x).argMax(axis)
    })
    return pred
}

// Given a logits or label vector, return the class indices.
export const classesFromLabel = (y: tf.Tensor): number[] => {
    const axis = 1
    const pred = y.argMax(axis)

    return Array.from(pred.dataSync())
}
