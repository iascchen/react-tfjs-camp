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
import { MnistWebDataset } from './data'

export const IMAGE_H = 28
export const IMAGE_W = 28
const IMAGE_SIZE = IMAGE_H * IMAGE_W
const NUM_CLASSES = 10

// const MNIST_IMAGES_SPRITE_PATH =
//     'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
// const MNIST_LABELS_PATH =
//     'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

// const MNIST_IMAGES_SPRITE_PATH = '/data/mnist_images.png'
// const MNIST_LABELS_PATH = '/data/mnist_labels_uint8'

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export class MnistCoreDataset extends MnistWebDataset {
    shuffledTrainIndex: number
    shuffledTestIndex: number

    constructor () {
        super()
        this.shuffledTrainIndex = 0
        this.shuffledTestIndex = 0
    }

    nextTrainBatch = (batchSize: number): tf.TensorContainerObject => {
        return this.nextBatch(batchSize, [this.trainImages, this.trainLabels],
            () => {
                this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length
                return this.trainIndices[this.shuffledTrainIndex]
            })
    }

    nextTestBatch = (batchSize: number): tf.TensorContainerObject => {
        return this.nextBatch(batchSize, [this.testImages, this.testLabels],
            () => {
                this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length
                return this.testIndices[this.shuffledTestIndex]
            })
    }

    nextBatch = (batchSize: number, data: [Float32Array, Uint8Array], index: Function): tf.TensorContainerObject => {
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE)
        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES)

        for (let i = 0; i < batchSize; i++) {
            const idx = index() as number

            const image = data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE)
            batchImagesArray.set(image, i * IMAGE_SIZE)

            const label = data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES)
            batchLabelsArray.set(label, i * NUM_CLASSES)
        }

        const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE])
        const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES])

        return { xs, labels }
    }
}
