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
import { IMnistDataSet, IMAGE_H, IMAGE_W, IMAGE_SIZE, NUM_CLASSES } from './mnistConsts'

const NUM_DATASET_ELEMENTS = 65000

const NUM_TRAIN_ELEMENTS = 35000
const NUM_TEST_ELEMENTS = 7000

const BASE_URL = '/preload/data/mnist'
const MNIST_IMAGES_SPRITE_PATH = `${BASE_URL}/mnist_images.png`
const MNIST_LABELS_PATH = `${BASE_URL}/mnist_labels_uint8`

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */

export class MnistDatasetPng implements IMnistDataSet {
    datasetImages!: Float32Array
    trainImages!: Float32Array
    testImages!: Float32Array

    datasetLabels!: Uint8Array
    trainLabels!: Uint8Array
    testLabels!: Uint8Array

    trainIndices!: Uint32Array
    testIndices!: Uint32Array

    shuffledTrainIndex: number
    shuffledTestIndex: number

    constructor () {
        this.shuffledTrainIndex = 0
        this.shuffledTestIndex = 0
    }

    loadData = async (): Promise<void> => {
        // Make a request for the MNIST sprited image.
        const img = new Image()
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')
        const imgRequest = new Promise((resolve, reject) => {
            img.crossOrigin = ''
            img.onload = () => {
                img.width = img.naturalWidth
                img.height = img.naturalHeight

                const datasetBytesBuffer =
                    new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4)

                const chunkSize = 5000
                canvas.width = img.width
                canvas.height = chunkSize

                for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
                    const datasetBytesView = new Float32Array(
                        datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
                        IMAGE_SIZE * chunkSize)
                    ctx?.drawImage(
                        img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
                        chunkSize)

                    const imageData = ctx?.getImageData(0, 0, canvas.width, canvas.height)

                    const length = imageData?.data.length ?? 0
                    for (let j = 0; j < length / 4; j++) {
                        // All channels hold an equal value since the image is grayscale, so
                        // just read the red channel.
                        const v = imageData?.data[j * 4] ?? 0
                        datasetBytesView[j] = v / 255
                    }
                }
                this.datasetImages = new Float32Array(datasetBytesBuffer)

                resolve()
            }
            img.src = MNIST_IMAGES_SPRITE_PATH
        })

        const labelsRequest = fetch(MNIST_LABELS_PATH)
        const [imgResponse, labelsResponse] =
            await Promise.all([imgRequest, labelsRequest])

        this.datasetLabels = new Uint8Array(await (labelsResponse as Response).arrayBuffer())

        // Create shuffled indices into the train/test set for when we select a
        // random dataset element for training / validation.
        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS)
        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS)

        // Slice the the images and labels into train and test sets.
        this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS)
        this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS)
        this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS)
        this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS)
    }

    /**
     * Get all training data as a data tensor and a labels tensor.
     *
     * @returns
     *   xs: The data tensor, of shape `[numTrainExamples, 28, 28, 1]`.
     *   labels: The one-hot encoded labels tensor, of shape
     *     `[numTrainExamples, 10]`.
     */
    getTrainData = (numExamples?: number): tf.TensorContainerObject => {
        let xs = tf.tensor4d(
            this.trainImages,
            [this.trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1])
        let labels = tf.tensor2d(
            this.trainLabels, [this.trainLabels.length / NUM_CLASSES, NUM_CLASSES])

        if (numExamples != null) {
            xs = xs.slice([0, 0, 0, 0], [numExamples, IMAGE_H, IMAGE_W, 1])
            labels = labels.slice([0, 0], [numExamples, NUM_CLASSES])
        }
        return { xs, ys: labels }
    }

    /**
     * Get all test data as a data tensor a a labels tensor.
     *
     * @param {number} numExamples Optional number of examples to get. If not
     *     provided,
     *   all test examples will be returned.
     * @returns
     *   xs: The data tensor, of shape `[numTestExamples, 28, 28, 1]`.
     *   labels: The one-hot encoded labels tensor, of shape
     *     `[numTestExamples, 10]`.
     */
    getTestData = (numExamples?: number): tf.TensorContainerObject => {
        let xs = tf.tensor4d(this.testImages,
            [this.testImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1])
        let labels = tf.tensor2d(
            this.testLabels, [this.testLabels.length / NUM_CLASSES, NUM_CLASSES])

        if (numExamples != null) {
            xs = xs.slice([0, 0, 0, 0], [numExamples, IMAGE_H, IMAGE_W, 1])
            labels = labels.slice([0, 0], [numExamples, NUM_CLASSES])
        }
        return { xs, ys: labels }
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

        const xs = tf.tensor4d(batchImagesArray, [batchSize, IMAGE_H, IMAGE_W, 1])
        const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES])

        return { xs, ys: labels }
    }
}
