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
import { fetchResource, logger } from '../../utils'
import { IMnistDataSet, IMAGE_H, IMAGE_W, IMAGE_SIZE, NUM_CLASSES } from './dataCore'

const NUM_TRAIN_ELEMENTS = 35000
const NUM_TEST_ELEMENTS = 7000

const IMAGE_HEADER_BYTES = 16
const LABEL_HEADER_BYTES = 8
const LABEL_RECORD_BYTE = 1

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */

const loadHeaderValues = (buffer: Buffer, headerLength: number): number[] => {
    const headerValues = []
    for (let i = 0; i < headerLength / 4; i++) {
        // Header data is stored in-order (aka big-endian)
        headerValues[i] = buffer.readUInt32BE(i * 4)
    }
    return headerValues
}

const loadImages = async (url: string): Promise<Float32Array[]> => {
    const buffer = await fetchResource(url, true)

    const headerBytes = IMAGE_HEADER_BYTES
    const recordBytes = IMAGE_SIZE

    // skip header
    const headerValues = loadHeaderValues(buffer, headerBytes)
    logger('image header', headerValues)

    const images = []
    let index = headerBytes
    while (index < buffer.byteLength) {
        const array = new Float32Array(recordBytes)
        for (let i = 0; i < recordBytes; i++) {
            // Normalize the pixel values into the 0-1 interval, from
            // the original 0-255 interval.
            array[i] = buffer.readUInt8(index++) / 255.0
        }
        images.push(array)
    }
    logger('Load images :', `${images.length.toString()} / ${headerValues[1].toString()}`)
    return images
}

const loadLabels = async (url: string): Promise<Uint8Array[]> => {
    const buffer = await fetchResource(url, true)

    const headerBytes = LABEL_HEADER_BYTES
    const recordBytes = LABEL_RECORD_BYTE

    // skip header
    const headerValues = loadHeaderValues(buffer, headerBytes)
    logger('label header', headerValues)

    const labels = []
    let index = headerBytes
    while (index < buffer.byteLength) {
        const array = new Uint8Array(recordBytes)
        for (let i = 0; i < recordBytes; i++) {
            array[i] = buffer.readUInt8(index++)
        }
        labels.push(array)
    }
    logger('Load labels :', `${labels.length.toString()} / ${headerValues[1].toString()}`)
    return labels
}

export class MnistGzDataset implements IMnistDataSet {
    source: string
    baseUrl: string
    trainImagesFileUrl: string
    trainLabelsFileUrl: string
    testImagesFileUrl: string
    testLabelsFileUrl: string

    trainImages!: Float32Array[]
    testImages!: Float32Array[]
    trainLabels!: Uint8Array[]
    testLabels!: Uint8Array[]

    trainIndices!: Uint32Array
    testIndices!: Uint32Array

    shuffledTrainIndex = 0
    shuffledTestIndex = 0

    constructor (source: string) {
        this.source = source

        this.baseUrl = `/preload/data/${source}`
        this.trainImagesFileUrl = `${this.baseUrl}/train-images-idx3-ubyte.gz`
        this.trainLabelsFileUrl = `${this.baseUrl}/train-labels-idx1-ubyte.gz`
        this.testImagesFileUrl = `${this.baseUrl}/t10k-images-idx3-ubyte.gz`
        this.testLabelsFileUrl = `${this.baseUrl}/t10k-labels-idx1-ubyte.gz`
    }

    /** Loads training and test data. */
    loadData = async (): Promise<void> => {
        // Create shuffled indices into the train/test set for when we select a
        // random dataset element for training / validation.
        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS)
        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS)

        // Slice the the images and labels into train and test sets.
        this.trainImages = await loadImages(this.trainImagesFileUrl)
        this.trainImages = this.trainImages.slice(0, NUM_TRAIN_ELEMENTS)
        this.trainLabels = await loadLabels(this.trainLabelsFileUrl)
        this.trainLabels = this.trainLabels.slice(0, NUM_TRAIN_ELEMENTS)

        this.testImages = await loadImages(this.testImagesFileUrl)
        this.testImages = this.testImages.slice(0, NUM_TEST_ELEMENTS)
        this.testLabels = await loadLabels(this.testLabelsFileUrl)
        this.testLabels = this.testLabels.slice(0, NUM_TEST_ELEMENTS)
    }

    getTrainData = (numExamples = NUM_TRAIN_ELEMENTS): tf.TensorContainerObject => {
        return this.getData_(this.trainImages, this.trainLabels, numExamples)
    }

    getTestData = (numExamples = NUM_TEST_ELEMENTS): tf.TensorContainerObject => {
        return this.getData_(this.testImages, this.testLabels, numExamples)
    }

    getData_ = (imageSet: Float32Array[], labelSet: Uint8Array[], numExamples?: number): tf.TensorContainerObject => {
        const size = imageSet.length

        // Only create one big array to hold batch of images.
        const imagesShape: [number, number, number, number] = [size, IMAGE_H, IMAGE_W, 1]
        const images = new Float32Array(tf.util.sizeFromShape(imagesShape))
        const labels = new Int32Array(tf.util.sizeFromShape([size, 1]))

        let imageOffset = 0
        let labelOffset = 0
        for (let i = 0; i < size; ++i) {
            images.set(imageSet[i], imageOffset)
            labels.set(labelSet[i], labelOffset)
            imageOffset += IMAGE_SIZE
            labelOffset += 1
        }

        let xs = tf.tensor4d(images, imagesShape)
        let ys = tf.oneHot(tf.tensor1d(labels, 'int32'), NUM_CLASSES)

        if (numExamples != null) {
            xs = xs.slice([0, 0, 0, 0], [numExamples, IMAGE_H, IMAGE_W, 1])
            ys = ys.slice([0, 0], [numExamples, NUM_CLASSES])
        }

        return { xs, ys }
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

    nextBatch = (batchSize: number, data: [Float32Array[], Uint8Array[]], index: Function): tf.TensorContainerObject => {
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE)
        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES)

        for (let i = 0; i < batchSize; i++) {
            const idx = index() as number

            const image = data[0].slice(idx, idx + 1)[0]
            batchImagesArray.set(image, i * IMAGE_SIZE)

            const label = data[1].slice(idx, idx + 1)[0]
            const ys = Array.from(tf.oneHot([label], NUM_CLASSES).dataSync())
            batchLabelsArray.set(ys, i * NUM_CLASSES)
        }

        const xs = tf.tensor4d(batchImagesArray, [batchSize, IMAGE_H, IMAGE_W, 1])
        const ys = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES])

        return { xs, ys }
    }
}
