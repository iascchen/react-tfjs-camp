/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import * as path from 'path'

import { OOV_INDEX, padSequences } from './sequenceUtils'
import { fetchResource } from '../../utils'

const BASE_URL = '/preload/data/imdb'

// const DATA_ZIP_URL = `${BASE_URL}/imdb_tfjs_data.zip`
// 'https://storage.googleapis.com/learnjs-data/imdb/imdb_tfjs_data.zip'
const METADATA_TEMPLATE_URL = `${BASE_URL}/metadata.json`
// 'https://storage.googleapis.com/learnjs-data/imdb/metadata.json.zip'

/**
 * Load IMDB data features from a local file.
 *
 * @param {string} filePath Data file on local filesystem.
 * @param {string} numWords Number of words in the vocabulary. Word indices
 *   that exceed this limit will be marked as `OOV_INDEX`.
 * @param {string} maxLen Length of each sequence. Longer sequences will be
 *   pre-truncated; shorter ones will be pre-padded.
 * @param {string} multihot Whether to use multi-hot encoding of the words.
 *   Default: `false`.
 * @return {tf.Tensor} If `multihot` is `false` (default), the dataset
 *   represented as a 2D `tf.Tensor` of shape `[numExamples, maxLen]` and
 *   dtype `int32`. Else, the dataset represented as a 2D `tf.Tensor` of
 *   shape `[numExamples, numWords]` and dtype `float32`.
 */
const loadFeatures = async (filePath: string, numWords: number, maxLen: number,
    multihot = false): Promise<tf.Tensor> => {
    const buffer = await fetchResource(filePath, false)
    const numBytes = buffer.byteLength

    const sequences = []
    let seq = []
    let index = 0

    while (index < numBytes) {
        const value = buffer.readInt32LE(index)
        if (value === 1) {
            // A new sequence has started.
            if (index > 0) {
                sequences.push(seq)
            }
            seq = []
        } else {
            // Sequence continues.
            seq.push(value >= numWords ? OOV_INDEX : value)
        }
        index += 4
    }
    if (seq.length > 0) {
        sequences.push(seq)
    }

    // Get some sequence length stats.
    let minLength = Infinity
    let maxLength = -Infinity
    sequences.forEach(seq => {
        const length = seq.length
        if (length < minLength) {
            minLength = length
        }
        if (length > maxLength) {
            maxLength = length
        }
    })
    console.log(`Sequence length: min = ${minLength}; max = ${maxLength}`)

    if (multihot) {
    // If requested by the arg, encode the sequences as multi-hot
    // vectors.
        const buffer = tf.buffer([sequences.length, numWords])
        sequences.forEach((seq, i) => {
            seq.forEach(wordIndex => {
                if (wordIndex !== OOV_INDEX) {
                    buffer.set(1, i, wordIndex)
                }
            })
        })
        return buffer.toTensor()
    } else {
        const paddedSequences =
        padSequences(sequences, maxLen, 'pre', 'pre')
        return tf.tensor2d(
            paddedSequences, [paddedSequences.length, maxLen], 'int32')
    }
}

/**
 * Load IMDB targets from a file.
 *
 * @param {string} filePath Path to the binary targets file.
 * @return {tf.Tensor} The targets as `tf.Tensor` of shape `[numExamples, 1]`
 *   and dtype `float32`. It has 0 or 1 values.
 */
const loadTargets = async (filePath: string): Promise<tf.Tensor2D> => {
    const buffer = await fetchResource(filePath, false)
    const numBytes = buffer.byteLength

    let numPositive = 0
    let numNegative = 0

    const ys = []
    for (let i = 0; i < numBytes; ++i) {
        const y = buffer.readUInt8(i)
        if (y === 1) {
            numPositive++
        } else {
            numNegative++
        }
        ys.push(y)
    }

    console.log(`Loaded ${numPositive} positive examples and ${numNegative} negative examples.`)
    return tf.tensor2d(ys, [ys.length, 1], 'float32')
}

/**
 * Load data by downloading and extracting files if necessary.
 *
 * @param {number} numWords Number of words to in the vocabulary.
 * @param {number} len Length of each sequence. Longer sequences will
 *   be pre-truncated and shorter ones will be pre-padded.
 * @return
 *   xTrain: Training data as a `tf.Tensor` of shape
 *     `[numExamples, len]` and `int32` dtype.
 *   yTrain: Targets for the training data, as a `tf.Tensor` of
 *     `[numExamples, 1]` and `float32` dtype. The values are 0 or 1.
 *   xTest: The same as `xTrain`, but for the test dataset.
 *   yTest: The same as `yTrain`, but for the test dataset.
 */
export const loadData = async (numWords: number, len: number, multihot = false): Promise<tf.TensorContainerObject> => {
    // const dataDir = await maybeDownloadAndExtract()

    const dataDir = `${BASE_URL}/`
    const trainFeaturePath = path.join(dataDir, 'imdb_train_data.bin')
    const xTrain = await loadFeatures(trainFeaturePath, numWords, len, multihot)
    const testFeaturePath = path.join(dataDir, 'imdb_test_data.bin')
    const xTest = await loadFeatures(testFeaturePath, numWords, len, multihot)
    const trainTargetsPath = path.join(dataDir, 'imdb_train_targets.bin')
    const yTrain = await loadTargets(trainTargetsPath)
    const testTargetsPath = path.join(dataDir, 'imdb_test_targets.bin')
    const yTest = await loadTargets(testTargetsPath)

    tf.util.assert(
        xTrain.shape[0] === yTrain.shape[0],
        () => 'Mismatch in number of examples between xTrain and yTrain')
    tf.util.assert(
        xTest.shape[0] === yTest.shape[0],
        () => 'Mismatch in number of examples between xTest and yTest')
    return { xTrain, yTrain, xTest, yTest }
}
