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

/**
 * TensorFlow.js Example: LSTM Text Generation.
 *
 * Inspiration comes from:
 *
 * -
 * https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
 * - Andrej Karpathy. "The Unreasonable Effectiveness of Recurrent Neural
 * Networks" http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 */

import * as tf from '@tensorflow/tfjs'

import { TextData } from './dataTextGen'

import { logger } from '../utils'

const sample = (probs: tf.Tensor, temperature: number): number => {
    return tf.tidy(() => {
        const logits = tf.div(tf.log(probs), Math.max(temperature, 1e-6))
        const isNormalized = false
        // `logits` is for a multinomial distribution, scaled by the temperature.
        // We randomly draw a sample from the distribution.
        return tf.multinomial(logits as tf.Tensor1D, 1, undefined, isNormalized).dataSync()[0]
    })
}

/**
 * Class that manages LSTM-based text generation.
 *
 * This class manages the following:
 *
 * - Creating and training a LSTM model, written with the tf.layers API, to
 *   predict the next character given a sequence of input characters.
 * - Generating random text using the LSTM model.
 */
export class LSTMTextGenerator {
    textData_: TextData
    charSetSize_: number
    sampleLen_: number
    textLen_: number

    model: tf.LayersModel | undefined
    stopFlag = false

    /**
     * Constructor of NeuralNetworkTextGenerator.
     *
     * @param {TextData} textData An instance of `TextData`.
     */
    constructor (textData: TextData) {
        this.textData_ = textData
        this.charSetSize_ = textData.charSetSize()
        this.sampleLen_ = textData.sampleLen()
        this.textLen_ = textData.textLen()
    }

    /**
     * Create LSTM model from scratch.
     *
     * @param {number | number[]} lstmLayerSizes Sizes of the LSTM layers, as a
     *   number or an non-empty array of numbers.
     */

    createModel = (lstmLayerSizes: number | number[]): void => {
        if (!Array.isArray(lstmLayerSizes)) {
            lstmLayerSizes = [lstmLayerSizes]
        }

        const model = tf.sequential()
        for (let i = 0; i < lstmLayerSizes.length; ++i) {
            const lstmLayerSize = lstmLayerSizes[i]
            model.add(tf.layers.lstm({
                units: lstmLayerSize,
                returnSequences: i < lstmLayerSizes.length - 1,
                inputShape: i === 0 ? [this.sampleLen_, this.charSetSize_] : undefined
            }))
        }
        model.add(tf.layers.dense({ units: this.charSetSize_, activation: 'softmax' }))

        this.model = model
    }

    /**
     * Compile model for training.
     *
     * @param {number} learningRate The learning rate to use during training.
     */
    compileModel = (learningRate: number): void => {
        if (!this.model) {
            return
        }
        // logger(`Compiled model with learning rate ${learningRate}`)
        const optimizer = tf.train.rmsprop(learningRate)
        this.model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' })
        // model.summary()
    }

    myCallback = {
        onBatchBegin: async (batch: number) => {
            if (!this.model) {
                return
            }

            if (this.stopFlag) {
                logger('Checked stop', this.stopFlag)
                this.model.stopTraining = this.stopFlag
            }
            await tf.nextFrame()
        }
    }

    fitModel = async (numEpochs: number, examplesPerEpoch: number, batchSize: number, validationSplit: number,
        callbacks: any[]): Promise<void> => {
        if (!this.model || !this.textData_) {
            return
        }

        const _callbacks = [this.myCallback, ...callbacks]

        this.stopFlag = false
        for (let i = 0; i < numEpochs; ++i) {
            if (this.stopFlag) {
                return
            }

            const [xs, ys] = tf.tidy(() => {
                return this.textData_.nextDataEpoch(examplesPerEpoch)
            })
            await this.model.fit(xs, ys, {
                epochs: 1,
                batchSize: batchSize,
                validationSplit,
                callbacks: _callbacks
            })
            xs.dispose()
            ys.dispose()
        }
    }

    /**
     * Generate text using the LSTM model.
     *
     * @param {number[]} sentenceIndices Seed sentence, represented as the
     *   indices of the constituent characters.
     * @param {number} length Length of the text to generate, in number of
     *   characters.
     * @param {number} temperature Temperature parameter. Must be a number > 0.
     * @returns {string} The generated text.
     */
    generateText = async (sentenceIndices: number[], length: number, temperature: number): Promise<string | void> => {
        if (!this.model) {
            return
        }

        const callbacks = (char: string): void => {
            // ignore
            logger('genCallback', char)
        }

        const sampleLen = this.model.inputs[0].shape[1] as number
        const charSetSize = this.model.inputs[0].shape[2] as number

        // Avoid overwriting the original input.
        sentenceIndices = sentenceIndices.slice()

        let generated = ''
        while (generated.length < length) {
        // Encode the current input sequence as a one-hot Tensor.
            const inputBuffer = new tf.TensorBuffer([1, sampleLen, charSetSize], 'float32')

            // Make the one-hot encoding of the seeding sentence.
            for (let i = 0; i < sampleLen; ++i) {
                inputBuffer.set(1, 0, i, sentenceIndices[i])
            }
            const input = inputBuffer.toTensor()

            // Call model.predict() to get the probability values of the next
            // character.
            const output = this.model.predict(input) as tf.Tensor

            // Sample randomly based on the probability values.
            const winnerIndex = sample(tf.squeeze(output), temperature)
            const winnerChar = this.textData_.getFromCharSet(winnerIndex)
            if (callbacks != null) {
                await callbacks(winnerChar)
            }

            generated += winnerChar
            sentenceIndices = sentenceIndices.slice(1)
            sentenceIndices.push(winnerIndex)

            // Memory cleanups.
            input.dispose()
            output.dispose()
        }
        return generated
    }

    stopTrain = (): void => {
        this.stopFlag = true
    }

    loadModelFromFile = async (url: string): Promise<tf.LayersModel> => {
        this.model = await tf.loadLayersModel(url)
        return this.model
    }
}

/**
 * A subclass of LSTMTextGenerator that supports model saving and loading.
 *
 * The model is saved to and loaded from browser's IndexedDB.
 */
export class SavableLSTMTextGenerator extends LSTMTextGenerator {
    modelIdentifier_: string
    MODEL_SAVE_PATH_PREFIX_: string
    modelSavePath_: string

    /**
     * Constructor of NeuralNetworkTextGenerator.
     *
     * @param {TextData} textData An instance of `TextData`.
     */
    constructor (textData: TextData) {
        super(textData)
        this.modelIdentifier_ = textData.dataIdentifier()
        this.MODEL_SAVE_PATH_PREFIX_ = 'indexeddb://lstm-text-generation'
        this.modelSavePath_ = `${this.MODEL_SAVE_PATH_PREFIX_}/${this.modelIdentifier_}`
    }

    /**
     * Get model identifier.
     *
     * @returns {string} The model identifier.
     */
    modelIdentifier = (): string => {
        return this.modelIdentifier_
    }

    /**
     * Create LSTM model if it is not saved locally; load it if it is.
     *
     * @param {number | number[]} lstmLayerSizes Sizes of the LSTM layers, as a
     *   number or an non-empty array of numbers.
     */
    loadModel = async (lstmLayerSizes: number): Promise<void> => {
        const modelsInfo = await tf.io.listModels()
        if (this.modelSavePath_ in modelsInfo) {
            logger('Loading existing model...')
            this.model = await tf.loadLayersModel(this.modelSavePath_)
            logger(`Loaded model from ${this.modelSavePath_}`)
        } else {
            throw new Error(
                `Cannot find model at ${this.modelSavePath_}. ` +
                'Creating model from scratch.')
        }
    }

    /**
     * Save the model in IndexedDB.
     *
     * @returns ModelInfo from the saving, if the saving succeeds.
     */
    saveModel = async (): Promise<tf.io.SaveResult> => {
        if (this.model == null) {
            throw new Error('Cannot save model before creating model.')
        } else {
            return this.model.save(this.modelSavePath_)
        }
    }

    /**
     * Remove the locally saved model from IndexedDB.
     */
    removeModel = async (): Promise<any> => {
        if (await this.checkStoredModelStatus() == null) {
            throw new Error(
                'Cannot remove locally saved model because it does not exist.')
        }
        return tf.io.removeModel(this.modelSavePath_)
    }

    /**
     * Check the status of locally saved model.
     *
     * @returns If the locally saved model exists, the model info as a JSON
     *   object. Else, `undefined`.
     */
    checkStoredModelStatus = async (): Promise<any> => {
        const modelsInfo = await tf.io.listModels()
        return modelsInfo[this.modelSavePath_]
    }

    /**
     * Get a representation of the sizes of the LSTM layers in the model.
     *
     * @returns {number | number[]} The sizes (i.e., number of units) of the
     *   LSTM layers that the model contains. If there is only one LSTM layer, a
     *   single number is returned; else, an Array of numbers is returned.
     */
    lstmLayerSizes = (): void => {
        if (this.model == null) {
            throw new Error('Create model first.')
        }
        const numLSTMLayers = this.model.layers.length - 1
        const layerSizes = []
        for (let i = 0; i < numLSTMLayers; ++i) {
            const layer = this.model.layers[i] as any
            layerSizes.push(layer.units)
        }
        return layerSizes.length === 1 ? layerSizes[0] : layerSizes
    }
}
