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
import * as model from './modelTextGen'

import { logger } from '../../utils'

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
        this.model = model.createModel(this.sampleLen_, this.charSetSize_, lstmLayerSizes)
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
        model.compileModel(this.model, learningRate)
    }

    /**
     * Train the LSTM model.
     *
     * @param {number} numEpochs Number of epochs to train the model for.
     * @param {number} examplesPerEpoch Number of epochs to use in each training
     *   epochs.
     * @param {number} batchSize Batch size to use during training.
     * @param {number} validationSplit Validation split to be used during the
     *   training epochs.
     */
    fitModel = async (numEpochs: number, examplesPerEpoch: number, batchSize: number, validationSplit: number,
        callbacks: any[]): Promise<void> => {
        if (!this.model || !this.textData_) {
            return
        }
        await model.fitModel(this.model, this.textData_, numEpochs, examplesPerEpoch, batchSize,
            validationSplit, callbacks)
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

        // onTextGenerationBegin()
        const callbacks = async (char: string): Promise<void> => {
            // ignore
        }
        return model.generateText(this.model, this.textData_, sentenceIndices, length, temperature, callbacks)
    }

    stopTrain = (stop: boolean): void => {
        if (!this.model) {
            return
        }
        this.model.stopTraining = stop
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
