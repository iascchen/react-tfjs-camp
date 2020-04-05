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
import { OOV_INDEX, padSequences } from './sequenceUtils'
import { logger } from '../../utils'
import { DATA_BASE_URL } from './dataSentiment'

const BASE_URL = '/preload/model'
export const PRETRAINED_HOSTED_URLS = {
    model: `${BASE_URL}/sentiment_cnn_v1/model.json`,
    // 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
    metadata: `${BASE_URL}/sentiment_cnn_v1/metadata.json`
    // 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
}

const DEFAULT_METADATA_URLS = {
    model: '',
    metadata: `${DATA_BASE_URL}/metadata.json`
    // 'https://storage.googleapis.com/learnjs-data/imdb/metadata.json.zip'
}

export class SentimentPredictor {
    urls = DEFAULT_METADATA_URLS
    model: tf.LayersModel | undefined
    metadata: any = {}

    constructor (urls = DEFAULT_METADATA_URLS) {
        this.urls = urls
    }

    /**
     * Initializes the Sentiment demo.
     */
    init = async (): Promise<tf.LayersModel | void> => {
        try {
            if (this.urls.model || this.urls.model.length > 0) {
                this.model = await tf.loadLayersModel(this.urls.model)
            }
            if (this.urls.metadata) {
                await this.loadMetadata(this.urls.metadata)
            }
            return this.model
        } catch (err) {
            console.error(err)
            // ui.status('Loading pretrained model failed.');
        }
    }

    loadMetadata = async (metadataUrl: string): Promise<void> => {
        try {
            const metadataJson = await fetch(metadataUrl)
            const sentimentMetadata = await metadataJson.json()

            logger('sentimentMetadata.model_type', sentimentMetadata.model_type)
            this.metadata = { ...sentimentMetadata }
        } catch (err) {
            console.error(err)
            // ui.status('Loading metadata failed.')
        }
    }

    setModel = (model: tf.LayersModel): void => {
        this.model = model
    }

    updateMetadata = (options: any, force = false): void => {
        if (!this.metadata) {
            return
        }

        if (force) {
            this.metadata = { ...this.metadata, ...options }
        } else {
            const keys = Object.keys(options)
            keys.forEach(key => {
                if (this.metadata[key] == null) {
                    this.metadata[key] = options[key]
                }
            })
        }
    }

    predict = (text: string): any => {
        if (!this.model) {
            return
        }

        // Convert to lower case and remove all punctuations.
        const inputText =
            text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ')
        // Convert the words to a sequence of word indices.
        const sequence = inputText.map((word: any) => {
            let wordIndex = this.metadata.word_index[word] + this.metadata.index_from
            if (wordIndex > this.metadata.vocabulary_size) {
                wordIndex = OOV_INDEX
            }
            return wordIndex
        })
        // Perform truncation and padding.
        const paddedSequence = padSequences([sequence], this.metadata.max_len)
        const input = tf.tensor2d(paddedSequence, [1, this.metadata.max_len])

        const beginMs = performance.now()
        const predictOut = this.model.predict(input) as tf.Tensor
        const score = predictOut.dataSync()[0]
        predictOut.dispose()
        const endMs = performance.now()

        return { score: score, elapsed: (endMs - beginMs) }
    }
}
