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
import {logger} from '../../utils'

const BASE_URL = '/preload/model'

const HOSTED_URLS = {
    model: `${BASE_URL}/sentiment_cnn_v1/model.json`,
    // 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
    metadata: `${BASE_URL}/sentiment_cnn_v1/metadata.json`
    // 'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
}

export class SentimentPredictor {
    urls = HOSTED_URLS
    model: tf.LayersModel | undefined

    indexFrom = 0
    maxLen = 0

    wordIndex: number[] = []
    vocabularySize = 0

    constructor () {
    }

    /**
     * Initializes the Sentiment demo.
     */
    init = async (): Promise<tf.LayersModel | void> => {
        // this.model = await loader.loadHostedPretrainedModel(urls.model)
        try {
            this.model = await tf.loadLayersModel(this.urls.model)
            await this.loadMetadata()

            return this.model
        } catch (err) {
            console.error(err)
            // ui.status('Loading pretrained model failed.');
        }
    }

    loadMetadata = async (): Promise<void> => {
        //     await loader.loadHostedMetadata(this.urls.metadata)

        try {
            const metadataJson = await fetch(this.urls.metadata)
            const sentimentMetadata = await metadataJson.json()
            // return metadata

            logger('sentimentMetadata.model_type', sentimentMetadata.model_type)

            // ui.showMetadata(sentimentMetadata)
            this.indexFrom = sentimentMetadata.index_from
            this.maxLen = sentimentMetadata.max_len
            console.log('indexFrom = ' + this.indexFrom)
            console.log('maxLen = ' + this.maxLen)

            this.wordIndex = sentimentMetadata.word_index
            this.vocabularySize = sentimentMetadata.vocabulary_size
            console.log('vocabularySize = ', this.vocabularySize)
        } catch (err) {
            console.error(err)
            // ui.status('Loading metadata failed.')
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
            let wordIndex = this.wordIndex[word] + this.indexFrom
            if (wordIndex > this.vocabularySize) {
                wordIndex = OOV_INDEX
            }
            return wordIndex
        })
        // Perform truncation and padding.
        const paddedSequence = padSequences([sequence], this.maxLen)
        const input = tf.tensor2d(paddedSequence, [1, this.maxLen])

        const beginMs = performance.now()
        const predictOut = this.model.predict(input) as tf.Tensor
        const score = predictOut.dataSync()[0]
        predictOut.dispose()
        const endMs = performance.now()

        return { score: score, elapsed: (endMs - beginMs) }
    }
}
