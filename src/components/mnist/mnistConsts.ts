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

export const IMAGE_H = 28
export const IMAGE_W = 28
export const IMAGE_SIZE = IMAGE_H * IMAGE_W
export const NUM_CLASSES = 10

export interface IMnistDataSet {
    loadData: () => Promise<void>
    getTrainData: (numExamples?: number) => tf.TensorContainerObject
    getTestData: (numExamples?: number) => tf.TensorContainerObject

    nextTrainBatch: (batchSize: number) => tf.TensorContainerObject
    nextTestBatch: (batchSize: number) => tf.TensorContainerObject
}
