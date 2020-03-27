/**
 * @Author: iascchen@gmail.com
 * @Comments:
 * Adapted from some codes in Google tfjs-examples or tfjs-models.
 * Refactoring to typescript for RTL(React Tensorflow.js Lab)'s needs
 */

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

export const buildLinearRegressionModel = (inputShape: tf.Shape): tf.LayersModel => {
    const model = tf.sequential()
    model.add(tf.layers.flatten({ inputShape }))
    model.add(tf.layers.dense({ units: 1 }))
    return model
}

export const buildMLPModel = (inputShape: tf.Shape,
    options: {
        kernelRegularizer?: any
        dropoutRate?: number
    } = {}): tf.LayersModel => {
    const model = tf.sequential()

    model.add(tf.layers.flatten({ inputShape }))

    const { kernelRegularizer, dropoutRate } = options
    if (kernelRegularizer) {
        model.add(tf.layers.dense({ units: 32, kernelRegularizer, activation: 'relu' }))
    } else {
        model.add(tf.layers.dense({ units: 32, activation: 'relu' }))
    }

    if (dropoutRate && dropoutRate > 0) {
        model.add(tf.layers.dropout({ rate: dropoutRate }))
    }
    model.add(tf.layers.dense({ units: 1 }))
    return model
}

export const buildSimpleRNNModel = (inputShape: tf.Shape): tf.LayersModel => {
    const model = tf.sequential()
    const rnnUnits = 32
    model.add(tf.layers.simpleRNN({ units: rnnUnits, inputShape }))
    model.add(tf.layers.dense({ units: 1 }))
    return model
}

export const buildGRUModel = (inputShape: tf.Shape, dropout?: number, recurrentDropout?: number): tf.LayersModel => {
    // TODO(cais): Recurrent dropout is currently not fully working.
    //   Make it work and add a flag to train-rnn.js.
    const model = tf.sequential()
    const rnnUnits = 32
    model.add(tf.layers.gru({
        units: rnnUnits,
        inputShape,
        dropout: dropout ?? 0,
        recurrentDropout: recurrentDropout ?? 0
    }))
    model.add(tf.layers.dense({ units: 1 }))
    return model
}
