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

import * as tf from '@tensorflow/tfjs-node'
import { MnistGzDataset } from './dataGz'
import { addSimpleConvLayers } from './model'

const EPOCHS = 10
const BATCH_SIZE = 128
const VALID_SPLIT = 0.15

const run = async (epochs: number, batchSize: number, modelSavePath: string) => {
    const model = tf.sequential()
    addSimpleConvLayers(model)
    const optimizer = 'rmsprop'
    model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] })
    model.summary()

    const mnistDataset = new MnistGzDataset()
    await mnistDataset.loadData()

    let tSet: tf.TensorContainerObject
    let vSet: tf.TensorContainerObject
    tSet = mnistDataset.getTrainData()
    vSet = mnistDataset.getTestData()

    console.log("before fit")
    await model.fit(tSet.xs as tf.Tensor, tSet.ys as tf.Tensor, {
        epochs: epochs,
        batchSize: batchSize,
        validationSplit: VALID_SPLIT
        // callbacks:{
        //     onBatchEnd: async (batch: number, logs: tf.Logs): Promise<void> => {
        //         console.log(batch, logs)
        //     }
        // }
    })

    const evalOutput = model.evaluate(vSet.xs as tf.Tensor, vSet.ys as tf.Tensor)

    console.log(
        '\nEvaluation result:\n' +
        `  Loss = ${evalOutput}; `)

    if (modelSavePath != null) {
        await model.save(`file://${modelSavePath}`); console.log(`Saved model to path: ${modelSavePath}`);
    }
}

run(EPOCHS, BATCH_SIZE, './mmm')
