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
import { ILabeledImageSet, logger } from '../../utils'
import { formatImageForMobilenet } from './mobilenetUtils'

/**
 * A dataset for webcam controls which allows the user to add example Tensors
 * for particular labels. This object will concat them into two large xs and ys.
 */
export class TransferDataset {
    numClasses: number
    xs?: tf.Tensor
    ys?: tf.Tensor
    labels: string[]

    constructor (numClasses: number) {
        this.numClasses = numClasses
        this.labels = []
    }

    /**
     * Adds an example to the controller dataset.
     * @param {Tensor} example A tensor representing the example. It can be an image,
     *     an activation, or any other type of Tensor.
     * @param {number} label The label of the example. Should be a number.
     */
    addExample (example: tf.Tensor, label: number): void {
        // One-hot encode the label.
        const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses))

        if (this.xs == null || this.ys == null) {
            // For the first example that gets added, keep example and y so that the
            // ControllerDataset owns the memory of the inputs. This makes sure that
            // if addExample() is called in a tf.tidy(), these Tensors will not get
            // disposed.
            this.xs = tf.keep(example)
            this.ys = tf.keep(y)
        } else {
            const oldX = this.xs
            this.xs = tf.keep(oldX.concat(example, 0))

            const oldY = this.ys
            this.ys = tf.keep(oldY.concat(y, 0))

            oldX?.dispose()
            oldY?.dispose()
            y.dispose()
        }
    }

    addExamples (truncatedMobileNet: tf.LayersModel, labeledImages: ILabeledImageSet[]): void {
        if (!labeledImages) {
            return
        }

        this.labels = labeledImages.map((labeled) => labeled.label)
        labeledImages.forEach((labeled) => {
            const imgs = labeled.imageList
            const label = labeled.label
            imgs?.forEach(item => {
                if (item.tensor) {
                    const batched = formatImageForMobilenet(item.tensor)
                    const predicted = truncatedMobileNet.predict(batched)
                    this.addExample(predicted as tf.Tensor, this.labels.indexOf(label))
                }
            })
        })
    }

    getData (): tf.TensorContainerObject {
        logger(this.xs?.shape)
        return { xs: this.xs, ys: this.ys }
    }

    dispose (): void {
        this.xs?.dispose()
        this.ys?.dispose()
    }
}
