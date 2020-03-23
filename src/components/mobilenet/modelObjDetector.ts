import * as tf from '@tensorflow/tfjs'
import { MOBILENET_IMAGE_SIZE, MOBILENET_MODEL_PATH } from './mobilenetUtils'

// const LABEL_MULTIPLIER = [MOBILENET_IMAGE_SIZE, 1, 1, 1, 1]

// Name prefixes of layers that will be unfrozen during fine-tuning.
const topLayerGroupNames = ['conv_pw_9', 'conv_pw_10', 'conv_pw_11']

// Name of the layer that will become the top layer of the truncated base.
const topLayerName = `${topLayerGroupNames[topLayerGroupNames.length - 1]}_relu`

interface IModelWithFineTuning {
    model: tf.LayersModel
    fineTuningLayers: tf.layers.Layer[]
}

const loadTruncatedBase = async (): Promise<IModelWithFineTuning> => {
    const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH)

    // Return a model that outputs an internal activation.
    const fineTuningLayers: tf.layers.Layer[] = []
    const layer = mobilenet.getLayer(topLayerName)
    const truncatedBase = tf.model({ inputs: mobilenet.inputs, outputs: layer.output })
    // Freeze the model's layers.
    for (const layer of truncatedBase.layers) {
        layer.trainable = false
        for (const groupName of topLayerGroupNames) {
            if (layer.name.indexOf(groupName) === 0) {
                fineTuningLayers.push(layer)
                break
            }
        }
    }

    return { model: truncatedBase, fineTuningLayers }
}

const buildNewHead = (inputShape: tf.Shape): tf.LayersModel => {
    const newHead = tf.sequential()
    newHead.add(tf.layers.flatten({ inputShape }))
    newHead.add(tf.layers.dense({ units: 200, activation: 'relu' }))
    // Five output units:
    //   - The first is a shape indictor: predicts whether the target
    //     shape is a triangle or a rectangle.
    //   - The remaining four units are for bounding-box prediction:
    //     [left, right, top, bottom] in the unit of pixels.
    newHead.add(tf.layers.dense({ units: 5 }))
    return newHead
}

export const buildObjectDetectionModel = async (): Promise<IModelWithFineTuning> => {
    const { model: truncatedBase, fineTuningLayers } = await loadTruncatedBase()

    // Build the new head model.
    const newHead = buildNewHead(truncatedBase.outputs[0].shape.slice(1))
    const newOutput = newHead.apply(truncatedBase.outputs[0])
    const model = tf.model({ inputs: truncatedBase.inputs, outputs: newOutput as tf.SymbolicTensor })

    return { model, fineTuningLayers }
}

export const customLossFunction = (yTrue: tf.Tensor, yPred: tf.Tensor): tf.Tensor => {
    return tf.tidy(() => {
        // Scale the the first column (0-1 shape indicator) of `yTrue` in order
        // to ensure balanced contributions to the final loss value
        // from shape and bounding-box predictions.
        // return tf.metrics.meanSquaredError(yTrue.mul(LABEL_MULTIPLIER), yPred)
        return tf.metrics.meanSquaredError(yTrue, yPred)
    })
}
