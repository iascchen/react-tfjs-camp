import * as tf from '@tensorflow/tfjs'
import { MOBILENET_MODEL_PATH } from '../../constant'

export const createModel = async (outputClasses: number, learningRate = 0.15, denseUnits = 10): Promise<tf.LayersModel> => {
    const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH)

    // Return a model that outputs an internal activation.
    const layer = mobilenet.getLayer('conv_pw_13_relu')
    const truncatedMobileNet = tf.model({ inputs: mobilenet.inputs, outputs: layer.output })
    truncatedMobileNet.trainable = false

    const modelWillTrained = tf.sequential({
        layers: [
            // Flattens the input to a vector so we can use it in a dense layer. While
            // technically a layer, this only performs a reshape (and has no training
            // parameters).
            tf.layers.flatten(
                { inputShape: truncatedMobileNet.outputs[0].shape.slice(1) }),
            // Layer 1.
            tf.layers.dense({
                units: denseUnits,
                activation: 'relu',
                kernelInitializer: 'varianceScaling',
                useBias: true
            }),
            // Layer 2. The number of units of the last layer should correspond
            // to the number of classes we want to predict.
            tf.layers.dense({
                units: outputClasses,
                kernelInitializer: 'varianceScaling',
                useBias: false,
                activation: 'softmax'
            })
        ]
    })

    const output = modelWillTrained.apply(truncatedMobileNet.apply(truncatedMobileNet.inputs)) as tf.SymbolicTensor
    const finalModel = tf.model({ inputs: truncatedMobileNet.inputs, outputs: output })

    // const finalModel = tf.sequential()
    // finalModel.add(truncatedMobileNet)
    // finalModel.add(modelWillTrained)

    // Creates the optimizers which drives training of the model.
    const optimizer = tf.train.adam(learningRate)
    // We use categoricalCrossentropy which is the loss function we use for
    // categorical classification which measures the error between our predicted
    // probability distribution over classes (probability that an input is of each
    // class), versus the label (100% probability in the true class)>
    finalModel.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' })

    return finalModel
}
