import * as tf from '@tensorflow/tfjs'
import { IMAGE_HEIGHT, IMAGE_WIDTH } from './dataGz'

/**
 * Creates a convolutional neural network (Convnet) for the MNIST data.
 *
 * @returns {tf.Model} An instance of tf.Model.
 */
export const addCnnLayers = (_model: tf.Sequential): void => {
    // Create a sequential neural network model. tf.sequential provides an API
    // for creating "stacked" models where the output from one layer is used as
    // the input to the next layer.
    // const _model = tf.sequential()

    _model.add(tf.layers.conv2d({
        inputShape: [IMAGE_HEIGHT, IMAGE_WIDTH, 1], kernelSize: 3, filters: 32, activation: 'relu'
    }))
    _model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }))
    _model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }))
    _model.add(tf.layers.conv2d({ kernelSize: 3, filters: 64, activation: 'relu' }))
    _model.add(tf.layers.conv2d({ kernelSize: 3, filters: 64, activation: 'relu' }))
    _model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }))

    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    _model.add(tf.layers.flatten({}))
    _model.add(tf.layers.dropout({ rate: 0.25 }))
    _model.add(tf.layers.dense({ units: 512, activation: 'relu' }))
    _model.add(tf.layers.dropout({ rate: 0.5 }))
    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
    // represent numbers, but it's the same idea if you had classes that
    // represented other entities like dogs and cats (two output classes: 0, 1).
    // We use the softmax function as the activation for the output layer as it
    // creates a probability distribution over our 10 classes so their output
    // values sum to 1.
    _model.add(tf.layers.dense({ units: 10, activation: 'softmax' }))

    // _model.summary()
}

export const addDenseLayers = (_model: tf.Sequential): void => {
    // const _model = tf.sequential()
    _model.add(tf.layers.flatten({ inputShape: [IMAGE_HEIGHT, IMAGE_WIDTH, 1] }))
    _model.add(tf.layers.dense({ units: 42, activation: 'relu' }))
    // _model.add(tf.layers.dropout({ rate: 0.2 }))
    _model.add(tf.layers.dense({ units: 10, activation: 'softmax' }))

    // _model.summary()
}

export const addSimpleConvLayers = (_model: tf.Sequential): void => {
    // // const _model = tf.sequential()
    _model.add(tf.layers.conv2d({ inputShape: [IMAGE_HEIGHT, IMAGE_WIDTH, 1], kernelSize: 3, filters: 8, activation: 'relu' }))
    _model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }))
    _model.add(tf.layers.conv2d({ kernelSize: 3, filters: 16, activation: 'relu' }))
    _model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }))
    _model.add(tf.layers.flatten({}))
    _model.add(tf.layers.dense({ units: 10, activation: 'softmax' }))
}
