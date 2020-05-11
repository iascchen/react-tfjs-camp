import * as tf from '@tensorflow/tfjs'
import { IMAGE_H, IMAGE_W } from './mnistConsts'

export const addCovDropoutLayers = (model: tf.Sequential): void => {
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_H, IMAGE_W, 1], filters: 32, kernelSize: 3, activation: 'relu'
    }))
    model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu' }))
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
    model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }))
    model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }))
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
    model.add(tf.layers.flatten())
    model.add(tf.layers.dropout({ rate: 0.25 }))
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }))
    model.add(tf.layers.dropout({ rate: 0.5 }))
}

export const addDenseLayers = (model: tf.Sequential): void => {
    model.add(tf.layers.flatten({ inputShape: [IMAGE_H, IMAGE_W, 1] }))
    model.add(tf.layers.dense({ units: 42, activation: 'relu' }))
}

export const addCovPoolingLayers = (model: tf.Sequential): void => {
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_H, IMAGE_W, 1], kernelSize: 3, filters: 16, activation: 'relu'
    }))
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
    model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }))
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }))
    model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }))
    model.add(tf.layers.flatten({}))
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }))
}
