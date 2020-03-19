import * as tf from '@tensorflow/tfjs'

// export const MOBILENET_MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
export const MOBILENET_MODEL_PATH = '/preload/model/mobilenet_v1_0.25_224/model.json'

export const MOBILENET_IMAGE_SIZE = 224

export const formatImageForMobilenet = (imgTensor: tf.Tensor): tf.Tensor => {
    const sample = tf.image.resizeBilinear(imgTensor as tf.Tensor3D, [MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE])
    // logger(JSON.stringify(sample))

    const offset = tf.scalar(127.5)
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = sample.sub(offset).div(offset)
    // Reshape to a single-element batch so we can pass it to predict.
    return normalized.reshape([1, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE, 3])
}
