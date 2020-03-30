import * as tf from '@tensorflow/tfjs'
import {ILabeledImage, ILabeledImageSet} from '../../utils'

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

export const encodeImageTensor = (labeledImgs: ILabeledImageSet[]): any[] => {
    if (!labeledImgs) {
        return []
    }

    labeledImgs.forEach((labeled, index) => {
        labeled.imageList?.forEach((imgItem: ILabeledImage) => {
            if (imgItem.tensor && !imgItem.img) {
                const f32Buf = new Float32Array(imgItem.tensor.dataSync())
                // logger(f32Buf.length)
                const ui8Buf = new Uint8Array(f32Buf.buffer)
                // logger(ui8Buf.length)
                imgItem.img = Buffer.from(ui8Buf).toString('base64')
            }
        })
    })
    return labeledImgs
}

export const decodeImageTensor = (labeledImgs: ILabeledImageSet[]): any[] => {
    // logger('decodeImageTensor', labeledImgs)
    if (!labeledImgs) {
        return []
    }

    labeledImgs.forEach((labeled, index) => {
        labeled.imageList?.forEach((imgItem: ILabeledImage) => {
            if (imgItem.tensor && imgItem.img) {
                const buf = Buffer.from(imgItem.img, 'base64')
                const ui8Buf = new Uint8Array(buf)
                // logger(ui8Buf.length)
                const f32Buf = new Float32Array(ui8Buf.buffer)
                // logger(f32Buf.length)
                imgItem.tensor = tf.tensor3d(f32Buf, imgItem.tensor.shape, imgItem.tensor.dtype)
                delete imgItem.img
            }
            // logger(imgItem)
        })
    })
    return labeledImgs
}
