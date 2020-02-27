import * as tf from '@tensorflow/tfjs'
import { MOBILENET_MODEL_PATH } from '../../constant'

export const loadTruncatedMobileNet = async (): Promise<tf.LayersModel> => {
    const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH)

    // Return a model that outputs an internal activation.
    const layer = mobilenet.getLayer('conv_pw_13_relu')
    return tf.model({ inputs: mobilenet.inputs, outputs: layer.output })
}

