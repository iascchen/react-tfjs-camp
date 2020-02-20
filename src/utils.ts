import * as tf from '@tensorflow/tfjs'
import * as zlib from 'zlib'

export const logger = console.log

export type ITensor = tf.Tensor | undefined
export type IModel = tf.Sequential | tf.LayersModel | undefined
export type IDataSet = tf.data.Dataset<tf.TensorContainer>
export type IArray = any[]

export enum STATUS {
    INIT = 'Init',
    LOADING = 'Loading',
    LOADED = 'Loaded',
    TRAINING = 'Training',
    TRAINED = 'Trained',
}

export interface IValidInfo {
    xs: ITensor
    ys: ITensor
    preds?: ITensor
}

export interface ITrainDataSet {
    xs: tf.Tensor
    ys: tf.Tensor
}

export interface ITrainInfo {
    step?: number
    logs: tf.Logs
}

export interface ISampleInfo {
    data: number[]
    shape: number[]
    shapeStr: string
    shapeSize: number
    length: number
}

export const range = (from: number, to = 0): number[] => {
    return [...Array(Math.abs(to - from)).keys()].map(v => v + from)
}

export const splitDataSet = (shuffled: IArray, testSplit: number, shuffle = false): IArray[] => {
    if (shuffle) {
        tf.util.shuffle(shuffled)
    }

    const totalRecord = shuffled.length
    // Split the data into training and testing portions.
    const numTestExamples = Math.round(totalRecord * testSplit)
    const numTrainExamples = totalRecord - numTestExamples

    const train = shuffled.slice(0, numTrainExamples)
    const test = shuffled.slice(numTrainExamples)

    return [train, test]
}

export const arrayDispose = (_array: any[]): void => {
    _array?.splice(0, _array.length)
}

export const fetchResource = async (url: string, isUnzip?: boolean): Promise<Buffer> => {
    const response = await fetch(url)
    const buf = await response.arrayBuffer()
    if (isUnzip) {
        logger('unzip...')
        return zlib.unzipSync(Buffer.from(buf))
    } else {
        return Buffer.from(buf)
    }
}
