import * as tf from '@tensorflow/tfjs-node'
import * as zlib from 'zlib'
// import fetch from 'isomorphic-fetch'
import * as fs from 'fs'
import * as util from 'util'
export const logger = console.log

export type IDataSet = tf.data.Dataset<tf.TensorContainer>
export type IArray = any[]

export enum STATUS {
    INIT = 'Init',
    LOADING = 'Loading',
    LOADED = 'Loaded',
    TRAINING = 'Training',
    TRAINED = 'Trained',
}

export interface ITrainInfo {
    iteration?: number
    logs: tf.Logs
}

export interface ISampleInfo {
    data: number[]
    shape: number[]
    shapeStr: string
    shapeSize: number
    length: number
}

// export const range = (from: number, to = 0): number[] => {
//     return [...Array(Math.abs(to - from)).keys()].map(v => v + from)
// }

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

const readFile = util.promisify(fs.readFile);

// Downloads a test file only once and returns the buffer for the file.
export const fetchLocal = async (filename: string, isUnzip?: boolean): Promise<Buffer | undefined> => {
    const buf = await readFile(filename)
    // const buf = await response.arrayBuffer()
    // const buf = Buffer.from(response)
    if (isUnzip) {
        logger('unzip...', filename)
        return zlib.unzipSync(Buffer.from(buf))
    } else {
        return Buffer.from(buf)
    }
}

export const fetchResource = async (url: string, isUnzip?: boolean): Promise<Buffer | undefined> => {
    try{
        const response = await fetch(url)
        // const buf = await response.arrayBuffer()
        const buf = await Buffer.from(response.text())
        if (isUnzip) {
            logger('unzip...', url)
            return zlib.unzipSync(Buffer.from(buf))
        } else {
            return Buffer.from(buf)
        }
    } catch(e){
        logger(e)
    }
}

export const getUploadFileBase64 = async (blob: File | Blob | undefined): Promise<string> => {
    if (!blob) {
        throw (new Error('File Blob is undefined'))
    }

    return new Promise((resolve, reject) => {
        const reader = new FileReader()
        console.log('blob', JSON.stringify(blob))
        reader.readAsDataURL(blob)
        reader.onload = () => {
            const text = reader.result?.toString()
            // logger('getUploadFileBase64', text)
            resolve(text)
        }
        reader.onerror = error => reject(error)
    })
}

export const getUploadFileArray = async (blob: File | Blob | undefined): Promise<Buffer> => {
    if (!blob) {
        throw (new Error('File Blob is undefined'))
    }

    return new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.readAsArrayBuffer(blob)
        reader.onload = () => {
            const buffer = reader.result as ArrayBuffer
            // logger('getUploadFileBase64', text)
            resolve(Buffer.from(buffer))
        }
        reader.onerror = error => reject(error)
    })
}
