import * as tf from '@tensorflow/tfjs'
import * as zlib from 'zlib'
import { UploadFile } from 'antd/es/upload/interface'

export const logger = console.log

export type IDataSet = tf.data.Dataset<tf.TensorContainer>
export type IArray = any[]

export enum STATUS {
    INIT = 'Init',
    LOADING = 'Loading',
    LOADED = 'Loaded',
    TRAINING = 'Training',
    TRAINED = 'Trained',
    PREDICTING = 'Predicting',
    PREDICTED = 'Predicted',
    STOPPED = 'Stopped',
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

export interface ILabeledImage {
    uid: string
    name: string
    tensor?: tf.Tensor3D | undefined
    img?: string | undefined // base64 of image
}

export interface ILabeledImageSet {
    label: string
    imageList?: ILabeledImage[]
}

export interface ILabeledImageFileJson {
    keys?: number[]
    labeledImageSetList: ILabeledImageSet[]
}

export interface IKnnPredictResult {
    label: string
    classIndex: number
    confidences: {
        [label: string]: number
    }
}

export interface ILayerSelectOption {
    name: string
    index: number
}

export interface ILabelMap {
    [index: number]: string
}

export const range = (from: number, to = 0): number[] => {
    // return [...Array(Math.abs(to - from)).keys()].map(v => v + from)
    return Array.from(tf.range(from, to).dataSync())
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
        logger('unzip...', url)
        return zlib.unzipSync(Buffer.from(buf))
    } else {
        return Buffer.from(buf)
    }
}

// const base64ToArray = (base64: string): [Uint8Array, string] => {
//     const arr = base64.split(',')
//     const mime = arr[0].match(/:(.*?);/)[1]
//     const bstr = atob(arr[1])
//     let n = bstr.length
//     const u8arr = new Uint8Array(n)
//
//     while (n--) {
//         u8arr[n] = bstr.charCodeAt(n)
//     }
//     return [u8arr, mime]
// }
//
// export const base64ToFile = (base64: string, filename: string): File => {
//     const [u8arr, mime] = base64ToArray(base64)
//     return new File([u8arr], filename, { type: mime })
// }
//
// export const base64ToBlob = (base64: string): Blob => {
//     const [u8arr, mime] = base64ToArray(base64)
//     return new Blob([u8arr], { type: mime })
// }

export const getUploadFileBase64 = async (blob: File | Blob | undefined): Promise<string> => {
    if (!blob) {
        throw (new Error('File Blob is undefined'))
    }

    return new Promise((resolve, reject) => {
        const reader = new FileReader()
        // logger('blob', JSON.stringify(blob))
        reader.onload = () => {
            const text = reader.result?.toString()
            // logger('getUploadFileBase64', text)
            resolve(text)
        }
        reader.onerror = error => reject(error)
        reader.readAsDataURL(blob)
    })
}

export const getUploadFileArray = async (blob: File | Blob | undefined): Promise<Buffer> => {
    if (!blob) {
        throw (new Error('File Blob is undefined'))
    }

    return new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => {
            const buffer = reader.result as ArrayBuffer
            // logger('getUploadFileArray', text)
            resolve(Buffer.from(buffer))
        }
        reader.onerror = error => reject(error)
        reader.readAsArrayBuffer(blob)
    })
}

export const getImageDataFromBase64 = async (imgBase64: string): Promise<ImageData> => {
    const img = new Image()
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    return new Promise((resolve) => {
        img.crossOrigin = ''
        img.onload = () => {
            img.width = img.naturalWidth
            img.height = img.naturalHeight

            ctx?.drawImage(img, 0, 0, img.width, img.height)
            const imageData = ctx?.getImageData(0, 0, canvas.width, canvas.height)
            // logger('imageData', imageData)
            resolve(imageData)
        }
        img.src = imgBase64
    })
}

export const checkUploadDone = (fileList: UploadFile[]): number => {
    let unload: number = fileList.length
    fileList.forEach(item => {
        // logger(item.status)
        if (item.status === 'done') {
            unload--
        }
    })
    logger('waiting checkUploadDone : ', fileList.length, unload)
    return unload
}

export const loggerError = (e: any): void => {
    logger(e)
    // eslint-disable-next-line @typescript-eslint/no-floating-promises
    // message.error(e)
}
