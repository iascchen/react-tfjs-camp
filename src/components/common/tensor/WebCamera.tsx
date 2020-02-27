import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as tfd from '@tensorflow/tfjs-data'
import { Button } from 'antd'

import { logger } from '../../../utils'
import { MOBILENET_IMAGE_SIZE } from '../../../constant'
import { ImagenetClasses } from '../../mobilenet/ImagenetClasses'

interface IProps {
    model?: tf.LayersModel
    prediction?: tf.Tensor

    onSubmit?: (tensor: tf.Tensor) => void
}

const VIDEO_SHAPE = [480, 360]
const webcamConfig = {
    // facingMode: 'user',
    centerCrop: false,
    resizeWidth: MOBILENET_IMAGE_SIZE,
    resizeHeight: MOBILENET_IMAGE_SIZE
}

const webcamCapture = async (ref: HTMLVideoElement): Promise<tf.Tensor> => {
    const cam = await tfd.webcam(ref, webcamConfig)
    const img = await cam.capture()
    const processedImg = tf.tidy(() => img.toFloat().div(255))
    img.dispose()
    return processedImg
}

const WebCamera = (props: IProps): JSX.Element => {
    const [label, setLabel] = useState()

    const videoRef = useRef<HTMLVideoElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        if (!videoRef.current) {
            return
        }

        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        tf.data.webcam(videoRef.current, webcamConfig).then()
    }, [videoRef])

    useEffect(() => {
        if (!props.prediction) {
            return
        }

        // Imagenet Classes
        const imagenetRet = props.prediction
        const labelIndex = imagenetRet.arraySync() as number
        logger('labelIndex', labelIndex)
        const _label = ImagenetClasses[labelIndex]
        setLabel(`${labelIndex.toString()} : ${_label}`)
    }, [props.prediction])

    const handleSubmit = async (): Promise<void> => {
        if (!videoRef.current) {
            return
        }
        const imgTensor = await webcamCapture(videoRef.current)
        props.onSubmit && props.onSubmit(imgTensor)
        canvasRef?.current && tf.browser.toPixels(imgTensor as tf.Tensor3D, canvasRef?.current)
    }

    /***********************
     * Render
     ***********************/

    return (
        <>
            <div>
                <video autoPlay muted playsInline height={VIDEO_SHAPE[1]} width={VIDEO_SHAPE[0]} ref={videoRef}
                    style={{ backgroundColor: 'lightgray' }}/>
            </div>
            <div>
                <Button onClick={handleSubmit} type='primary' style={{ width: '30%', margin: '0 10%' }}>Capture &
                    Predict</Button>
            </div>
            <div>Captured Images</div>
            <div>
                <canvas ref={canvasRef} height={MOBILENET_IMAGE_SIZE} width={MOBILENET_IMAGE_SIZE}/>
            </div>
            Prediction Result : {label}
        </>
    )
}

export default WebCamera
