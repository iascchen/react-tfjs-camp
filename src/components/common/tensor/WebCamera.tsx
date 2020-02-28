import React, { forwardRef, Ref, useEffect, useImperativeHandle, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { WebcamIterator } from '@tensorflow/tfjs-data/dist/iterators/webcam_iterator'
import { Button } from 'antd'

import { logger } from '../../../utils'
import { MOBILENET_IMAGE_SIZE } from '../../../constant'
import { ImagenetClasses } from '../../mobilenet/ImagenetClasses'
import TensorImageThumbWidget from './TensorImageThumbWidget'

const VIDEO_SHAPE = [480, 360] // [width, height]
const webcamConfig = {
    // facingMode: 'user',
    centerCrop: false,
    resizeWidth: MOBILENET_IMAGE_SIZE,
    resizeHeight: MOBILENET_IMAGE_SIZE
}

export interface IWebCameraHandler {
    capture: () => Promise<tf.Tensor3D | void>
}

interface IProps {
    model?: tf.LayersModel
    prediction?: tf.Tensor
    isPreview?: boolean

    onSubmit?: (tensor: tf.Tensor) => void
}

const WebCamera = (props: IProps, ref: Ref<IWebCameraHandler>): JSX.Element => {
    const [sLabel, setLabel] = useState<string>()
    const [sPreview, setPreview] = useState<tf.Tensor3D>()

    const [sCamera, setCamera] = useState<WebcamIterator>()
    const videoRef = useRef<HTMLVideoElement>(null)

    useImperativeHandle(ref, (): IWebCameraHandler => ({
        capture: capture
    }))

    useEffect(() => {
        if (!videoRef.current) {
            return
        }
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        tf.data.webcam(videoRef.current, webcamConfig).then(
            (cam) => setCamera(cam)
        )
    }, [videoRef])

    useEffect(() => {
        if (!props.prediction) {
            return
        }

        // Imagenet Classes
        const imagenetRet = props.prediction
        const labelIndex = imagenetRet.arraySync() as number
        logger('labelIndex', labelIndex)
        const label = ImagenetClasses[labelIndex]
        setLabel(`${labelIndex.toString()} : ${label}`)
    }, [props.prediction])

    const capture = async (): Promise<tf.Tensor3D | void> => {
        if (!sCamera) {
            return
        }
        const img = await sCamera.capture()
        const processedImg: tf.Tensor3D = tf.tidy(() => img.toFloat().div(255))
        img.dispose()

        props.isPreview && setPreview(processedImg)
        return processedImg
    }

    const handleCapture = async (): Promise<void> => {
        await capture()
    }

    const handleSubmit = async (): Promise<void> => {
        const imgTensor = await capture()
        if (imgTensor) {
            // setPreview(imgTensor)
            props.onSubmit && props.onSubmit(imgTensor)
        }
    }

    /***********************
     * Render
     ***********************/

    return (
        <>
            <div>
                <video autoPlay muted playsInline width={VIDEO_SHAPE[0]} height={VIDEO_SHAPE[1]} ref={videoRef}
                    style={{ backgroundColor: 'lightgray' }}/>
            </div>
            <div style={{ margin: 16 }}>
                <Button onClick={handleSubmit} type='primary' style={{ width: '30%', margin: '0 10%' }}>Predict</Button>
                {props.isPreview && (
                    <Button onClick={handleCapture} icon='camera'
                        style={{ width: '30%', margin: '0 10%' }}>Capture</Button>
                )}
            </div>
            {props.isPreview && (
                <>
                    <div>Captured Images</div>
                    <div>
                        {sPreview && <TensorImageThumbWidget width={VIDEO_SHAPE[0] / 2} height={VIDEO_SHAPE[1] / 2}
                            data={sPreview}/>}
                    </div>
                </>
            )}
            Prediction Result : {sLabel}
        </>
    )
}

export default forwardRef(WebCamera)
