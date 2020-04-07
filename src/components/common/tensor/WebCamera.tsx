import React, { forwardRef, Ref, useEffect, useImperativeHandle, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { WebcamIterator } from '@tensorflow/tfjs-data/dist/iterators/webcam_iterator'
import { Button, Row, Col } from 'antd'
import { CameraOutlined } from '@ant-design/icons'

import { IKnnPredictResult, ILabelMap, logger, loggerError } from '../../../utils'
import { ImagenetClasses } from '../../mobilenet/ImagenetClasses'

import TensorImageThumbWidget from './TensorImageThumbWidget'

const VIDEO_SHAPE = [480, 360] // [width, height]
const IMAGE_HEIGHT = 86

const DEFAULT_CONFIG = {
    // facingMode: 'user',
    // resizeWidth: VIDEO_SHAPE[0],
    // resizeHeight: VIDEO_SHAPE[1],
    centerCrop: false
}

export interface IWebCameraHandler {
    capture: () => Promise<tf.Tensor3D | void>
}

interface IProps {
    prediction?: tf.Tensor | IKnnPredictResult
    isPreview?: boolean
    labelsMap?: ILabelMap
    config?: tf.data.WebcamConfig

    onSubmit?: (tensor: tf.Tensor) => void
}

const WebCamera = (props: IProps, ref: Ref<IWebCameraHandler>): JSX.Element => {
    const [sLabel, setLabel] = useState<string>()
    const [sPreview, setPreview] = useState<tf.Tensor3D>()

    const [sCamera, setCamera] = useState<WebcamIterator>()
    const videoRef = useRef<HTMLVideoElement>(null)

    useImperativeHandle(ref, (): IWebCameraHandler => ({
        capture
    }))

    useEffect(() => {
        if (!videoRef.current) {
            return
        }

        let _cam: WebcamIterator

        const config = props.config ? props.config : DEFAULT_CONFIG
        tf.data.webcam(videoRef.current, config).then(
            (cam) => {
                _cam = cam
                setCamera(cam)
            },
            loggerError
        )

        return () => {
            _cam?.stop()
        }
    }, [videoRef, props.config])

    useEffect(() => {
        if (!props.prediction) {
            return
        }

        const knnRet = props.prediction as IKnnPredictResult
        if (knnRet.label) {
            const knnRet = props.prediction as IKnnPredictResult
            setLabel(`${knnRet.label} : ${JSON.stringify(knnRet.confidences)}`)
        } else {
            // Imagenet Classes
            const imagenetRet = props.prediction as tf.Tensor
            const labelIndex = imagenetRet.arraySync() as number
            logger('labelIndex', labelIndex)
            const _label = props.labelsMap ? props.labelsMap[labelIndex] : ImagenetClasses[labelIndex]
            setLabel(`${labelIndex.toString()} : ${_label}`)
        }
    }, [props.prediction])

    const capture = async (): Promise<tf.Tensor3D | void> => {
        if (!sCamera) {
            return
        }
        return sCamera.capture()
    }

    const handleCapture = async (): Promise<void> => {
        const imgTensor = await capture()
        props.isPreview && setPreview(imgTensor as tf.Tensor3D)
    }

    const handleSubmit = async (): Promise<void> => {
        const imgTensor = await capture()
        if (imgTensor) {
            props.isPreview && setPreview(imgTensor)
            props.onSubmit && props.onSubmit(imgTensor)
        }
    }

    /***********************
     * Render
     ***********************/

    return (
        <>
            <Row className='centerContainer'>
                <video autoPlay muted playsInline width={VIDEO_SHAPE[0]} height={VIDEO_SHAPE[1]} ref={videoRef}
                    style={{ backgroundColor: 'lightgray' }}/>
            </Row>
            <Row className='centerContainer'>
                <div style={{ width: 500, padding: '8px' }}>
                    {props.isPreview && (
                        <Button style={{ width: '30%', margin: '0 10%' }} icon={<CameraOutlined />}
                            onClick={handleCapture} >Capture</Button>
                    )}
                    {props.onSubmit && (
                        <Button onClick={handleSubmit} type='primary' style={{ width: '30%', margin: '0 10%' }}>Predict</Button>
                    )}
                </div>
            </Row>
            <Row >
                {props.isPreview && (
                    <Col span={12}>
                        <div className='centerContainer'>Captured Images</div>
                        <div className='centerContainer'>
                            {sPreview && <TensorImageThumbWidget height={IMAGE_HEIGHT} data={sPreview}/>}
                        </div>
                    </Col>
                )}
                {sLabel && (
                    <Col span={12}>
                        <div className='centerContainer'> Prediction Result </div>
                        <div className='centerContainer' style={{ margin: '8px' }}>{sLabel}</div>
                    </Col>
                )}
            </Row>
        </>
    )
}

export default forwardRef(WebCamera)
