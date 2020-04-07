import React, { forwardRef, Ref, useEffect, useImperativeHandle, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { WebcamIterator } from '@tensorflow/tfjs-data/dist/iterators/webcam_iterator'
import { Row, Col, Card } from 'antd'

import { logger, loggerError } from '../../../utils'

import { IWebCameraHandler } from './WebCamera'

const VIDEO_SHAPE = [480, 360] // [width, height]

const DEFAULT_CONFIG = {
    // facingMode: 'user',
    resizeWidth: VIDEO_SHAPE[0],
    resizeHeight: VIDEO_SHAPE[1],
    centerCrop: false
}

export interface IWebVideoHandler extends IWebCameraHandler {
    getContext: () => CanvasRenderingContext2D | void
    setPred: (value: any[]) => void
}

interface IProps {
    predict: (value: tf.Tensor3D) => Promise<any>
    prediction?: any[]
    config?: tf.data.WebcamConfig

    show: (value: any) => void
}

const WebVideo = (props: IProps, ref: Ref<IWebVideoHandler>): JSX.Element => {
    const [sPrediction, setPrediction] = useState<any>()

    const [sCamera, setCamera] = useState<WebcamIterator>()
    const videoRef = useRef<HTMLVideoElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)

    useImperativeHandle(ref, (): IWebVideoHandler => ({
        capture, getContext, setPred
    }))

    useEffect(() => {
        if (!videoRef.current) {
            return
        }
        logger('init video...')

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
        if (!sCamera) {
            return
        }
        logger('init VideoFrame ...')

        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        detectFromVideoFrame(sCamera).then()
    }, [sCamera, props.predict])

    useEffect(() => {
        if (!props.prediction) {
            return
        }
        setPrediction(props.prediction)
    }, [props.prediction])

    const detectFromVideoFrame = async (video: WebcamIterator): Promise<void> => {
        try {
            const stream = await video.capture()
            const predictions = await props.predict(stream)
            if (props.show && predictions.length > 0) {
                props.show(predictions)
            }
            stream.dispose()

            requestAnimationFrame(() => {
                // eslint-disable-next-line @typescript-eslint/no-floating-promises
                detectFromVideoFrame(video).then()
            })
        } catch (error) {
            logger("Couldn't start the webcam")
            console.error(error)
        }
    }

    const capture = async (): Promise<tf.Tensor3D | void> => {
        if (!sCamera) {
            return
        }
        return sCamera.capture()
    }

    const getContext = (): CanvasRenderingContext2D | void => {
        if (!canvasRef.current) {
            return
        }
        const ctx = canvasRef.current.getContext('2d')
        if (ctx) {
            return ctx
        }
    }

    const setPred = (value: any): void => {
        setPrediction(value)
    }

    /***********************
     * Render
     ***********************/

    return (
        <div>
            <Row className='centerContainer'>
                <Col span={12} >
                    <div className='centerContainer'>
                        <video autoPlay muted playsInline width={VIDEO_SHAPE[0]} height={VIDEO_SHAPE[1]} ref={videoRef}
                            style={{ backgroundColor: 'lightgray' }}/>
                    </div>
                </Col>
                <Col span={12}>
                    <div className='centerContainer'>
                        <canvas width={VIDEO_SHAPE[0]} height={VIDEO_SHAPE[1]} ref={canvasRef}
                            style={{ backgroundColor: 'lightgray' }}/>
                    </div>
                </Col>
            </Row>
            <Row >
                <Col span={24} >
                    <Card title={'Key Points'} style={{ margin: '8px' }} size='small'>
                        <div>{JSON.stringify(sPrediction)}</div>
                    </Card>
                </Col>
            </Row>
        </div>
    )
}

export default forwardRef(WebVideo)
