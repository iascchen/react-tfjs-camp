import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as handpose from '@tensorflow-models/handpose'
import { WebcamIterator } from '@tensorflow/tfjs-data/dist/iterators/webcam_iterator'
import { Card, Col, Row } from 'antd'

import { logger, loggerError, STATUS } from '../../utils'
import { drawPath, drawPoint } from './pretrainedUtils'

const VIDEO_SHAPE = [640, 500] // [width, height]
const HAND_POSE_CONFIG = {
    // facingMode: 'user',
    resizeWidth: VIDEO_SHAPE[0],
    resizeHeight: VIDEO_SHAPE[1],
    centerCrop: false
}

interface IKeyIndices {
    [index: string]: number[]
}
const fingerLookupIndices: IKeyIndices = {
    thumb: [0, 1, 2, 3, 4],
    indexFinger: [0, 5, 6, 7, 8],
    middleFinger: [0, 9, 10, 11, 12],
    ringFinger: [0, 13, 14, 15, 16],
    pinky: [0, 17, 18, 19, 20]
} // for rendering each finger as a polyline

const drawKeypoints = (ctx: CanvasRenderingContext2D, keypoints: any[]): void => {
    const keypointsArray = keypoints

    for (let i = 0; i < keypointsArray.length; i++) {
        const y = keypointsArray[i][0]
        const x = keypointsArray[i][1]
        drawPoint(ctx, x - 2, y - 2, 3)
    }

    const fingers = Object.keys(fingerLookupIndices)
    for (let i = 0; i < fingers.length; i++) {
        const finger = fingers[i]
        const points = fingerLookupIndices[finger].map(idx => keypoints[idx])
        drawPath(ctx, points, false)
    }
}

const HandPose = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sModel, setModel] = useState<handpose.HandPose>()
    const [sPredictResult, setPredictResult] = useState<any[]>([])

    const [sCamera, setCamera] = useState<WebcamIterator>()
    const videoRef = useRef<HTMLVideoElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)

    /***********************
     * useEffect
     ***********************/
    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.WAITING)

        let _model: handpose.HandPose
        handpose.load().then(
            (model) => {
                _model = model
                setModel(_model)

                setStatus(STATUS.LOADED)
            },
            loggerError
        )

        return () => {
            logger('Model Dispose')
            // _model.dispose()
        }
    }, [])

    useEffect(() => {
        if (!videoRef.current) {
            return
        }
        logger('init video...')

        let _cam: WebcamIterator
        tf.data.webcam(videoRef.current, HAND_POSE_CONFIG).then(
            (cam) => {
                _cam = cam
                setCamera(cam)
            },
            loggerError
        )

        return () => {
            _cam?.stop()
        }
    }, [videoRef])

    useEffect(() => {
        if (!sModel || !sCamera) {
            return
        }
        logger('init VideoFrame ...')

        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        detectFromVideoFrame(sModel, sCamera).then()
    }, [sModel, sCamera])

    /***********************
     * Functions
     ***********************/

    const detectFromVideoFrame = async (model: handpose.HandPose, video: WebcamIterator): Promise<void> => {
        try {
            const stream = await video.capture()
            const predictions = await model.estimateHands(stream)
            if (predictions.length > 0) {
                showDetectionsText(predictions)
                showDetections(predictions)
            }
            stream.dispose()

            requestAnimationFrame(() => {
                // eslint-disable-next-line @typescript-eslint/no-floating-promises
                detectFromVideoFrame(model, video).then()
            })
        } catch (error) {
            logger("Couldn't start the webcam")
            console.error(error)
        }
    }

    const showDetections = (predictions: any[]): void => {
        if (!canvasRef.current || predictions.length <= 0) {
            return
        }

        const ctx = canvasRef.current.getContext('2d')
        if (!ctx) {
            return
        }
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
        const keypoints = predictions[0].landmarks
        drawKeypoints(ctx, keypoints)
    }

    const showDetectionsText = (predictions: any[]): void => {
        if (predictions.length > 0) {
            const keypoints = predictions[0].landmarks
            const data = keypoints.map((point: number[]) => {
                return point.map((v: number) => +v.toFixed(2))
            })
            setPredictResult(data)
        }
    }

    /***********************
     * Render
     ***********************/

    return (
        <>
            <h1>手势识别 Hand Pose</h1>
            <Row>
                <Col span={12} >
                    <div className='centerContainer'>
                        <video autoPlay muted playsinline width={VIDEO_SHAPE[0]} height={VIDEO_SHAPE[1]} ref={videoRef}
                            style={{ backgroundColor: 'lightgray' }}/>
                    </div>
                </Col>
                <Col span={12}>
                    <div className='centerContainer'>
                        <canvas width={VIDEO_SHAPE[0]} height={VIDEO_SHAPE[1]} ref={canvasRef}
                            style={{ backgroundColor: 'lightgray' }}/>
                    </div>
                </Col>
                <Col span={24}>
                    <Card title={'Key Points'} style={{ margin: '8px' }} size='small'>
                        <div>{JSON.stringify(sPredictResult)}</div>
                    </Card>
                </Col>
                <Col span={24}>
                    <div>Status : {sStatus}</div>
                    <div>TfBackend : {sTfBackend}</div>
                </Col>
            </Row>
        </>
    )
}

export default HandPose
