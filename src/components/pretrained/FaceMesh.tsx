import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as facemesh from '@tensorflow-models/facemesh'
import { WebcamIterator } from '@tensorflow/tfjs-data/dist/iterators/webcam_iterator'
import { Card, Col, Row, Switch } from 'antd'

import { logger, loggerError, STATUS } from '../../utils'
import { TRIANGULATION } from './triangulation'
import { drawPath, drawPoint } from './pretrainedUtils'

const VIDEO_SHAPE = [640, 500] // [width, height]
const VIDEO_CONFIG = {
    // facingMode: 'user',
    resizeWidth: VIDEO_SHAPE[0],
    resizeHeight: VIDEO_SHAPE[1],
    centerCrop: false
}

const FaceMesh = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sModel, setModel] = useState<facemesh.FaceMesh>()

    const [sPredictResult, setPredictResult] = useState<any[]>([])
    const triangulateMesh = useRef<boolean>(false)

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

        let _model: facemesh.FaceMesh
        facemesh.load().then(
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
        tf.data.webcam(videoRef.current, VIDEO_CONFIG).then(
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

    const detectFromVideoFrame = async (model: facemesh.FaceMesh, video: WebcamIterator): Promise<void> => {
        try {
            const stream = await video.capture()
            const predictions = await model.estimateFaces(stream)
            if (predictions.length > 0) {
                showDetectionsText(predictions)
                showDetections(predictions)
            }
            stream.dispose()
            // await tf.nextFrame()

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
        const keypoints = predictions[0].scaledMesh

        if (triangulateMesh.current) {
            for (let i = 0; i < TRIANGULATION.length / 3; i++) {
                const points = [
                    TRIANGULATION[i * 3], TRIANGULATION[i * 3 + 1],
                    TRIANGULATION[i * 3 + 2]
                ].map(index => keypoints[index])

                drawPath(ctx, points, true)
            }
        } else {
            for (let i = 0; i < keypoints.length; i++) {
                const x = keypoints[i][0]
                const y = keypoints[i][1]

                drawPoint(ctx, y, x, 1)
            }
        }
    }

    const showDetectionsText = (predictions: any[]): void => {
        if (predictions.length > 0) {
            const data = predictions[0].scaledMesh.map((point: number[]) => {
                return point.map((v: number) => +v.toFixed(2))
            })
            setPredictResult(data)
        }
    }

    const handleSwitch = (value: boolean): void => {
        // logger('handleSwitch', value)
        triangulateMesh.current = value
    }

    /***********************
     * Render
     ***********************/

    return (
        <>
            <h1>面部特征 Face Mesh</h1>
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
                    <div className='centerContainer'>
                        <Switch onChange={handleSwitch} />
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

export default FaceMesh
