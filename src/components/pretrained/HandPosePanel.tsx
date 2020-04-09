import React, { useCallback, useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { HandPose, load as handPoseLoad } from '@tensorflow-models/handpose'
import { Button, Col, Row } from 'antd'

import { logger, loggerError, STATUS } from '../../utils'
import WebVideo, { IWebVideoHandler } from '../common/tensor/WebVideo'
import { downloadJson, drawPath, drawPoint } from './pretrainedUtils'

const JSON_NAME = 'hand-pose.json'

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

const HandPosePanel = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sModel, setModel] = useState<HandPose>()

    // const switchRef = useRef<boolean>(false)
    const webVideoRef = useRef<IWebVideoHandler>(null)

    const [sJson, setJson] = useState<any>()
    const downloadRef = useRef<HTMLAnchorElement>(null)

    /***********************
     * useEffect
     ***********************/
    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.WAITING)

        let _model: HandPose
        handPoseLoad().then(
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

    const handlePredict = useCallback(async (data: tf.Tensor3D): Promise<any[]> => {
        if (!sModel) {
            return []
        }
        return sModel.estimateHands(data)
    }, [sModel])

    /***********************
     * Functions
     ***********************/

    const showDetections = (predictions: any[]): void => {
        if (!webVideoRef.current || predictions?.length <= 0) {
            return
        }

        const ctx = webVideoRef.current.getContext()
        if (!ctx) {
            return
        }
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
        predictions.forEach((hand: any) => {
            const result = hand.landmarks
            drawKeypoints(ctx, result)
        })
    }

    const showDetectionsText = (predictions: any[]): void => {
        if (!webVideoRef.current || predictions?.length <= 0) {
            return
        }

        logger('predictions', predictions.length)
        // If you want to show more data, please revise it
        const data = predictions[0]
        webVideoRef.current.setPred(data)
        setJson(data)
    }

    const showPrediction = (predictions: any[]): void => {
        showDetectionsText(predictions)
        showDetections(predictions)
    }

    const handleJsonSave = (): void => {
        if (!sJson || !downloadRef.current) {
            return
        }
        downloadJson(sJson, JSON_NAME, downloadRef.current)
    }

    /***********************
     * Render
     ***********************/

    return (
        <>
            <h1>手势识别 Hand Pose</h1>
            <Row>
                <Col span={12}>
                </Col>
                <Col span={12}>
                    <div className='centerContainer'>
                        <Button style={{ width: '30%', margin: '0 10%' }} onClick={handleJsonSave}
                            disabled={sStatus === STATUS.WAITING} >Save Json</Button>
                    </div>
                </Col>
                <Col span={24}>
                    <div className='centerContainer'>
                        <WebVideo ref={webVideoRef} show={showPrediction} predict={handlePredict}/>
                    </div>
                </Col>
                <Col span={24}>
                    <div>Status : {sStatus}</div>
                    <div>TfBackend : {sTfBackend}</div>
                </Col>
            </Row>
            <a ref={downloadRef}/>
        </>
    )
}

export default HandPosePanel
