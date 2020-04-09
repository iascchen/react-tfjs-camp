import React, { useCallback, useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import {
    getAdjacentKeyPoints,
    Keypoint,
    load as poseNetLoad,
    MobileNetMultiplier,
    Pose,
    PoseNet,
    PoseNetOutputStride
} from '@tensorflow-models/posenet'
import { PoseNetArchitecture, PoseNetDecodingMethod, PoseNetQuantBytes } from '@tensorflow-models/posenet/dist/types'
import { Button, Col, Row } from 'antd'

import { logger, loggerError, STATUS } from '../../utils'
import WebVideo, { IWebVideoHandler } from '../common/tensor/WebVideo'
import { downloadJson, drawPoint, drawSegment } from './pretrainedUtils'

const JSON_NAME = 'pose-net.json'

const POSENET_CONFIG = {
    architecture: 'MobileNetV1' as PoseNetArchitecture, // ['MobileNetV1', 'ResNet50']
    outputStride: 16 as PoseNetOutputStride,
    multiplier: 0.75 as MobileNetMultiplier,
    quantBytes: 2 as PoseNetQuantBytes,
    inputResolution: 513
}

const PREDICT_CONFIG = {
    decodingMethod: 'single-person' as PoseNetDecodingMethod,
    flipHorizontal: false,
    maxDetections: 1,
    scoreThreshold: 0.9,
    nmsRadius: 2
}

const toTuple = ({ y, x }: any): number[] => {
    return [y, x]
}

const drawSkeleton = (keypoints: Keypoint[], minConfidence: number, ctx: CanvasRenderingContext2D, scale = 1): void => {
    const adjacentKeyPoints = getAdjacentKeyPoints(keypoints, minConfidence)

    adjacentKeyPoints.forEach((keypoints) => {
        drawSegment(ctx, toTuple(keypoints[0].position), toTuple(keypoints[1].position), scale)
    })
}

const drawKeypoints = (keypoints: Keypoint[], minConfidence: number, ctx: CanvasRenderingContext2D, scale = 1): void => {
    for (let i = 0; i < keypoints.length; i++) {
        const keypoint = keypoints[i]
        // logger(keypoint.score)
        if (keypoint.score < minConfidence) {
            continue
        }
        const { y, x } = keypoint.position
        drawPoint(ctx, y * scale, x * scale, 3)
    }
}

const PoseNetPanel = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sModel, setModel] = useState<PoseNet>()
    const [sMinConfidence] = useState<number>(0.1)

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

        let _model: PoseNet
        poseNetLoad(POSENET_CONFIG).then(
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

    const handlePredict = useCallback(async (data: tf.Tensor3D): Promise<Pose[]> => {
        if (!sModel) {
            return []
        }
        return sModel.estimatePoses(data, PREDICT_CONFIG)
    }, [sModel])

    /***********************
     * Functions
     ***********************/

    const showDetections = (predictions: Pose[]): void => {
        if (!webVideoRef.current || predictions?.length <= 0) {
            return
        }

        const ctx = webVideoRef.current.getContext()
        if (!ctx) {
            return
        }
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

        predictions.forEach(({ score, keypoints }) => {
            if (score >= sMinConfidence) {
                drawKeypoints(keypoints, sMinConfidence, ctx)
                drawSkeleton(keypoints, sMinConfidence, ctx)
            }
        })
    }

    const showDetectionsText = (predictions: Pose[]): void => {
        if (!webVideoRef.current || predictions?.length <= 0) {
            return
        }

        logger('predictions', predictions.length)
        // If you want to show more data, please revise it
        const data = predictions[0]
        webVideoRef.current.setPred(data)
        setJson(data)
    }

    const showPrediction = (predictions: Pose[]): void => {
        showDetectionsText(predictions)
        showDetections(predictions)
    }

    const handleJsonSave = (): void => {
        if (!sJson || !downloadRef.current) {
            return
        }
        logger('handleJsonSave')
        downloadJson(sJson, JSON_NAME, downloadRef.current)
    }

    /***********************
     * Render
     ***********************/

    return (
        <>
            <h1>姿态识别 Posenet</h1>
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

export default PoseNetPanel
