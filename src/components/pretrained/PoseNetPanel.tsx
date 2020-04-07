import React, { useEffect, useRef, useState } from 'react'
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
import { Col, Row } from 'antd'

import { logger, loggerError, STATUS } from '../../utils'
import WebVideo, { IWebVideoHandler } from '../common/tensor/WebVideo'
import { drawPoint, drawSegment } from './pretrainedUtils'

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

    const webvideoRef = useRef<IWebVideoHandler>(null)

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

    /***********************
     * Functions
     ***********************/
    const handlePredict = async (data: tf.Tensor3D): Promise<Pose[]> => {
        if (!sModel) {
            return []
        }
        const poses = await sModel.estimatePoses(data, PREDICT_CONFIG)
        return poses
    }

    const showDetections = (predictions: Pose[]): void => {
        if (!webvideoRef.current || predictions.length <= 0) {
            return
        }

        const ctx = webvideoRef.current.getContext()
        if (!ctx) {
            return
        }
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
        const poses = predictions

        poses.forEach(({ score, keypoints }) => {
            logger(score)

            if (score >= sMinConfidence) {
                drawKeypoints(keypoints, sMinConfidence, ctx)
                drawSkeleton(keypoints, sMinConfidence, ctx)
            }
        })
    }

    const showDetectionsText = (predictions: Pose[]): void => {
        if (!webvideoRef.current || predictions.length <= 0) {
            return
        }

        const data = predictions.map((pose: Pose) => {
            return pose.keypoints.map((point: Keypoint) => {
                return [+point.position.x.toFixed(2), +point.position.y.toFixed(2)]
            })
        })
        webvideoRef.current.setPred(data)
    }

    const showPrediction = (predictions: any): void => {
        showDetectionsText(predictions)
        showDetections(predictions)
    }

    /***********************
     * Render
     ***********************/

    return (
        <>
            <h1>姿态识别 Posenet</h1>
            <Row>
                <Col span={24}>
                    <div className='centerContainer'>
                        <WebVideo ref={webvideoRef} show={showPrediction} predict={handlePredict}/>
                    </div>
                </Col>
                <Col span={24}>
                    <div>Status : {sStatus}</div>
                    <div>TfBackend : {sTfBackend}</div>
                </Col>
            </Row>
        </>
    )
}

export default PoseNetPanel
