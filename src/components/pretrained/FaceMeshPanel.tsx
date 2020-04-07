import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { AnnotatedPrediction, FaceMesh, load as faceMeshLoad } from '@tensorflow-models/facemesh'
import { Col, Row, Switch } from 'antd'

import { logger, loggerError, STATUS } from '../../utils'
import WebVideo, { IWebVideoHandler } from '../common/tensor/WebVideo'
import { TRIANGULATION } from './triangulation'
import { drawPath, drawPoint } from './pretrainedUtils'

const FaceMeshPanel = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sModel, setModel] = useState<FaceMesh>()

    const webvideoRef = useRef<IWebVideoHandler>(null)

    const switchRef = useRef<boolean>(false)

    /***********************
     * useEffect
     ***********************/
    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.WAITING)

        let _model: FaceMesh
        faceMeshLoad().then(
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

    const handlePredict = async (data: tf.Tensor3D): Promise<AnnotatedPrediction[]> => {
        if (!sModel) {
            return []
        }
        return sModel.estimateFaces(data)
    }

    const showDetections = (predictions: AnnotatedPrediction[]): void => {
        if (!webvideoRef.current || predictions.length <= 0) {
            return
        }

        const ctx = webvideoRef.current.getContext()
        if (!ctx) {
            return
        }
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
        const keypoints = predictions[0].scaledMesh as number[][]

        if (switchRef.current) {
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

    const showDetectionsText = (predictions: AnnotatedPrediction[]): void => {
        if (!webvideoRef.current || predictions.length <= 0) {
            return
        }

        const data = (predictions[0].scaledMesh as number[][]).map((point: number[]) => {
            return point.map((v: number) => +v.toFixed(2))
        })
        webvideoRef.current.setPred(data)
    }

    const showPrediction = (predictions: AnnotatedPrediction[]): void => {
        showDetectionsText(predictions)
        showDetections(predictions)
    }

    const handleSwitch = (value: boolean): void => {
        // logger('handleSwitch', value)
        switchRef.current = value
    }

    /***********************
     * Render
     ***********************/

    return (
        <>
            <h1>面部特征 Face Mesh</h1>
            <Row>
                <Col span={12}>
                </Col>
                <Col span={12}>
                    <div className='centerContainer'>
                        <Switch onChange={handleSwitch}/>
                    </div>
                </Col>
                <Col span={24}>
                    <div className='centerContainer'>
                        <WebVideo ref={webvideoRef} show={showPrediction} predict={handlePredict}/>
                    </div>
                </Col>
                <Col span={12}>
                    <div>Status : {sStatus}</div>
                    <div>TfBackend : {sTfBackend}</div>
                </Col>
            </Row>
        </>
    )
}

export default FaceMeshPanel
