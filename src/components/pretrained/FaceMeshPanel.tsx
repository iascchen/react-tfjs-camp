import React, { useCallback, useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { AnnotatedPrediction, FaceMesh, load as faceMeshLoad } from '@tensorflow-models/facemesh'
import { Button, Col, Row, Switch } from 'antd'

import { logger, loggerError, STATUS } from '../../utils'
import WebVideo, { IWebVideoHandler } from '../common/tensor/WebVideo'
import { TRIANGULATION } from './triangulation'
import { downloadJson, drawPath, drawPoint } from './pretrainedUtils'

const JSON_NAME = 'face-mesh.json'

const FaceMeshPanel = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sModel, setModel] = useState<FaceMesh>()

    const switchRef = useRef<boolean>(false)
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

    const handlePredict = useCallback(async (data: tf.Tensor3D): Promise<AnnotatedPrediction[]> => {
        if (!sModel) {
            return []
        }
        return sModel.estimateFaces(data)
    }, [sModel])

    /***********************
     * Functions
     ***********************/
    const showDetections = (predictions: AnnotatedPrediction[]): void => {
        if (!webVideoRef.current || predictions?.length <= 0) {
            return
        }

        const ctx = webVideoRef.current.getContext()
        if (!ctx) {
            return
        }
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

        predictions.forEach((face: AnnotatedPrediction) => {
            const keyPoints = face.scaledMesh as number[][]

            if (switchRef.current) {
                for (let i = 0; i < TRIANGULATION.length / 3; i++) {
                    const points = [
                        TRIANGULATION[i * 3], TRIANGULATION[i * 3 + 1],
                        TRIANGULATION[i * 3 + 2]
                    ].map(index => keyPoints[index])

                    drawPath(ctx, points, true)
                }
            } else {
                for (let i = 0; i < keyPoints.length; i++) {
                    const x = keyPoints[i][0]
                    const y = keyPoints[i][1]

                    drawPoint(ctx, y, x, 1)
                }
            }
        })
    }

    const showDetectionsText = (predictions: AnnotatedPrediction[]): void => {
        if (!webVideoRef.current || predictions?.length <= 0) {
            return
        }

        logger('predictions', predictions.length)
        // If you want to show more data, please revise it
        const data = predictions[0]
        webVideoRef.current.setPred(data)
        setJson(data)
    }

    const showPrediction = (predictions: AnnotatedPrediction[]): void => {
        showDetectionsText(predictions)
        showDetections(predictions)
    }

    const handleSwitch = (value: boolean): void => {
        // logger('handleSwitch', value)
        switchRef.current = value
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
            <h1>面部特征 Face Mesh</h1>
            <Row>
                <Col span={12}>
                </Col>
                <Col span={12}>
                    <div className='centerContainer'>
                        <Button style={{ width: '30%', margin: '0 10%' }} onClick={handleJsonSave}
                            disabled={sStatus === STATUS.WAITING} >Save Json</Button>
                        <Switch onChange={handleSwitch}/>
                    </div>
                </Col>
                <Col span={24}>
                    <div className='centerContainer'>
                        <WebVideo ref={webVideoRef} show={showPrediction} predict={handlePredict}/>
                    </div>
                </Col>
                <Col span={12}>
                    <div>Status : {sStatus}</div>
                    <div>TfBackend : {sTfBackend}</div>
                </Col>
            </Row>
            <a ref={downloadRef}/>
        </>
    )
}

export default FaceMeshPanel
