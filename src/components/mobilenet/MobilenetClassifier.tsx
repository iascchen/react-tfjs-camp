import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Card, Col, Row, Select, Tabs } from 'antd'

import { logger, STATUS, ILayerSelectOption } from '../../utils'
import { MOBILENET_IMAGE_SIZE, MOBILENET_MODEL_PATH } from '../../constant'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import ImageUploadWidget from '../common/tensor/ImageUploadWidget'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'
import WebCamera, { IWebCameraHandler } from '../common/tensor/WebCamera'

const { Option } = Select
const { TabPane } = Tabs

const MobilenetClassifier = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTabCurrent, setTabCurrent] = useState<number>(1)

    const [tfBackend, setTfBackend] = useState<string>()
    const [status, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sModel, setModel] = useState<tf.LayersModel>()
    const [layersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [curLayer, setCurLayer] = useState<tf.layers.Layer>()

    const [sPredictResult, setPredictResult] = useState<tf.Tensor>()

    const webcamRef = useRef<IWebCameraHandler>(null)

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.LOADING)

        let _model: tf.LayersModel
        tf.loadLayersModel(MOBILENET_MODEL_PATH).then(
            (mobilenet) => {
                _model = mobilenet

                // Warmup the model. This isn't necessary, but makes the first prediction
                // faster. Call `dispose` to release the WebGL memory allocated for the return
                // value of `predict`.
                const _temp = _model.predict(tf.zeros([1, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE, 3])) as tf.Tensor
                _temp.dispose()

                setModel(_model)

                const _layerOptions: ILayerSelectOption[] = _model?.layers.map((l, index) => {
                    return { name: l.name, index }
                })
                setLayersOption(_layerOptions)

                setStatus(STATUS.LOADED)
            },
            (error) => {
                logger(error)
            }
        )

        return () => {
            logger('Model Dispose')
            _model?.dispose()
        }
    }, [])

    /***********************
     * Functions
     ***********************/

    const handlePredict = (imageTensor: tf.Tensor): void => {
        if (!imageTensor) {
            return
        }
        const [p] = tf.tidy(() => {
            const _sample = tf.image.resizeBilinear(imageTensor as tf.Tensor4D, [224, 224])
            const offset = tf.scalar(127.5)
            // Normalize the image from [0, 255] to [-1, 1].
            const normalized = _sample.sub(offset).div(offset)
            // Reshape to a single-element batch so we can pass it to predict.
            const batched = normalized.reshape([1, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE, 3])

            const result = sModel?.predict(batched) as tf.Tensor
            logger(result)

            const p = result?.argMax(-1)
            return [p]
        })

        setPredictResult(p)
    }

    const handleLayerChange = (value: number): void => {
        logger('handleLayerChange', value)
        const _layer = sModel?.getLayer(undefined, value)
        setCurLayer(_layer)
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    /***********************
     * Render
     ***********************/

    return (
        <AIProcessTabs title={'Mobilenet Classifier'} current={sTabCurrent} onChange={handleTabChange}
            invisiblePanes={[AIProcessTabPanes.DATA, AIProcessTabPanes.TRAIN]}>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/mobilenet.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Row>
                    <Col span={12}>
                        <Card title='Mobilenet Model' style={{ margin: '8px' }} size='small'>
                            <TfvisModelWidget model={sModel}/>
                            <p>status: {status}</p>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Layers' style={{ margin: '8px' }} size='small'>
                            Select Layer : <Select onChange={handleLayerChange} defaultValue={0}>
                                {layersOption?.map((v) => {
                                    return <Option key={v.index} value={v.index}>{v.name}</Option>
                                })}
                            </Select>
                            <TfvisLayerWidget layer={curLayer}/>
                            <p>backend: {tfBackend}</p>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.PREDICT}>
                <Row>
                    <Col span={12}>
                        <Card title='Predict' style={{ margin: '8px' }} size='small'>
                            <ImageUploadWidget model={sModel} onSubmit={handlePredict} prediction={sPredictResult}/>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Prediction' size='small'>
                            <WebCamera ref={webcamRef} model={sModel} onSubmit={handlePredict} prediction={sPredictResult}
                                isPreview />
                        </Card>
                    </Col>
                </Row>
            </TabPane>
        </AIProcessTabs>
    )
}

export default MobilenetClassifier
