import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Card, Col, Form, Row, Select, Tabs } from 'antd'

import { logger, STATUS, ILayerSelectOption, loggerError } from '../../utils'
import { layout, MOBILENET_IMAGE_SIZE, MOBILENET_MODEL_PATH, tailLayout } from '../../constant'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import ImageUploadWidget from '../common/tensor/ImageUploadWidget'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'
import WebCamera, { IWebCameraHandler } from '../common/tensor/WebCamera'
import { ImagenetClasses } from './ImagenetClasses'

const { Option } = Select
const { TabPane } = Tabs

const MobilenetClassifier = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTabCurrent, setTabCurrent] = useState<number>(5)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [sCurLayer, setCurLayer] = useState<tf.layers.Layer>()

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

        let model: tf.LayersModel
        tf.loadLayersModel(MOBILENET_MODEL_PATH).then(
            (mobilenet) => {
                model = mobilenet

                // Warmup the model. This isn't necessary, but makes the first prediction
                // faster. Call `dispose` to release the WebGL memory allocated for the return
                // value of `predict`.
                const temp = model.predict(tf.zeros([1, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE, 3])) as tf.Tensor
                temp.dispose()

                setModel(model)

                const layerOptions: ILayerSelectOption[] = model?.layers.map((l, index) => {
                    return { name: l.name, index }
                })
                setLayersOption(layerOptions)

                setStatus(STATUS.LOADED)
            },
            loggerError
        )

        return () => {
            logger('Model Dispose')
            model?.dispose()
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
            // let _sample = tf.image.resizeBilinear(imageTensor as tf.Tensor4D, [224, 224])
            //
            // const [_sampleMax] = _sample.max().dataSync()
            // const [_sampleMin] = _sample.min().dataSync()
            // logger('_sampleMax', _sampleMax, _sampleMin)
            // if (_sampleMax > 1) {
            //     logger('[0, 255]')
            //     // const offset = tf.scalar(127.5)
            //     // Normalize the image from [0, 255] to [-1, 1].
            //     _sample = _sample.sub(127.5).div(127.5)
            // } else if (_sampleMin > 0) {
            //     // logger('[0, 1]')
            //     // _sample = _sample.sub(0.5).mul(2)
            // } else if (_sampleMin < 0) {
            //     logger('[-1, 1]')
            //     // do nothing
            // }

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
            invisiblePanes={[AIProcessTabPanes.TRAIN]}>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/ai/mobilenet.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <h2> 1000 Classes of ImageNet </h2>
                {Object.keys(ImagenetClasses).map((key, index) => `[ ${key} : ${ImagenetClasses[index]} ], `)}
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Row>
                    <Col span={12}>
                        <Card title='Mobilenet Model Info' style={{ margin: '8px' }} size='small'>
                            <TfvisModelWidget model={sModel}/>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Show Layer' style={{ margin: '8px' }} size='small'>
                            <Form {...layout} initialValues={{
                                layer: 0
                            }}>
                                <Form.Item name='layer' label='Show Layer'>
                                    <Select onChange={handleLayerChange} >
                                        {sLayersOption?.map((v) => {
                                            return <Option key={v.index} value={v.index}>{v.name}</Option>
                                        })}
                                    </Select>
                                </Form.Item>
                                <Form.Item {...tailLayout}>
                                    <p>status: {sStatus}</p>
                                    <p>backend: {sTfBackend}</p>
                                </Form.Item>
                            </Form>
                        </Card>
                        <Card title='Layer Info' style={{ margin: '8px' }} size='small'>
                            <TfvisLayerWidget layer={sCurLayer}/>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.PREDICT}>
                <Row>
                    <Col span={12}>
                        <Card title='Prediction with picture' style={{ margin: '8px' }} size='small'>
                            <ImageUploadWidget model={sModel} onSubmit={handlePredict} prediction={sPredictResult}/>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Prediction with camera' style={{ margin: '8px' }} size='small'>
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
