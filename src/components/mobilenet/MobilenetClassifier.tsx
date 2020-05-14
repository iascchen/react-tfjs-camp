import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Card, Col, Form, Row, Select, Tabs } from 'antd'

import { ILayerSelectOption, logger, loggerError, STATUS } from '../../utils'
import { layout, tailLayout } from '../../constant'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import ImageUploadWidget from '../common/tensor/ImageUploadWidget'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'
import WebCamera, { IWebCameraHandler } from '../common/tensor/WebCamera'

import { formatImageForMobileNet, MOBILENET_IMAGE_SIZE, MOBILENET_MODEL_PATH } from './mobilenetUtils'
import ImageNetTagsWidget from './ImageNetTagsWidget'

const { Option } = Select
const { TabPane } = Tabs

const MobileNetClassifier = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTabCurrent, setTabCurrent] = useState<number>(3)

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

        setStatus(STATUS.WAITING)

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
            const batched = formatImageForMobileNet(imageTensor)
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
        <AIProcessTabs title={'MobileNet Classifier'} current={sTabCurrent} onChange={handleTabChange}
            invisiblePanes={[AIProcessTabPanes.TRAIN]}>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/ai/mobilenet.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <ImageNetTagsWidget />
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Row>
                    <Col span={12}>
                        <Card title='MobileNet Model Info' style={{ margin: '8px' }} size='small'>
                            <TfvisModelWidget model={sModel}/>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Show Layer' style={{ margin: '8px' }} size='small'>
                            <Form {...layout} initialValues={{
                                layer: 0
                            }}>
                                <Form.Item name='layer' label='Show Layer'>
                                    <Select onChange={handleLayerChange}>
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
                            <ImageUploadWidget onSubmit={handlePredict} prediction={sPredictResult}/>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Prediction with camera' style={{ margin: '8px' }} size='small'>
                            <WebCamera ref={webcamRef} onSubmit={handlePredict} prediction={sPredictResult} isPreview/>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
        </AIProcessTabs>
    )
}

export default MobileNetClassifier
