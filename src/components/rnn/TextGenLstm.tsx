import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Form, Input, message, Row, Select, Tabs } from 'antd'

import { layout, tailLayout } from '../../constant'
import { fetchResource, ILayerSelectOption, logger, loggerError, STATUS } from '../../utils'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'

import { TEXT_DATA_URLS, TextData } from './dataTextGen'
import { LSTMTextGenerator, SaveableLSTMTextGenerator } from './TextGenerator'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

const { Option } = Select
const { TabPane } = Tabs
const { TextArea } = Input

const sampleLen = 40
const sampleStep = 3

const LSTM_LAYER_SIZE = [[128], [100, 50]]

const TextGenLstm = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTabCurrent, setTabCurrent] = useState<number>(4)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sLstmLayerSizes, setLstmLayerSizes] = useState<number[]>([128]) // [100，50]
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [sCurLayer, setCurLayer] = useState<tf.layers.Layer>()

    const [sEpochs] = useState<number>(10)
    const [sExamplesPerEpoch] = useState<number>(2048)
    const [sBatchSize] = useState<number>(128)
    const [sValidationSplit] = useState<number>(0.0625)
    const [sLearningRate] = useState<number>(1e-2)

    const [sDataIdentifier, setDataIdentifier] = useState<string>('nietzsche')
    const [sTextString, setTextString] = useState<string>('')

    const [sGenerator, setGenerator] = useState<LSTMTextGenerator>()
    const [sGenTextLen] = useState<number>(200)
    const [sTemperature] = useState<number>(0.75)
    const [sSeedText, setSeedText] = useState<string>()
    const [sGenText, setGenText] = useState<string>()

    const [sTextData, setTextData] = useState<TextData>()

    const historyRef = useRef<HTMLDivElement>(null)

    const [form] = Form.useForm()
    const [formPredict] = Form.useForm()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init data set ...')

        const url = TEXT_DATA_URLS[sDataIdentifier].url
        logger('url', url)

        setStatus(STATUS.WAITING)
        fetchResource(url).then(
            (buffer) => {
                const textString = buffer.toString()
                setTextString(textString)
                const dataHandler = new TextData(sDataIdentifier, textString, sampleLen, sampleStep)
                setTextData(dataHandler)

                setStatus(STATUS.LOADED)
            },
            loggerError
        )

        return () => {
            logger('Data Dispose')
            // dataHandler.dispose()
        }
    }, [sDataIdentifier])

    useEffect(() => {
        if (!sTextData) {
            return
        }

        logger('init model ...')
        setStatus(STATUS.WAITING)

        const generator = new SaveableLSTMTextGenerator(sTextData)
        setGenerator(generator)
        generator.createModel(sLstmLayerSizes)

        const _model = generator.model
        if (!_model) {
            return
        }

        // generator.compileModel(sLearningRate)
        setModel(_model)

        const _layerOptions: ILayerSelectOption[] = _model.layers.map((l, index) => {
            return { name: l.name, index }
        })
        setLayersOption(_layerOptions)

        setStatus(STATUS.LOADED)

        return () => {
            logger('Model Dispose')
            _model?.dispose()
        }
    }, [sTextData, sLstmLayerSizes])

    /***********************
     * Functions
     ***********************/

    const handleDataSourceChange = (value: string): void => {
        logger('handleDataSourceChange', value)
        setDataIdentifier(value)
    }

    const handleLstmLayerSizeChange = (value: number): void => {
        logger('handleLstmLayerSizeChange', value)
        setLstmLayerSizes(LSTM_LAYER_SIZE[value])
    }

    const handleLoadModel = (): void => {
        // TODO : Load saved model
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        message.info('TODO: Not Implemented')
    }

    const handleSaveModel = (): void => {
        // TODO : Load saved model
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        message.info('TODO: Not Implemented')
    }

    const handleLayerChange = (value: number): void => {
        logger('handleLayerChange', value)
        const _layer = sModel?.getLayer(undefined, value)
        setCurLayer(_layer)
    }

    const handlePredict = (values: any): void => {
        if (!sGenerator || !sTextData) {
            return
        }
        logger('handlePredict', values)
        setStatus(STATUS.WAITING)

        const { seedText, genTextLen, temperature } = values

        let seedSentence
        let seedSentenceIndices
        if (seedText.length === 0) {
            // Seed sentence is not specified yet. Get it from the data.
            [seedSentence, seedSentenceIndices] = sTextData.getRandomSlice()
            setSeedText(seedSentence)
        } else {
            seedSentence = seedText
            if (seedSentence.length < sTextData.sampleLen()) {
                return
            }
            seedSentence = seedSentence.slice(seedSentence.length - sTextData.sampleLen(), seedSentence.length)
            seedSentenceIndices = sTextData.textToIndices(seedSentence)
        }

        // const sentenceIndices = seedText
        sGenerator.generateText(seedSentenceIndices, genTextLen, temperature).then(
            (result) => {
                if (result) {
                    setGenText(result)
                    setStatus(STATUS.PREDICTED)
                }
            },
            loggerError
        )
    }

    const handleTrain = (values: any): void => {
        if (!sGenerator) {
            return
        }
        logger('handleTrain', values)
        setStatus(STATUS.WAITING)

        const { epochs, examplesPerEpoch, batchSize, validationSplit, learningRate } = values
        sGenerator.compileModel(learningRate)
        const callbacks = tfvis.show.fitCallbacks(historyRef?.current, ['loss', 'acc', 'val_loss', 'val_acc'])
        sGenerator.fitModel(epochs, examplesPerEpoch, batchSize, validationSplit, callbacks).then(
            () => {
                setStatus(STATUS.TRAINED)
            },
            loggerError
        )
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    /***********************
     * Render
     ***********************/

    return (
        <AIProcessTabs title={'LSTM Text Generator'} current={sTabCurrent} onChange={handleTabChange} >
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/lstm.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={12}>
                        <Card title='Source Data' size='small' style={{ margin: '8px' }}>
                            Select Source Text : <Select onChange={handleDataSourceChange} defaultValue={'nietzsche'} style={{ marginLeft: 16 }}>
                                {
                                    Object.keys(TEXT_DATA_URLS).map((key: string) => {
                                        return <Option key={key} value={key}>{TEXT_DATA_URLS[key].needle}</Option>
                                    })
                                }
                            </Select> {sStatus}
                            <TextArea rows={20} value={sTextString}></TextArea>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Data Info' size='small' style={{ margin: '8px' }}>
                            <p>Sample Len : {sTextData?.sampleLen()}</p>
                            <p>Text Len : {sTextData?.textLen()}</p>
                            <p>Char Set Size : {sTextData?.charSetSize()}</p>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Row>
                    <Col span={24}>
                        <Card title='Create LSTM Model' style={{ margin: '8px' }} size='small'>
                            Select LSTM Layer Size : <Select onChange={handleLstmLayerSizeChange} defaultValue={0} style={{ marginLeft: 16 }}>
                                {
                                    LSTM_LAYER_SIZE.map((value, index) => {
                                        return <Option key={index} value={index}>{value.join(',')}</Option>
                                    })
                                }
                            </Select> {sStatus}
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Model Detail' style={{ margin: '8px' }} size='small'>
                            <TfvisModelWidget model={sModel}/>
                            <p>status: {sStatus}</p>
                            <p>backend: {sTfBackend}</p>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Layers Detail' style={{ margin: '8px' }} size='small'>
                                Select Layer : <Select onChange={handleLayerChange} defaultValue={0}>
                                {sLayersOption?.map((v) => {
                                    return <Option key={v.index} value={v.index}>{v.name}</Option>
                                })}
                            </Select>
                            <TfvisLayerWidget layer={sCurLayer}/>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                <Row>
                    <Col span={12}>
                        <Card title='Adjust Super Params' style={{ margin: '8px' }} size='small'>
                            <Form {...layout} form={form} onFinish={handleTrain} initialValues={{
                                epochs: sEpochs,
                                examplesPerEpoch: sExamplesPerEpoch,
                                batchSize: sBatchSize,
                                validationSplit: sValidationSplit,
                                learningRate: sLearningRate
                            }}>
                                <Form.Item name='epochs' label='Epoches' >
                                    <Input />
                                </Form.Item>
                                <Form.Item name='examplesPerEpoch' label='Examples Per Epoch' >
                                    <Input />
                                </Form.Item>
                                <Form.Item name='batchSize' label='Batch Size'>
                                    <Input />
                                </Form.Item>
                                <Form.Item name='validationSplit' label='Validation Split'>
                                    <Input />
                                </Form.Item>
                                <Form.Item name='learningRate' label='Learning Rate'>
                                    <Input />
                                </Form.Item>
                                <Form.Item {...tailLayout}>
                                    <Button type='primary' htmlType='submit' style={{ width: '30%', margin: '0 10%' }}> Train </Button>
                                </Form.Item>
                            </Form>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Save and Load model weights' style={{ margin: '8px' }} size='small'>
                            <Button onClick={handleSaveModel} style={{ width: '30%', margin: '0 10%' }}> Save </Button>
                            <Button onClick={handleLoadModel} style={{ width: '30%', margin: '0 10%' }}> Load </Button>
                            <div>status: {sStatus}</div>
                            <div>backend: {sTfBackend}</div>
                        </Card>
                        <Card title='Training History' style={{ margin: '8px' }} size='small'>
                            <div ref={historyRef} />
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.PREDICT}>
                <Row>
                    <Col span={12}>
                        <Card title='Prediction' style={{ margin: '8px' }} size='small'>
                            <Form {...layout} form={formPredict} onFinish={handlePredict} initialValues={{
                                genTextLen: sGenTextLen,
                                temperature: sTemperature,
                                seedText: sSeedText
                            }}>
                                <Form.Item name='genTextLen' label='Length of generated text' >
                                    <Input />
                                </Form.Item>
                                <Form.Item name='temperature' label='Generation temperature' >
                                    <Input />
                                </Form.Item>
                                <Form.Item name='seedText' label='Seed Text'>
                                    <TextArea rows={10} />
                                </Form.Item>
                                <Form.Item {...tailLayout}>
                                    <Button type='primary' htmlType='submit' style={{ width: '50%', margin: '0 10%' }}> Generate Text </Button>
                                </Form.Item>
                            </Form>
                            <p> {sStatus} </p>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Predict Result' style={{ margin: '8px' }} size='small'>
                            <p>Model Output : {sGenText} </p>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
        </AIProcessTabs>
    )
}

export default TextGenLstm
