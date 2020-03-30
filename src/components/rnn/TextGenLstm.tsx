import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Form, Input, Row, Select, Slider, Tabs } from 'antd'

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

const SAMPLE_LEN = 60
const SAMPLE_STEP = 3

const LSTM_LAYER_SIZE = [[128], [128, 128], [256, 128]]

const BATCH_SIZES = [128, 256, 512, 1024]
const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03]

const TextGenLstm = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTabCurrent, setTabCurrent] = useState<number>(4)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    // Data
    const [sDataIdentifier, setDataIdentifier] = useState<string>('nietzsche')
    const [sTextString, setTextString] = useState<string>('')

    // Model
    const [sLstmLayerSizes, setLstmLayerSizes] = useState<number[]>([256, 128])
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [sCurLayer, setCurLayer] = useState<tf.layers.Layer>()

    // Train
    const stopRef = useRef(false)
    const [sEpochs, setEpochs] = useState<number>(50)
    const [sExamplesPerEpoch, setExamplesPerEpoch] = useState<number>(5000)
    const [sBatchSize, setBatchSize] = useState<number>(128)
    const [sValidationSplit, setValidationSplit] = useState<number>(0.05)
    const [sLearningRate, setLearningRate] = useState<number>(1e-2)

    const [sGenerator, setGenerator] = useState<LSTMTextGenerator>()
    const [sGenTextLen] = useState<number>(200)
    const [sTemperature] = useState<number>(0.75)
    const [sSeedText, setSeedText] = useState<string>()
    const [sGenText, setGenText] = useState<string>()

    const [sTextData, setTextData] = useState<TextData>()

    const historyRef = useRef<HTMLDivElement>(null)

    // const [form] = Form.useForm()
    const [formTrain] = Form.useForm()
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
                const dataHandler = new TextData(sDataIdentifier, textString, SAMPLE_LEN, SAMPLE_STEP)
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

        tf.backend()
        setTfBackend(tf.getBackend())

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

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    const handleDataSourceChange = (value: string): void => {
        logger('handleDataSourceChange', value)
        setDataIdentifier(value)
    }

    const handleLstmLayerSizeChange = (value: number): void => {
        logger('handleLstmLayerSizeChange', value)
        setLstmLayerSizes(LSTM_LAYER_SIZE[value])
    }

    const handleLayerChange = (value: number): void => {
        logger('handleLayerChange', value)
        const _layer = sModel?.getLayer(undefined, value)
        setCurLayer(_layer)
    }

    const handleTrainParamsChange = (): void => {
        const values = formTrain.getFieldsValue()
        // logger('handleTrainParamsChange', values)
        const { epochs, batchSize, learningRate, validationSplit, examplesPerEpoch } = values
        setEpochs(epochs)
        setExamplesPerEpoch(examplesPerEpoch)
        setBatchSize(batchSize)
        setLearningRate(learningRate)
        setValidationSplit(validationSplit)
    }

    const myCallback = {
        onBatchBegin: async (batch: number) => {
            if (!sModel) {
                return
            }
            if (sModel && stopRef.current) {
                logger('Checked stop', stopRef.current)
                setStatus(STATUS.STOPPED)
                sModel.stopTraining = stopRef.current
            }
            await tf.nextFrame()
        }
    }

    const handleTrain = (values: any): void => {
        if (!sGenerator) {
            return
        }
        logger('handleTrain', values)
        setStatus(STATUS.WAITING)
        stopRef.current = false

        const { epochs, examplesPerEpoch, batchSize, validationSplit, learningRate } = values
        sGenerator.compileModel(learningRate)

        const callbacks = [
            tfvis.show.fitCallbacks(historyRef.current, ['loss', 'acc', 'val_loss', 'val_acc']),
            myCallback
        ]

        sGenerator.fitModel(epochs, examplesPerEpoch, batchSize, validationSplit, callbacks).then(
            () => {
                setStatus(STATUS.TRAINED)
            },
            loggerError
        )
    }

    const handleTrainStop = (): void => {
        logger('handleTrainStop')
        stopRef.current = true
    }

    const handleLoadModelWeight = (): void => {
        setStatus(STATUS.WAITING)
        const fileName = `lstm_${sDataIdentifier}`
        tf.loadLayersModel(`/model/${fileName}.json`).then(
            (model) => {
                model.summary()
                setModel(model)
                setStatus(STATUS.LOADED)
            },
            loggerError
        )
    }

    const handleSaveModelWeight = (): void => {
        if (!sModel) {
            return
        }

        // Save Model
        const fileName = `lstm_${sDataIdentifier}`
        const downloadUrl = `downloads://${fileName}`
        sModel.save(downloadUrl).then((saveResults) => {
            logger(saveResults)
        }, loggerError)
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

    /***********************
     * Render
     ***********************/

    const dataAdjustCard = (): JSX.Element => {
        return (
            <Card title='Source Text' size='small' style={{ margin: '8px' }}>
                <Form {...layout} initialValues={{ dataSource: 'nietzsche' }} >
                    <Form.Item name='dataSource' label='Text Source'>
                        <Select onChange={handleDataSourceChange}>
                            {
                                Object.keys(TEXT_DATA_URLS).map((key: string) => {
                                    return <Option key={key} value={key}>{TEXT_DATA_URLS[key].needle}</Option>
                                })
                            }
                        </Select>
                    </Form.Item>
                    <Form.Item label='Status Info'>
                        <div>{sStatus}</div>
                        <div>Sample Len : {sTextData?.sampleLen()}</div>
                        <div>Text Len : {sTextData?.textLen()}</div>
                        <div>Char Set Size : {sTextData?.charSetSize()}</div>
                    </Form.Item>
                </Form>
            </Card>
        )
    }

    const modelAdjustCard = (): JSX.Element => {
        return (
            <Card title='Adjust LSTM Model' style={{ margin: '8px' }} size='small'>
                <Form {...layout}>
                    <Form.Item name='layerSize' label='LSTM Layer Size'>
                        <Select onChange={handleLstmLayerSizeChange} defaultValue={0} >
                            {
                                LSTM_LAYER_SIZE.map((value, index) => {
                                    return <Option key={index} value={index}>{value.join(',')}</Option>
                                })
                            }
                        </Select>
                    </Form.Item>
                    <Form.Item label='Status'>
                        <div>{sStatus}</div>
                        <div>backend: {sTfBackend}</div>
                    </Form.Item>
                </Form>
            </Card>
        )
    }

    const trainAdjustCard = (): JSX.Element => {
        return (
            <Card title='Train' style={{ margin: '8px' }} size='small'>
                <Form {...layout} form={formTrain} onFinish={handleTrain} onFieldsChange={handleTrainParamsChange}
                    initialValues={{
                        epochs: sEpochs,
                        examplesPerEpoch: sExamplesPerEpoch,
                        batchSize: sBatchSize,
                        validationSplit: sValidationSplit,
                        learningRate: sLearningRate
                    }}>
                    <Form.Item name='epochs' label='Epochs'>
                        <Slider min={50} max={150} step={25} marks={{ 50: 50, 100: 100, 150: 150 }}/>
                    </Form.Item>
                    <Form.Item name='examplesPerEpoch' label='Examples Per Epoch' >
                        <Slider min={5000} max={15000} step={2500} marks={{ 5000: 5000, 10000: 10000, 15000: 15000 }}/>
                    </Form.Item>
                    <Form.Item name='batchSize' label='Batch Size'>
                        <Select>
                            {BATCH_SIZES.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item name='validationSplit' label='Validation Split'>
                        <Slider min={0.05} max={0.15} step={0.05} marks={{ 0.05: 0.05, 0.10: 0.10, 0.15: 0.15 }}/>
                    </Form.Item>
                    <Form.Item name='learningRate' label='Learning Rate'>
                        <Select>
                            {LEARNING_RATES.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <Button type='primary' htmlType={'submit'} style={{ width: '30%', margin: '0 10%' }}
                            disabled={sStatus === STATUS.WAITING}> Train </Button>
                        <Button onClick={handleTrainStop} style={{ width: '30%', margin: '0 10%' }}> Stop </Button>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <Button onClick={handleSaveModelWeight} style={{ width: '30%', margin: '0 10%' }}> Save
                            Weights </Button>
                        <Button onClick={handleLoadModelWeight} style={{ width: '30%', margin: '0 10%' }}> Load
                            Weights </Button>
                    </Form.Item>
                </Form>
            </Card>
        )
    }

    return (
        <AIProcessTabs title={'LSTM Text Generator'} current={sTabCurrent} onChange={handleTabChange} >
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/lstm.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={12}>
                        {dataAdjustCard()}
                    </Col>
                    <Col span={12}>
                        <Card title='Text Data' size='small' style={{ margin: '8px' }}>
                            <TextArea rows={20} value={sTextString}></TextArea>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Row>
                    <Col span={12}>
                        {modelAdjustCard()}
                        <Card title='Layers Detail' style={{ margin: '8px' }} size='small'>
                            <Form {...layout}>
                                <Form.Item name='layerSize' label='Select Layer'>
                                    <Select onChange={handleLayerChange} defaultValue={0}>
                                        {sLayersOption?.map((v) => {
                                            return <Option key={v.index} value={v.index}>{v.name}</Option>
                                        })}
                                    </Select>
                                </Form.Item>
                            </Form>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Model Detail' style={{ margin: '8px' }} size='small'>
                            <TfvisModelWidget model={sModel}/>
                            <p>backend: {sTfBackend}</p>
                        </Card>
                        <Card title='Layer Detail' style={{ margin: '8px' }} size='small'>
                            <TfvisLayerWidget layer={sCurLayer}/>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                <Row>
                    <Col span={8}>
                        {trainAdjustCard()}
                        {modelAdjustCard()}
                        {dataAdjustCard()}
                    </Col>
                    <Col span={16}>
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
                                    <Slider min={100} max={300} step={50} marks={{ 100: 100, 200: 200, 300: 300 }}/>
                                </Form.Item>
                                <Form.Item name='temperature' label='Generation temperature' >
                                    <Slider min={0.25} max={1} step={0.25} marks={{ 0.25: 0.25, 0.75: 0.75, 1.25: 1.25 }}/>
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
