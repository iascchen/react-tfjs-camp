import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Form, Input, message, Row, Select, Tabs } from 'antd'

import { ILayerSelectOption, logger, loggerError, STATUS } from '../../utils'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import TfvisDatasetInfoWidget from '../common/tfvis/TfvisDatasetInfoWidget'

import { SentimentPredictor } from './modelSentiment'
import { loadData } from './dataSentiment'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

const { Option } = Select
const { TabPane } = Tabs
const { TextArea } = Input

const layout = {
    labelCol: { span: 8 },
    wrapperCol: { span: 16 }
}

const tailLayout = {
    wrapperCol: { offset: 8, span: 16 }
}

const MODEL_OPTIONS = ['pretrained-cnn', 'multihot', 'flatten', 'cnn', 'simpleRNN', 'lstm', 'bidirectionalLSTM']

interface IKeyMap {
    [index: string]: string
}
const exampleReviews: IKeyMap = {
    positive:
        'die hard mario fan and i loved this game br br this game starts slightly boring but trust me it\'s worth it as soon as you start your hooked the levels are fun and exiting they will hook you OOV your mind turns to mush i\'m not kidding this game is also orchestrated and is beautifully done br br to keep this spoiler free i have to keep my mouth shut about details but please try this game it\'ll be worth it br br story 9 9 action 10 1 it\'s that good OOV 10 attention OOV 10 average 10',
    negative:
        'the mother in this movie is reckless with her children to the point of neglect i wish i wasn\'t so angry about her and her actions because i would have otherwise enjoyed the flick what a number she was take my advise and fast forward through everything you see her do until the end also is anyone else getting sick of watching movies that are filmed so dark anymore one can hardly see what is being filmed as an audience we are impossibly involved with the actions on the screen so then why the hell can\'t we have night vision'
}

const SentimentWidget = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTabCurrent, setTabCurrent] = useState<number>(1)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sNumWords] = useState<number>(10000)
    const [sMaxLen] = useState<number>(100)
    const [sEmbeddingSize] = useState<number>(128)

    const [sPredictor, setPredictor] = useState<SentimentPredictor>()

    const [sModelName, setModelName] = useState('multihot')
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [sCurLayer, setCurLayer] = useState<tf.layers.Layer>()

    const [sEpochs] = useState<number>(10)
    const [sBatchSize] = useState<number>(128)
    const [sValidationSplit] = useState<number>(0.2)
    const [sLearningRate] = useState<number>(1e-2)

    const [sTrainSet, setTrainSet] = useState<tf.TensorContainerObject>()
    const [sTestSet, setTestSet] = useState<tf.TensorContainerObject>()

    const [sSampleType] = useState<string>('positive')
    const [sSampleText, setSampleText] = useState<string>(exampleReviews[sSampleType])
    const [sPredictResult, setPredictResult] = useState<tf.Tensor>()

    const historyRef = useRef<HTMLDivElement>(null)

    const [form] = Form.useForm()
    const [formPredict] = Form.useForm()

    /***********************
     * useEffect
     ***********************/
    useEffect(() => {
        logger('init data set ...')

        loadData(sNumWords, sMaxLen, (sModelName === 'multihot')).then(
            (result) => {
                const { xTrain, yTrain, xTest, yTest } = result
                setTrainSet({ xs: xTrain, ys: yTrain })
                setTestSet({ xs: xTest, ys: yTest })
            },
            loggerError
        )

        return () => {
            logger('Data Dispose')
            // dataHandler.dispose()
        }
    }, [sModelName])

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.LOADING)

        let predictor: SentimentPredictor
        let _model: tf.LayersModel
        if (sModelName === 'pretrained-cnn') {
            predictor = new SentimentPredictor()
            predictor.init().then(
                (model) => {
                    if (!model) {
                        // eslint-disable-next-line @typescript-eslint/no-floating-promises
                        message.error('Can not load model')
                        return
                    }

                    _model = model
                    setModel(_model)
                    setPredictor(predictor)

                    setStatus(STATUS.LOADED)
                },
                loggerError
            )
        } else {
            const model = tf.sequential()
            if (sModelName === 'multihot') {
                // A 'multihot' model takes a multi-hot encoding of all words in the
                // sentence and uses dense layers with relu and sigmoid activation functions
                // to classify the sentence.
                model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [sNumWords] }))
                model.add(tf.layers.dense({ units: 16, activation: 'relu' }))
            } else {
                // All other model types use word embedding.
                model.add(tf.layers.embedding({
                    inputDim: sNumWords,
                    outputDim: sEmbeddingSize,
                    inputLength: sMaxLen
                }))

                switch (sModelName) {
                    case 'flatten' :
                        model.add(tf.layers.flatten())
                        break
                    case 'cnn' :
                        model.add(tf.layers.dropout({ rate: 0.5 }))
                        model.add(tf.layers.conv1d({
                            filters: 250,
                            kernelSize: 5,
                            strides: 1,
                            padding: 'valid',
                            activation: 'relu'
                        }))
                        model.add(tf.layers.globalMaxPool1d({}))
                        model.add(tf.layers.dense({ units: 250, activation: 'relu' }))
                        break
                    case 'simpleRNN' :
                        model.add(tf.layers.simpleRNN({ units: 32 }))
                        break
                    case 'lstm' :
                        model.add(tf.layers.lstm({ units: 32 }))
                        break
                    case 'bidirectionalLSTM' :
                        model.add(tf.layers.bidirectional({
                            layer: tf.layers.lstm({ units: 32 }) as tf.layers.RNN,
                            mergeMode: 'concat'
                        }))
                        break
                }
            }
            model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }))

            model.compile({ loss: 'binaryCrossentropy', optimizer: 'adam', metrics: ['acc'] })
            // model.summary()
            _model = model
            setModel(_model)

            setStatus(STATUS.LOADED)
        }

        return () => {
            logger('Model Dispose')
            _model?.dispose()
        }
    }, [sModelName, sNumWords])

    useEffect(() => {
        if (!sModel) {
            return
        }

        const _layerOptions: ILayerSelectOption[] = sModel.layers.map((l, index) => {
            return { name: l.name, index }
        })
        setLayersOption(_layerOptions)
    }, [sModel])

    /***********************
     * Functions
     ***********************/

    const handlePredict = (values: any): void => {
        if (!sPredictor) {
            return
        }
        logger('handlePredict', values)
        setStatus(STATUS.PREDICTING)
        const result = sPredictor.predict(sSampleText)
        logger('handlePredict', result)
        setStatus(STATUS.PREDICTED)
        setPredictResult(result)
    }

    const handleTrain = (values: any): void => {
        if (!sModel || !sTrainSet) {
            return
        }

        logger('handleTrain', values)
        setStatus(STATUS.TRAINING)

        const { epochs, batchSize, validationSplit } = values
        sModel.compile({ loss: 'binaryCrossentropy', optimizer: 'adam', metrics: ['acc'] })

        const callbacks = tfvis.show.fitCallbacks(historyRef?.current, ['loss', 'acc', 'val_loss', 'val_acc'])
        sModel.fit(sTrainSet.xs as tf.Tensor, sTrainSet.ys as tf.Tensor, { epochs, batchSize, validationSplit, callbacks }).then(
            () => {
                setStatus(STATUS.TRAINED)
            },
            loggerError
        )
    }

    const handleModelChange = (value: string): void => {
        setModelName(value)
    }

    const handleLayerChange = (value: number): void => {
        logger('handleLayerChange', value)
        const _layer = sModel?.getLayer(undefined, value)
        setCurLayer(_layer)
    }

    const handleLoadModel = (): void => {
        // TODO
    }

    const handleSaveModel = (): void => {
        // TODO
    }

    const handleSampleTypeChange = (key: string): void => {
        logger('handleSampleTypeChange', key)
        setSampleText(exampleReviews[key])
        formPredict?.setFieldsValue({ sampleText: exampleReviews[key] })
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    /***********************
     * Render
     ***********************/

    return (
        <AIProcessTabs title={'RNN Sentiment'} current={sTabCurrent} onChange={handleTabChange} >
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/sentiment.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={24}>
                        <Card title='Train Set' style={{ margin: '8px' }} size='small'>
                            <div>{sTrainSet && <TfvisDatasetInfoWidget value={sTrainSet}/>}</div>
                        </Card>
                    </Col>
                    <Col span={24}>
                        <Card title='Test Set' style={{ margin: '8px' }} size='small'>
                            <div>{sTestSet && <TfvisDatasetInfoWidget value={sTestSet}/>}</div>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Row>
                    <Col span={24}>
                        <Card title='Sentiment & RNN' style={{ margin: '8px' }} size='small'>
                            Select Model : <Select onChange={handleModelChange} defaultValue={'multihot'}>
                                {MODEL_OPTIONS?.map((v, index) => {
                                    return <Option key={index} value={v}>{v}</Option>
                                })}
                            </Select>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Model Details' style={{ margin: '8px' }} size='small'>
                            <TfvisModelWidget model={sModel}/>
                            <p>status: {sStatus}</p>
                            <p>backend: {sTfBackend}</p>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Layers Details' style={{ margin: '8px' }} size='small'>
                            Select Layer : <Select onChange={handleLayerChange} defaultValue={0}>
                                {sLayersOption?.map((v) => {
                                    return <Option key={v.index} value={v.index}>{v.name}</Option>
                                })}
                            </Select>
                            <TfvisLayerWidget layer={sCurLayer}/>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Predictor Info' size='small'>
                            <div> maxLen : {sPredictor?.maxLen} </div>
                            <div> indexFrom : {sPredictor?.indexFrom} </div>
                            <div> vocabularySize: {sPredictor?.vocabularySize} </div>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                <Row>
                    <Col span={12}>
                        <Card title='Train' style={{ margin: '8px' }} size='small'>
                            <Form {...layout} form={form} onFinish={handleTrain} initialValues={{
                                epochs: sEpochs,
                                batchSize: sBatchSize,
                                validationSplit: sValidationSplit,
                                learningRate: sLearningRate
                            }}>
                                <Form.Item name='epochs' label='Epoches' >
                                    <Input />
                                </Form.Item>
                                <Form.Item name='batchSize' label='Batch Size'>
                                    <Input />
                                </Form.Item>
                                <Form.Item name='validationSplit' label='Validation Split'>
                                    <Input />
                                </Form.Item>
                                <Form.Item {...tailLayout}>
                                    <Button type='primary' htmlType='submit' disabled={ sModelName === 'pretrained-cnn' }
                                        style={{ width: '30%', margin: '0 10%' }}> Train </Button>
                                </Form.Item>
                            </Form>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Save and Load Model Weights' style={{ margin: '8px' }} size='small'>
                            <Button onClick={handleSaveModel} style={{ width: '30%', margin: '0 10%' }}> Save Model </Button>
                            <Button onClick={handleLoadModel} style={{ width: '30%', margin: '0 10%' }}> Load Model </Button>
                            <div>status: {sStatus}</div>
                            <div>backend: {sTfBackend}</div>
                        </Card>
                        <Card title='Training History' style={{ margin: '8px' }} size='small'>
                            <div ref={historyRef}/>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.PREDICT}>
                <Col span={12}>
                    <Card title='Prediction' size='small'>
                        <Form {...layout} form={formPredict} onFinish={handlePredict} initialValues={{
                            sampleType: sSampleType,
                            sampleText: sSampleText
                        }}>
                            <Form.Item name='sampleType' label='Sample Type' >
                                <Select onChange={handleSampleTypeChange}>
                                    {Object.keys(exampleReviews).map((key, index) => {
                                        return <Option key={index} value={key}>{key}</Option>
                                    })}
                                </Select>
                            </Form.Item>
                            <Form.Item name='sampleText' label='Sample Text'>
                                <TextArea rows={6} />
                            </Form.Item>
                            <Form.Item {...tailLayout}>
                                <Button type='primary' htmlType='submit' style={{ width: '50%', margin: '0 10%' }}> Predict </Button>
                            </Form.Item>
                        </Form>
                        {JSON.stringify(sPredictResult)}
                    </Card>
                </Col>
            </TabPane>
        </AIProcessTabs>
    )
}

export default SentimentWidget
