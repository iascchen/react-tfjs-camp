import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Form, Input, message, Row, Select, Slider, Tabs } from 'antd'

import { layout, tailLayout } from '../../constant'
import { ILayerSelectOption, logger, loggerError, STATUS } from '../../utils'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import TfvisDatasetInfoWidget from '../common/tfvis/TfvisDatasetInfoWidget'

import { SentimentPredictor } from './modelSentiment'
import { loadData, loadMetadataTemplate } from './dataSentiment'
import SentimentSampleDataVis from './SentimentSampleDataVis'
import { writeEmbeddingMatrixAndLabels } from './embedding'

// cannot use import
// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

const { Option } = Select
const { TabPane } = Tabs
const { TextArea } = Input

const NUM_TRAIN_ELEMENTS = 25000
const NUM_TEST_ELEMENTS = 20

// Can not run Word Embed Model
const MODEL_OPTIONS = ['pretrained-cnn', 'multihot', 'flatten', 'cnn', 'simpleRNN', 'lstm', 'bidirectionalLSTM']
// const MODEL_OPTIONS = ['pretrained-cnn', 'multihot']
const DEFAULT_MODEL = 'simpleRNN'
const BATCH_SIZES = [128, 256, 512, 1024]
const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03]
const SHOW_SAMPLE = NUM_TEST_ELEMENTS

interface IKeyMap {
    [index: string]: string
}
const exampleReviews: IKeyMap = {
    positive:
        'die hard mario fan and i loved this game br br this game starts slightly boring but trust me it\'s worth it as soon as you start your hooked the levels are fun and exiting they will hook you OOV your mind turns to mush i\'m not kidding this game is also orchestrated and is beautifully done br br to keep this spoiler free i have to keep my mouth shut about details but please try this game it\'ll be worth it br br story 9 9 action 10 1 it\'s that good OOV 10 attention OOV 10 average 10',
    negative:
        'the mother in this movie is reckless with her children to the point of neglect i wish i wasn\'t so angry about her and her actions because i would have otherwise enjoyed the flick what a number she was take my advise and fast forward through everything you see her do until the end also is anyone else getting sick of watching movies that are filmed so dark anymore one can hardly see what is being filmed as an audience we are impossibly involved with the actions on the screen so then why the hell can\'t we have night vision'
}

const saveToDownload = (anchorElm: HTMLAnchorElement, filename: string, body: Buffer, options?: {
    type: string
}): void => {
    const blob = options ? new Blob([body], options) : new Blob([body])
    const blobUrl = window.URL.createObjectURL(blob)
    logger(blobUrl)

    // logger(a)
    anchorElm.href = blobUrl
    anchorElm.download = filename
    anchorElm.click()
    window.URL.revokeObjectURL(blobUrl)
}

const ImdbSentiment = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTabCurrent, setTabCurrent] = useState<number>(4)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    // Data
    const [sNumWords] = useState<number>(10000)
    const [sMaxLen] = useState<number>(100)
    const [sEmbeddingSize] = useState<number>(128)

    const [sMetadata, setMetadata] = useState<any>()

    const [sTrainSet, setTrainSet] = useState<tf.TensorContainerObject>()
    const [sTestSet, setTestSet] = useState<tf.TensorContainerObject>()

    // Model
    const [sModelName, setModelName] = useState(DEFAULT_MODEL)
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sPredictor, setPredictor] = useState<SentimentPredictor>()

    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [sCurLayer, setCurLayer] = useState<tf.layers.Layer>()

    // Train
    const stopRef = useRef(false)
    const [sEpochs, setEpochs] = useState<number>(10)
    const [sBatchSize, setBatchSize] = useState<number>(128)
    const [sLearningRate, setLearningRate] = useState<number>(1e-2)
    const [sValidationSplit, setValidationSplit] = useState<number>(0.2)

    const historyRef = useRef<HTMLDivElement>(null)
    const downloadRef = useRef<HTMLAnchorElement>(null)

    // predict
    const [sSampleType] = useState<string>('positive')
    const [sSampleText, setSampleText] = useState<string>(exampleReviews[sSampleType])
    const [sPredictSet, setPredictSet] = useState<tf.Tensor>()
    const [sPredictResult, setPredictResult] = useState<tf.Tensor>()

    const [formTrain] = Form.useForm()
    const [formPredict] = Form.useForm()

    /***********************
     * useEffect
     ***********************/
    useEffect(() => {
        logger('init model metadata ...')

        loadMetadataTemplate().then(
            (metadata) => {
                setMetadata(metadata)
                setStatus(STATUS.LOADED)
            },
            loggerError
        )
    }, [])

    useEffect(() => {
        logger('init data set ...')

        setStatus(STATUS.WAITING)
        loadData(sNumWords, sMaxLen, (sModelName === 'multihot')).then(
            (result) => {
                const { xTrain, yTrain, xTest, yTest } = result
                const xData = (xTrain as tf.Tensor).slice(0, NUM_TRAIN_ELEMENTS)
                const yData = (yTrain as tf.Tensor).slice(0, NUM_TRAIN_ELEMENTS)
                setTrainSet({ xs: xData, ys: yData })

                const xTData = (xTest as tf.Tensor).slice(0, NUM_TEST_ELEMENTS)
                const yTData = (yTest as tf.Tensor).slice(0, NUM_TEST_ELEMENTS)
                setTestSet({ xs: xTData, ys: yTData })

                setStatus(STATUS.LOADED)
            },
            loggerError
        )
    }, [sModelName])

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.WAITING)

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
                    inputDim: sNumWords, outputDim: sEmbeddingSize, inputLength: sMaxLen
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
            // model.summary()

            _model = model
            setModel(_model)

            const _layerOptions: ILayerSelectOption[] = _model.layers.map((l, index) => {
                return { name: l.name, index }
            })
            setLayersOption(_layerOptions)

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
        // const optimizer = tf.train.adam(sLearningRate)
        const optimizer = 'adam'
        sModel.compile({ loss: 'binaryCrossentropy', optimizer, metrics: ['acc'] })
    }, [sModel, sLearningRate])

    /***********************
     * Functions
     ***********************/

    const predictModel = (_model: tf.LayersModel, _xs: tf.TensorContainer): void => {
        if (!_model || !_xs) {
            return
        }
        const [preds] = tf.tidy(() => {
            const preds = _model.predict(_xs as tf.Tensor) as tf.Tensor
            return [preds]
        })
        setPredictSet(preds)
    }

    const myCallback = {
        onBatchBegin: async (batch: number) => {
            logger('onBatchBegin', sModelName)
            if (!sModel) {
                return
            }
            if (sModel && stopRef.current) {
                logger('Checked stop', stopRef.current)
                setStatus(STATUS.STOPPED)
                sModel.stopTraining = stopRef.current
            }
            await tf.nextFrame()
        },
        onBatchEnd: async (batch: number) => {
            if (!sModel || !sTestSet) {
                return
            }
            predictModel(sModel, sTestSet?.xs)
            await tf.nextFrame()
        }
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    const handleModelChange = (value: string): void => {
        setModelName(value)
    }

    const handleLayerChange = (value: number): void => {
        logger('handleLayerChange', value)
        const _layer = sModel?.getLayer(undefined, value)
        setCurLayer(_layer)
    }

    const handleTrainParamsChange = (): void => {
        const values = formTrain.getFieldsValue()
        // logger('handleTrainParamsChange', values)
        const { epochs, batchSize, learningRate, validationSplit } = values
        setEpochs(epochs)
        setBatchSize(batchSize)
        setLearningRate(learningRate)
        setValidationSplit(validationSplit)
    }

    const handleTrain = (): void => {
        if (!sModel || !sTrainSet || !sMetadata) {
            return
        }

        setStatus(STATUS.WAITING)
        stopRef.current = false

        const callbacks = [
            tfvis.show.fitCallbacks(historyRef.current, ['loss', 'acc', 'val_loss', 'val_acc']),
            myCallback
        ]

        logger('Train Begin')
        sModel.fit(sTrainSet.xs as tf.Tensor, sTrainSet.ys as tf.Tensor, {
            epochs: sEpochs,
            batchSize: sBatchSize,
            validationSplit: sValidationSplit,
            callbacks
        }).then(
            () => {
                setStatus(STATUS.TRAINED)
                handleSaveModelWeight()
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
        const fileName = `imdb_${sModelName}`
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
        if (!sModel || !sMetadata) {
            return
        }

        // Save Model
        const fileName = `imdb_${sModelName}`
        const downloadUrl = `downloads://${fileName}`
        sModel.save(downloadUrl).then((saveResults) => {
            logger(saveResults)
        }, loggerError)

        // Save metadata.
        sMetadata.epochs = sEpochs
        sMetadata.batch_size = sBatchSize
        sMetadata.model_type = sModelName
        sMetadata.embedding_size = sEmbeddingSize
        sMetadata.max_len = sMaxLen
        sMetadata.vocabulary_size = sNumWords

        const a = downloadRef.current
        if (a) {
            saveToDownload(a, `${fileName}.metadata.json`, Buffer.from(JSON.stringify(sMetadata)),
                { type: 'application/json' })
        }

        if (sModelName !== 'multihot') {
            // writeEmbeddingMatrixAndLabels
            writeEmbeddingMatrixAndLabels(sModel, fileName, sMetadata.word_index, sMetadata.index_from).then(
                (result) => {
                    const { vectorsFilePath, vectorsStr, labelsFilePath, labelsStr } = result
                    if (a && vectorsStr) {
                        saveToDownload(a, vectorsFilePath, Buffer.from(vectorsStr, 'utf-8'))
                    }
                    if (a && labelsStr) {
                        saveToDownload(a, labelsFilePath, Buffer.from(labelsStr, 'utf-8'))
                    }
                },
                loggerError
            )
        }
    }

    const handleSampleTypeChange = (key: string): void => {
        logger('handleSampleTypeChange', key)
        setSampleText(exampleReviews[key])
        formPredict?.setFieldsValue({ sampleText: exampleReviews[key] })
    }

    const handlePredict = (values: any): void => {
        if (!sPredictor) {
            return
        }
        // logger('handlePredict', values)
        setStatus(STATUS.WAITING)
        const result = sPredictor.predict(sSampleText)
        logger('handlePredict', result)
        setStatus(STATUS.PREDICTED)
        setPredictResult(result)
    }

    /***********************
     * Render
     ***********************/

    const modelAdjustCard = (): JSX.Element => {
        return (
            <Card title='Adjust Model' style={{ margin: '8px' }} size='small'>
                <Form {...layout} initialValues={{
                    modelName: DEFAULT_MODEL
                }}>
                    <Form.Item name='modelName' label='Select Model'>
                        <Select onChange={handleModelChange}>
                            {MODEL_OPTIONS.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
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
                        batchSize: sBatchSize,
                        validationSplit: sValidationSplit,
                        learningRate: sLearningRate
                    }}>
                    <Form.Item name='epochs' label='Epochs'>
                        <Slider min={2} max={10} marks={{ 2: 2, 6: 6, 10: 10 }}/>
                    </Form.Item>
                    <Form.Item name='batchSize' label='Batch Size'>
                        <Select>
                            {BATCH_SIZES.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item name='validationSplit' label='Validation Split'>
                        <Slider min={0.1} max={0.2} step={0.05} marks={{ 0.1: 0.1, 0.15: 0.15, 0.2: 0.2 }}/>
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
                            disabled={ sModelName === 'pretrained-cnn' }> Train </Button>
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
        <AIProcessTabs title={'IMDB Sentiment'} current={sTabCurrent} onChange={handleTabChange} >
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/sentiment.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={12}>
                        <Card title='Train Set' style={{ margin: '8px' }} size='small'>
                            <div>{sTrainSet && <TfvisDatasetInfoWidget value={sTrainSet}/>}</div>
                            <SentimentSampleDataVis sampleCount={SHOW_SAMPLE} pageSize={5}
                                xDataset={sTrainSet?.xs as tf.Tensor} yDataset={sTrainSet?.ys as tf.Tensor}/>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Test Set' style={{ margin: '8px' }} size='small'>
                            <div>{sTestSet && <TfvisDatasetInfoWidget value={sTestSet}/>}</div>
                            <SentimentSampleDataVis sampleCount={SHOW_SAMPLE} pageSize={5}
                                xDataset={sTestSet?.xs as tf.Tensor} yDataset={sTestSet?.ys as tf.Tensor}/>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Row>
                    <Col span={8}>
                        {modelAdjustCard()}
                        <Card title={`Show Layers of ${sModelName}`} style={{ margin: '8px' }} size='small'>
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
                            </Form>
                        </Card>
                    </Col>
                    <Col span={16}>
                        <Card title='Model Details' style={{ margin: '8px' }} size='small'>
                            <TfvisModelWidget model={sModel}/>
                            <TfvisLayerWidget layer={sCurLayer}/>
                            <p>status: {sStatus}</p>
                            <p>backend: {sTfBackend}</p>
                        </Card>
                        <Card title='Predictor Info' style={{ margin: '8px' }} size='small'>
                            <div> maxLen : {sPredictor?.maxLen} </div>
                            <div> indexFrom : {sPredictor?.indexFrom} </div>
                            <div> vocabularySize: {sPredictor?.vocabularySize} </div>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                <Row>
                    <Col span={8}>
                        {trainAdjustCard()}
                        {modelAdjustCard()}
                        <a ref={downloadRef}/>
                    </Col>
                    <Col span={10}>
                        <Card title='Evaluate' style={{ margin: '8px' }} size='small'>
                            <SentimentSampleDataVis pageSize={5}
                                xDataset={sTestSet?.xs as tf.Tensor} yDataset={sTestSet?.ys as tf.Tensor}
                                pDataset={sPredictSet} />
                        </Card>
                    </Col>
                    <Col span={6}>
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

export default ImdbSentiment
