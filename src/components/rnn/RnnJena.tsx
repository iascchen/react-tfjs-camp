import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Form, InputNumber, message, Row, Select, Slider, Switch, Tabs } from 'antd'

import { layout, tailLayout } from '../../constant'
import { ILayerSelectOption, logger, loggerError, STATUS } from '../../utils'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import MarkdownWidget from '../common/MarkdownWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'

import { buildGRUModel, buildLinearRegressionModel, buildMLPModel, buildSimpleRNNModel } from './modelJena'
import { JenaWeatherData, TRAIN_MAX_ROW, TRAIN_MIN_ROW, VAL_MAX_ROW, VAL_MIN_ROW } from './dataJena'

// cannot use import
const tfvis = require('@tensorflow/tfjs-vis')

const { TabPane } = Tabs
const { Option } = Select

const MODEL_OPTIONS = ['linear-regression', 'mlp', 'mlp-l2', 'mlp-dropout', 'simpleRnn', 'gru']
const NUM_FEATURES = 14

export const TIME_SPAN_RANGE_MAP: IKeyMap = {
    hour: 6,
    day: 6 * 24,
    week: 6 * 24 * 7,
    tenDays: 6 * 24 * 10,
    month: 6 * 24 * 30,
    year: 6 * 24 * 365
    // full: undefined
}

export const TIME_SPAN_STRIDE_MAP: IKeyMap = {
    day: 1,
    week: 1,
    tenDays: 6,
    month: 6,
    year: 6 * 6
    // full: 6 * 24
}

export const TIME_SPAN_MAX: IKeyMap = {
    day: 6 * 365,
    week: Math.floor(6 * 365 / 7),
    tenDays: Math.floor(6 * 365 / 10),
    month: 6 * 12,
    year: 6
}

const BATCH_SIZES = [64, 128, 256, 512]
const STEP = 6 // 1-hour steps.

const MODEL_URL_NAME = 'mobilenet-simple-obj-detector'

const mdInfo = '**注意** \n' +
    '\n' +
    '* 如果您要在本地环境运行这个例子，最好预先下载数据文件。并将数据文件放在此项目的 `./public/data` 目录下。\n' +
    '\n' +
    '    [https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv](https://storage.googleapis.com/learnjs-data/jena_climate/jena_climate_2009_2016.csv)\n' +
    '\n' +
    '* 所需的数据大约有 41.2MB。\n' +
    '* 刷新页面，会丢失已经加载的数据。'

interface IKeyMap {
    [index: string]: number
}

export const getBaselineMeanAbsoluteError = async (jenaWeatherData: JenaWeatherData, normalize: boolean,
    includeDateTime: boolean, lookBack: number, step: number, delay: number): Promise<number> => {
    const batchSize = 128
    const dataset = tf.data.generator(
        () => jenaWeatherData.getNextBatchFunction(false, lookBack, delay, batchSize, step,
            VAL_MIN_ROW, VAL_MAX_ROW, normalize, includeDateTime))

    const batchMeanAbsoluteErrors: tf.Tensor[] = []
    const batchSizes: number[] = []

    await dataset.forEachAsync(dataItem => {
        const features = dataItem.xs as tf.Tensor
        const targets = dataItem.ys as tf.Tensor
        const timeSteps = features.shape[1] as number
        batchSizes.push(features.shape[0])
        batchMeanAbsoluteErrors.push(tf.tidy(() => tf.losses.absoluteDifference(targets,
            features.gather([timeSteps - 1], 1).gather([1], 2).squeeze([2]))))
    })

    const meanAbsoluteError = tf.tidy(() => {
        const batchSizesTensor = tf.tensor1d(batchSizes)
        const batchMeanAbsoluteErrorsTensor = tf.stack(batchMeanAbsoluteErrors)
        return batchMeanAbsoluteErrorsTensor.mul(batchSizesTensor)
            .sum()
            .div(batchSizesTensor.sum())
    })
    tf.dispose(batchMeanAbsoluteErrors)
    return meanAbsoluteError.dataSync()[0]
}

export const trainModel =
    async (model: tf.LayersModel, jenaWeatherData: JenaWeatherData, normalize: boolean, includeDateTime: boolean,
        lookBack: number, step: number, delay: number, batchSize: number, epochs: number,
        callbacks: tf.Callback[]): Promise<void> => {
        const trainShuffle = true
        const trainDataset = tf.data.generator(
            () => jenaWeatherData.getNextBatchFunction(
                trainShuffle, lookBack, delay, batchSize, step, TRAIN_MIN_ROW,
                TRAIN_MAX_ROW, normalize, includeDateTime)).prefetch(8)

        const evalShuffle = false
        const valDataset = tf.data.generator(
            () => jenaWeatherData.getNextBatchFunction(
                evalShuffle, lookBack, delay, batchSize, step, VAL_MIN_ROW,
                VAL_MAX_ROW, normalize, includeDateTime))

        await model.fitDataset(trainDataset, {
            batchesPerEpoch: 500,
            epochs,
            callbacks,
            validationData: valDataset
        })
    }

const RnnJena = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTabCurrent, setTabCurrent] = useState<number>(2)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    // Data
    const [sDataLoaded, setDataLoaded] = useState(false)
    const [sDataHandler, setDataHandler] = useState<JenaWeatherData>()
    const [sLoadSpendSec, setLoadSpendSec] = useState(0)
    const [sSeriesOptions, setSeriesOptions] = useState<string[]>([])
    const [sTimeSpan, setTimeSpan] = useState<string>('tenDays')

    // Model
    const [sModelName, setModelName] = useState('simpleRnn')
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [sCurLayer, setCurLayer] = useState<tf.layers.Layer>()

    // Train
    const stopRef = useRef(false)
    const [sEpochs, setEpochs] = useState<number>(3)
    const [sBatchSize, setBatchSize] = useState<number>(256)
    const [sNormalize, setNormalize] = useState(false)
    const [sIncludeTime, setIncludeTime] = useState(false)
    const [sLookBack, setLookBack] = useState(10 * 24 * STEP)
    const [sDelay, setDelay] = useState(1 * 24 * STEP)
    const [sBaseLine, setBaseLine] = useState<number>()

    const elementRef = useRef<HTMLDivElement>(null)
    const dataChartRef = useRef<HTMLDivElement>(null)
    const chartRef = useRef<HTMLDivElement>(null)

    const [formDataShow] = Form.useForm()
    const [formTrain] = Form.useForm()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init data handler...')
        const dataHandler = new JenaWeatherData()

        if (dataHandler.csvLines.length === 0) {
            dataHandler.loadCsv().then(
                (result) => {
                    dataHandler.csvLines = result
                    dataHandler.loadDataColumnNames()
                    logger('Data fetched', dataHandler.csvLines.length)
                    logger('dataColumnNames', dataHandler.dataColumnNames)
                    setSeriesOptions(dataHandler.dataColumnNames)
                }, (e) => {
                    logger(e)
                    // eslint-disable-next-line @typescript-eslint/no-floating-promises
                    message.error(e.message)
                })
        }
        setDataHandler(dataHandler)

        return () => {
            logger('Data Dispose')
        }
    }, [])

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        const lookBack = 10 * 24 * 6 // Look back 10 days.
        const step = 6 // 1-hour steps.
        const numTimeSteps = Math.floor(lookBack / step)
        const numFeatures = NUM_FEATURES
        const inputShape: tf.Shape = [numTimeSteps, numFeatures]

        let _model: tf.LayersModel
        switch (sModelName) {
            case 'linear-regression' :
                _model = buildLinearRegressionModel(inputShape)
                break
            case 'mlp' :
                _model = buildMLPModel(inputShape)
                break
            case 'mlp-l2' :
                _model = buildMLPModel(inputShape, { kernelRegularizer: tf.regularizers.l2() })
                break
            case 'mlp-dropout' :
                _model = buildMLPModel(inputShape, { dropoutRate: 0.25 })
                break
            case 'gru' :
                _model = buildGRUModel(inputShape)
                break
            case 'simpleRnn' :
            default:
                _model = buildSimpleRNNModel(inputShape)
                break
        }

        _model.compile({ loss: 'meanAbsoluteError', optimizer: 'rmsprop' })
        // _model.summary()
        setModel(_model)

        const _layerOptions: ILayerSelectOption[] = _model?.layers.map((l, index) => {
            return { name: l.name, index }
        })
        setLayersOption(_layerOptions)

        return () => {
            logger('Model Dispose')
            _model?.dispose()
        }
    }, [sModelName])

    /***********************
     * Functions
     ***********************/

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

    const train = async (): Promise<void> => {
        if (!sModel || !sDataHandler) {
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            message.warn('Data and model are not ready')
            return
        }

        setStatus(STATUS.WAITING)
        logger('train...')

        console.log('Starting model training...')
        const callbacks = [
            tfvis.show.fitCallbacks(elementRef.current, ['loss', 'val_loss'], { callbacks: ['onBatchEnd', 'onEpochEnd'] }),
            myCallback
        ]

        await trainModel(sModel, sDataHandler, sNormalize, sIncludeTime, sLookBack, STEP, sDelay, sBatchSize,
            sEpochs, callbacks)
        setStatus(STATUS.TRAINED)
    }

    const makeTimeSeriesChart = async (series1: string, series2: string, timeSpan: string, currBeginIndex: number, normalize: boolean,
        chartConatiner: HTMLDivElement): Promise<void> => {
        if (!sDataHandler || !chartConatiner) {
            return
        }

        const values = []
        const series = []
        const includeTime = true
        if (series1 !== 'None') {
            values.push(sDataHandler.getColumnData(
                series1, includeTime, normalize, currBeginIndex * TIME_SPAN_RANGE_MAP[timeSpan],
                TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]))
            series.push(normalize ? `${series1} (normalized)` : series1)
        }
        if (series2 !== 'None') {
            values.push(sDataHandler.getColumnData(
                series2, includeTime, normalize, currBeginIndex * TIME_SPAN_RANGE_MAP[timeSpan],
                TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]))
            series.push(normalize ? `${series2} (normalized)` : series2)
        }
        // NOTE(cais): On a Linux workstation running latest Chrome, the length
        // limit seems to be around 120k.
        await tfvis.render.linechart(chartConatiner, { values, series: series }, {
            width: chartConatiner.offsetWidth * 0.95,
            height: chartConatiner.offsetWidth * 0.3,
            xLabel: 'Time',
            yLabel: series.length === 1 ? series[0] : ''
        })
    }

    const makeTimeSeriesScatterPlot = async (series1: string, series2: string, timeSpan: string, currBeginIndex: number, normalize: boolean): Promise<void> => {
        if (!sDataHandler || !dataChartRef.current) {
            return
        }

        const includeTime = false
        const xs = sDataHandler.getColumnData(
            series1, includeTime, normalize, currBeginIndex * TIME_SPAN_RANGE_MAP[timeSpan],
            TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan])
        const ys = sDataHandler.getColumnData(
            series2, includeTime, normalize, currBeginIndex * TIME_SPAN_RANGE_MAP[timeSpan],
            TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan])
        const values = [xs.map((x, i) => {
            return { x, y: ys[i] }
        })]
        let seriesLabel1 = series1
        let seriesLabel2 = series2
        if (normalize) {
            seriesLabel1 += ' (normalized)'
            seriesLabel2 += ' (normalized)'
        }
        const series = [`${seriesLabel1} - ${seriesLabel2}`]

        await tfvis.render.scatterplot(dataChartRef.current, { values, series }, {
            width: dataChartRef.current.offsetWidth * 0.7,
            height: dataChartRef.current.offsetWidth * 0.5,
            xLabel: seriesLabel1,
            yLabel: seriesLabel2
        })
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    const handleDataLoad = (): void => {
        setStatus(STATUS.WAITING)
        const beginMs = performance.now()
        sDataHandler?.load().then(() => {
            setLoadSpendSec((performance.now() - beginMs) / 1000)
            setStatus(STATUS.LOADED)
            setDataLoaded(true)
        }, (e) => {
            logger(e)
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            message.error(e.message)
        })
    }

    const handleCalc = (): void => {
        if (!sDataHandler || !sDataLoaded) {
            return
        }
        setStatus(STATUS.WAITING)
        const beginMs = performance.now()
        getBaselineMeanAbsoluteError(sDataHandler, sNormalize, sIncludeTime, sLookBack, STEP, sDelay).then(
            (value) => {
                logger('getBaselineMeanAbsoluteError', value)
                setLoadSpendSec((performance.now() - beginMs) / 1000)
                setBaseLine(value)
                setStatus(STATUS.READY)
            },
            (e) => {
                logger(e.msg)
            }
        )
    }

    const handleShowPlots = (): void => {
        if (!sDataHandler || !sDataLoaded || !dataChartRef.current || !chartRef.current) {
            return
        }

        const values = formDataShow.getFieldsValue()
        logger('handleDataShowParamsChange', values)
        const { normalize, timeSpan, series1, series2, currBeginIndex } = values
        setTimeSpan(timeSpan)

        logger('Draw Samples')

        makeTimeSeriesChart(series1, series2, timeSpan, currBeginIndex, normalize, chartRef.current).then(
            () => {
                logger('Draw TimeSeriesChart')
            },
            (e) => {
                logger(e.msg)
            }
        )
        makeTimeSeriesScatterPlot(series1, series2, timeSpan, currBeginIndex, normalize).then(
            () => {
                logger('Draw TimeSeriesScatterPlot')
            },
            (e) => {
                logger(e.msg)
            }
        )
    }

    const handleBeginIndexInc = (): void => {
        let index = +formDataShow.getFieldValue('currBeginIndex')
        index = index < TIME_SPAN_MAX[sTimeSpan] ? index + 1 : TIME_SPAN_MAX[sTimeSpan]
        formDataShow.setFieldsValue({ currBeginIndex: index })
        handleShowPlots()
    }

    const handleBeginIndexMinus = (): void => {
        let index = +formDataShow.getFieldValue('currBeginIndex')
        index = index > 0 ? index - 1 : 0
        formDataShow.setFieldsValue({ currBeginIndex: index })
        handleShowPlots()
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
        const { epochs, batchSize, lookBackDays, delayDays, normalize, includeTime } = values
        setEpochs(epochs)
        setBatchSize(batchSize)

        setLookBack(lookBackDays * 24 * 6)
        setDelay(delayDays * 24 * 6)
        setNormalize(normalize)
        setIncludeTime(includeTime)
    }

    const handleTrain = (): void => {
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        train().then()
    }

    const handleTrainStop = (): void => {
        logger('handleTrainStop')
        stopRef.current = true
    }

    const handleLoadModelWeight = (): void => {
        setStatus(STATUS.WAITING)
        tf.loadLayersModel(`/model/${MODEL_URL_NAME}.json`).then(
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

        const downloadUrl = `downloads://${MODEL_URL_NAME}`
        sModel.save(downloadUrl).then((saveResults) => {
            logger(saveResults)
        }, loggerError)
        logger()
    }

    /***********************
     * Render
     ***********************/

    const statusInfo = (): string => {
        if (sStatus === STATUS.WAITING) {
            return 'Please waiting...'
        } else if (sStatus === STATUS.LOADED) {
            return `${sDataHandler?.getDataLength()} items with ${sLoadSpendSec.toFixed(3)} s.`
        } else if (sStatus === STATUS.READY) {
            return `Mean absolute error of baseline is : ${sBaseLine}`
        } else {
            return 'No data load'
        }
    }

    const dataDisplayCard = (): JSX.Element => {
        return (
            <Card title='Show Data' style={{ margin: '8px' }} size='small'>
                <Form {...layout} form={formDataShow} onFinish={handleShowPlots}
                    initialValues={{
                        normalize: false,
                        timeSpan: 'tenDays',
                        currBeginIndex: 0,
                        series1: 'T (degC)',
                        series2: 'p (mbar)'
                    }}>
                    <Form.Item name='normalize' label='Normalize Data'>
                        <Switch/>
                    </Form.Item>
                    <Form.Item name='timeSpan' label='Time Span'>
                        <Select >
                            {Object.keys(TIME_SPAN_RANGE_MAP).map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item name='series1' label='Data Serial 1'>
                        <Select>
                            {sSeriesOptions.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item name='series2' label='Data Serial 2'>
                        <Select >
                            {sSeriesOptions.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item name='currBeginIndex' label='Begin Index'>
                        <InputNumber />
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <Button style={{ width: '30%', margin: '0 35%' }} htmlType={'submit'}
                            disabled={!sDataLoaded} > Show Plots </Button>
                    </Form.Item>
                </Form>
                <Row>
                    <Col span={1}></Col>
                    <Col span={22}>
                        <div ref={dataChartRef}/>
                        <div ref={chartRef} />
                    </Col>
                    <Col span={1}></Col>
                </Row>
                <Row>
                    <Col span={1}>
                        <Button disabled={!sDataLoaded} onClick={handleBeginIndexMinus}> &lt; </Button></Col>
                    <Col span={22}>
                    </Col>
                    <Col span={1}><Button disabled={!sDataLoaded} onClick={handleBeginIndexInc}> &gt; </Button></Col>
                </Row>
            </Card>
        )
    }

    const modelAdjustCard = (): JSX.Element => {
        return (
            <Card title='Adjust Model' style={{ margin: '8px' }} size='small'>
                <Form {...layout} initialValues={{
                    modelName: 'simpleRnn'
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
                        epochs: 3,
                        batchSize: 256,
                        lookBackDays: 10,
                        delayDays: 1,
                        normalize: true,
                        includeTime: false
                    }}>
                    <Form.Item name='epochs' label='Epochs'>
                        <Slider min={1} max={10} marks={{ 1: 1, 5: 5, 9: 9 }} />
                    </Form.Item>
                    <Form.Item name='batchSize' label='Batch Size'>
                        <Select>
                            {BATCH_SIZES.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item name='lookBackDays' label='Look Back Days'>
                        <Slider min={8} max={12} marks={{ 8: 8, 10: 10, 12: 12 }} />
                    </Form.Item>
                    <Form.Item name='delayDays' label='Delay Days'>
                        <Slider min={1} max={3} marks={{ 1: 1, 2: 2, 3: 3 }} />
                    </Form.Item>
                    <Form.Item name='normalize' label='Normalize Data'>
                        <Switch/>
                    </Form.Item>
                    <Form.Item name='includeTime' label='Include Time'>
                        <Switch/>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <Button type='primary' htmlType={'submit'} style={{ width: '30%', margin: '0 10%' }}
                            disabled={!sDataLoaded}> Train </Button>
                        <Button onClick={handleTrainStop} style={{ width: '30%', margin: '0 10%' }}> Stop </Button>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <Button onClick={handleSaveModelWeight} style={{ width: '30%', margin: '0 10%' }}> Save
                            Weights </Button>
                        <Button onClick={handleLoadModelWeight} style={{ width: '30%', margin: '0 10%' }}> Load
                            Weights </Button>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <Button style={{ width: '30%', margin: '0 10%' }}
                            disabled={!sDataLoaded} onClick={handleCalc}> Calc Baseline </Button>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <div>Status: {sStatus}</div>
                        <div>{statusInfo()}</div>
                        <div>Backend: {sTfBackend}</div>
                    </Form.Item>
                </Form>
            </Card>
        )
    }

    return (
        <AIProcessTabs title={'RNN'} current={sTabCurrent} onChange={handleTabChange}
            invisiblePanes={[AIProcessTabPanes.PREDICT]}>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/rnnJena.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={8}>
                        <Card title={'Data'} style={{ margin: 8 }}>
                            <MarkdownWidget source={mdInfo}/>
                            <Button style={{ width: '30%', margin: '0 35%' }} type='primary'
                                disabled={sStatus === STATUS.WAITING} onClick={handleDataLoad}> Load </Button>
                            <p>Status: {sStatus}</p>
                            <p>{statusInfo()}</p>
                        </Card>
                    </Col>
                    <Col span={16}>
                        {dataDisplayCard()}
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
                                    <Select onChange={handleLayerChange} >
                                        {sLayersOption?.map((v) => {
                                            return <Option key={v.index} value={v.index}>{v.name}</Option>
                                        })}
                                    </Select>
                                </Form.Item>
                            </Form>
                        </Card>
                    </Col>
                    <Col span={16}>
                        <Card title='Model Info' style={{ margin: '8px' }} size='small'>
                            <TfvisModelWidget model={sModel}/>
                        </Card>
                        <Card title='Layer Info' style={{ margin: '8px' }} size='small'>
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
                    </Col>
                    <Col span={8}>
                        Hello
                    </Col>
                    <Col span={8}>
                        <Card title='Training History' style={{ margin: '8px' }} size='small'>
                            <div ref={elementRef} />
                        </Card>
                    </Col>
                </Row>
            </TabPane>
        </AIProcessTabs>
    )
}

export default RnnJena
