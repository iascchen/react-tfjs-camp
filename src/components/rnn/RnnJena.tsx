import React, { useEffect, useReducer, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, message, Row, Select, Tabs } from 'antd'

import { ILayerSelectOption, logger, STATUS } from '../../utils'
import ImageUploadWidget from '../common/tensor/ImageUploadWidget'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import MarkdownWidget from '../common/MarkdownWidget'
import {
    buildGRUModel,
    buildLinearRegressionModel,
    buildMLPModel,
    buildSimpleRNNModel,
    trainModel
} from './jena/modelJena'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'

import { JenaWeatherData } from './jena/dataJena'

// cannot use import
// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

const { TabPane } = Tabs
const { Option } = Select

const MODEL_OPTIONS = ['linear-regression', 'mlp', 'mlp-l2', 'mlp-dropout', 'simpleRnn', 'gru']
const NUM_FEATURES = 14

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
    year: 6 * 6,
    full: 6 * 24
}

const RnnJena = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTabCurrent, setTabCurrent] = useState<number>(1)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    // const [sNumFeatures, setNumFeatures] = useState(10)
    const [sDataHandler, setDataHandler] = useState<JenaWeatherData>()
    const [sSampleData, setSampleData] = useState<tf.TensorContainerObject>()
    const [sLoadSpendSec, setLoadSpendSec] = useState<number>(0)

    const [sModelName, setModelName] = useState('simpleRnn')
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [curLayer, setCurLayer] = useState<tf.layers.Layer>()

    const [sCurrBeginIndex, setCurrBeginIndex] = useState<number>(0)
    const [ignore, forceUpdate] = useReducer((x: number) => x + 1, 0)

    const [sEpochs, setEpochs] = useState<number>(5)
    const [sPredictResult, setPredictResult] = useState<tf.Tensor>()

    const elementRef = useRef<HTMLDivElement>(null)
    const dataChartRef = useRef<HTMLDivElement>(null)
    const chartRef = useRef<HTMLDivElement>(null)

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init data ...')
        const dataHandler = new JenaWeatherData()

        if (dataHandler.csvLines.length === 0) {
            dataHandler.loadCsv().then(
                (result) => {
                    dataHandler.csvLines = result
                    dataHandler.loadDataColumnNames()
                    logger('Data fetched', dataHandler.csvLines.length)
                    logger('dataColumnNames', dataHandler.dataColumnNames)
                }, (e) => {
                    logger(e)
                    // eslint-disable-next-line @typescript-eslint/no-floating-promises
                    message.error(e.message)
                })
        }
        setDataHandler(dataHandler)

        return () => {
            logger('Data Dispose')
            dataHandler.dispose()
        }
    }, [])

    useEffect(() => {
        logger('Draw 1')
        if (!sDataHandler || !dataChartRef.current || !chartRef.current) {
            return
        }
        logger('Draw 2')

        const series1 = 'T (degC)'
        const series2 = 'p (mbar)'
        makeTimeSeriesChart(series1, series2, 'week', true, chartRef.current).then(
            () => {
                logger('Draw 3')
            },
            (e) => {
                logger(e.msg)
            }
        )
        makeTimeSeriesScatterPlot(series1, series2, 'week', true).then(
            () => {
                logger('Draw 4')
            },
            (e) => {
                logger(e.msg)
            }
        )
    }, [ignore])

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

    const train = async (): Promise<void> => {
        if (!sModel || !sDataHandler) {
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            message.warn('Data and model are not ready')
            return
        }

        setStatus(STATUS.TRAINING)
        logger('train...')

        const lookBack = 10 * 24 * 6 // Look back 10 days.
        const step = 6 // 1-hour steps.
        const delay = 24 * 6 // Predict the weather 1 day later.
        const batchSize = 128
        const normalize = true
        const includeDateTime = false

        console.log('Starting model training...')
        const epochs = sEpochs
        await trainModel(sModel, sDataHandler, normalize, includeDateTime,
            lookBack, step, delay, batchSize, epochs,
            tfvis.show.fitCallbacks(elementRef.current, ['loss', 'val_loss'], {
                callbacks: ['onBatchEnd', 'onEpochEnd']
            }))
        setStatus(STATUS.TRAINED)
    }

    const handleTrain = (): void => {
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        train().then()
    }

    const makeTimeSeriesChart = async (series1: string, series2: string, timeSpan: string, normalize: boolean, chartConatiner: HTMLDivElement): Promise<void> => {
        if (!sDataHandler || !chartConatiner) {
            return
        }

        const values = []
        const series = []
        const includeTime = true
        if (series1 !== 'None') {
            values.push(sDataHandler.getColumnData(
                series1, includeTime, normalize, sCurrBeginIndex,
                TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan]))
            series.push(normalize ? `${series1} (normalized)` : series1)
        }
        if (series2 !== 'None') {
            values.push(sDataHandler.getColumnData(
                series2, includeTime, normalize, sCurrBeginIndex,
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

    const makeTimeSeriesScatterPlot = async (series1: string, series2: string, timeSpan: string, normalize: boolean): Promise<void> => {
        if (!sDataHandler || !dataChartRef.current) {
            return
        }

        const includeTime = false
        const xs = sDataHandler.getColumnData(
            series1, includeTime, normalize, sCurrBeginIndex,
            TIME_SPAN_RANGE_MAP[timeSpan], TIME_SPAN_STRIDE_MAP[timeSpan])
        const ys = sDataHandler.getColumnData(
            series2, includeTime, normalize, sCurrBeginIndex,
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

    const handlePredict = async (imgTensor: tf.Tensor): Promise<void> => {
        if (!imgTensor) {

        }
    }

    // const handleLabeledImagesSubmit = (value: ILabeledImageFileJson): void => {
    //     logger('handleLabeledImagesSubmit', value)
    //
    //     const labeledImageSetList = value.labeledImageSetList
    //     setLabeledImgs(labeledImageSetList)
    // }
    //
    // const handleLoadJson = (values: ILabeledImageSet[]): void => {
    //     sLabeledImgs && arrayDispose(sLabeledImgs)
    //     setLabeledImgs(values)
    // }

    const handleDataLoad = (): void => {
        setStatus(STATUS.LOADING)

        const beginMs = performance.now()
        sDataHandler?.load().then(() => {
            const _sample = sDataHandler.getSampleData()
            setSampleData(_sample)

            setLoadSpendSec((performance.now() - beginMs) / 1000)
            setStatus(STATUS.LOADED)

            forceUpdate()
        }, (e) => {
            logger(e)
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            message.error(e.message)
        })
    }

    const handleModelChange = (value: string): void => {
        setModelName(value)
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

    const modelTitle = (
        <>
            Model
            <Select onChange={handleModelChange} defaultValue={'simpleRnn'} style={{ marginLeft: 16 }}>
                {MODEL_OPTIONS.map((v) => {
                    return <Option key={v} value={v}>{v}</Option>
                })}
            </Select>
        </>
    )

    const layerTitle = (
        <>
            {sModelName} Layers
            <Select onChange={handleLayerChange} defaultValue={0} style={{ marginLeft: 16 }}>
                {sLayersOption?.map((v) => {
                    return <Option key={v.index} value={v.index}>{v.name}</Option>
                })}
            </Select>
        </>
    )

    const statusInfo = (): string => {
        if (sStatus === STATUS.LOADING) {
            return 'Please waiting...'
        } else if (sStatus === STATUS.LOADED) {
            return `${sDataHandler?.getDataLength()} items with ${sLoadSpendSec.toFixed(3)} s`
        } else {
            return 'No data load'
        }
    }

    return (
        <AIProcessTabs title={'RNN'} current={sTabCurrent} onChange={handleTabChange} >
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/rnnJena.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={12}>
                        <Card title={'Data'} style={{ margin: 8 }}>
                            <MarkdownWidget source={mdInfo}/>
                            <Button style={{ width: '30%', margin: '0 auto' }} type='primary'
                                disabled={sStatus === STATUS.LOADING} onClick={handleDataLoad}> Load </Button>
                            {/* {JSON.stringify(sSampleData)} */}
                            {/* {sSampleData && (<SampleDataVis xFloatFixed={4} xDataset={sSampleData.data as tf.Tensor} */}
                            {/*    yDataset={sSampleData.normalizedTimeOfDay as tf.Tensor} ></SampleDataVis>)} */}

                            <p>Status: {sStatus}</p>
                            <p>{statusInfo()}</p>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title={'Loaded Data'} style={{ margin: 8 }}>
                            <div ref={dataChartRef}/>
                            <div ref={chartRef} />
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Row>
                    <Col span={12} >
                        <Card title={modelTitle} style={{ margin: 8 }}>
                            { sModel && <TfvisModelWidget model={sModel}/>}
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title={layerTitle} style={{ margin: 8 }}>
                            <TfvisLayerWidget layer={curLayer}/>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                <Row>
                    <Col span={12}>
                        <Card title='Jena Weather' style={{ margin: '8px' }} size='small'>
                            <div>
                                <Button onClick={handleTrain} type='primary' style={{ width: '30%', margin: '0 10%' }}> Train </Button>
                                <div>status: {sStatus}</div>
                            </div>
                            <p>backend: {sTfBackend}</p>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Training History' style={{ margin: '8px' }} size='small'>
                            <div ref={elementRef} />
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.PREDICT}>
                <Card title='Prediction' style={{ margin: '8px' }} size='small'>
                    {sModel && <ImageUploadWidget model={sModel} onSubmit={handlePredict} prediction={sPredictResult}/>}
                </Card>
            </TabPane>
        </AIProcessTabs>
    )
}

export default RnnJena
