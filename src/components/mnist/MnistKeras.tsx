import React, {useEffect, useRef, useState} from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Row, Select, Tabs } from 'antd'

import SampleDataVis from '../common/tensor/SampleDataVis'
import TfvisHistoryWidget from '../common/tfvis/TfvisHistoryWidget'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import TfvisDatasetInfoWidget from '../common/tfvis/TfvisDatasetInfoWidget'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'
import DrawPanelWidget from '../common/tensor/DrawPanelWidget'

import { ILayerSelectOption, ITrainInfo, logger, STATUS } from '../../utils'
import { MnistGzDataset } from './dataGz'
import { addCnnLayers, addDenseLayers, addSimpleConvLayers } from './model'
import { MnistWebDataset } from './data'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

const { Option } = Select
const { TabPane } = Tabs

const EPOCHS = 10
const BATCH_SIZE = 128
const VALID_SPLIT = 0.15
const LEARNING_RATE = 0.1

const DATA_SOURCE = ['Web', 'Gz']
const Models = ['dense', 'cnn-pooling', 'cnn-dropout']
// const LEARNING_RATE = [0.0001, 0.001, 0.01, 0.1, 1, 3, 5]

const MnistKeras = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTabCurrent, setTabCurrent] = useState<number>(1)

    const [tfBackend, setTfBackend] = useState<string>()
    const [status, setStatus] = useState<STATUS>(STATUS.INIT)
    const [errors, setErrors] = useState()

    const [sDataSourceName, setDataSourceName] = useState('Web')
    const [trainSet, setTrainSet] = useState<tf.TensorContainerObject>()
    const [validSet, setValidSet] = useState<tf.TensorContainerObject>()

    const [sModelName, setModelName] = useState('dense')
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [curLayer, setCurLayer] = useState<tf.layers.Layer>()
    const [logMsg, setLogMsg] = useState<ITrainInfo>()

    const [predictSet, setPredictSet] = useState<tf.TensorContainerObject>()
    const [predictResult, setPredictResult] = useState<tf.Tensor>()

    const [drawPred, setDrawPred] = useState<tf.Tensor>()

    const historyRef = useRef<HTMLDivElement>(null)

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        const _model = tf.sequential()
        switch (sModelName) {
            case 'dense' :
                addDenseLayers(_model)
                break
            case 'cnn-pooling' :
                addSimpleConvLayers(_model)
                break
            case 'cnn-dropout' :
                addCnnLayers(_model)
                break
        }

        // const optimizer = 'rmsprop'
        const optimizer = tf.train.sgd(LEARNING_RATE)
        _model.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] })
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

    useEffect(() => {
        logger('init data set ...')

        setStatus(STATUS.LOADING)

        let mnistDataset: MnistGzDataset | MnistWebDataset
        if (sDataSourceName === 'Gz') {
            mnistDataset = new MnistGzDataset()
        } else {
            mnistDataset = new MnistWebDataset()
        }

        let tSet: tf.TensorContainerObject
        let vSet: tf.TensorContainerObject
        mnistDataset.loadData().then(
            () => {
                tSet = mnistDataset.getTrainData()
                vSet = mnistDataset.getTestData()

                setTrainSet(tSet)
                setValidSet(vSet)

                setStatus(STATUS.LOADED)
            },
            (error) => {
                logger(error)
            }
        )

        return () => {
            logger('Data Set Dispose')
            tf.dispose([tSet.xs, tSet.ys])
            tf.dispose([vSet.xs, vSet.ys])
        }
    }, [sDataSourceName])

    useEffect(() => {
        logger('init predict data set ...')

        const xTest = validSet?.xs as tf.Tensor
        const yTest = validSet?.ys as tf.Tensor

        const [ys] = tf.tidy(() => {
            const ys = yTest?.argMax(-1)
            return [ys]
        })
        setPredictSet({ xs: xTest, ys })
    }, [validSet])

    /***********************
     * useEffects only for dispose
     ***********************/

    useEffect(() => {
        // Do Nothing

        return () => {
            logger('Predict Set Dispose')
            tf.dispose([predictSet?.xs, predictSet?.ys])
        }
    }, [predictSet])

    useEffect(() => {
        // Do Nothing
        return () => {
            logger('Predict Result Dispose')
            predictResult?.dispose()
        }
    }, [predictResult])

    /***********************
     * Functions
     ***********************/

    const trainModel = (_model: tf.LayersModel, _trainDataset: tf.TensorContainerObject): void => {
        if (!_model || !_trainDataset) {
            return
        }

        setStatus(STATUS.TRAINING)

        // We'll keep a buffer of loss and accuracy values over time.
        const trainBatchCount = 0
        const beginMs = performance.now()

        // Call `model.fit` to train the model.
        const iteration = 0
        _model.fit(_trainDataset.xs as tf.Tensor, _trainDataset.ys as tf.Tensor, {
            epochs: EPOCHS,
            batchSize: BATCH_SIZE,
            validationSplit: VALID_SPLIT,
            callbacks: tfvis.show.fitCallbacks(historyRef.current, ['loss', 'acc', 'val_loss', 'val_acc'])
        }).then(
            () => {
                setStatus(STATUS.TRAINED)

                const secSpend = (performance.now() - beginMs) / 1000
                logger(`Spend : ${secSpend.toString()}s`)
            },
            (error) => {
                setErrors(error)
            }
        )
    }

    const evaluateModel = (_model: tf.LayersModel, _validDataset: tf.TensorContainerObject): void => {
        if (!_model || !_validDataset) {
            return
        }
        const evalOutput = _model.evaluate(_validDataset.xs as tf.Tensor, _validDataset.ys as tf.Tensor) as tf.Tensor[]
        logger(`Final evaluate Loss: ${evalOutput[0].dataSync()[0].toFixed(3)} %`)
        logger(`Final evaluate Accuracy: ${evalOutput[1].dataSync()[0].toFixed(3)} %`)
    }

    const predictModel = (_model: tf.LayersModel, _xs: tf.Tensor): void => {
        if (!_model || !_xs) {
            return
        }

        const [preds] = tf.tidy(() => {
            const preds = (_model.predict(_xs) as tf.Tensor).argMax(-1)
            return [preds]
        })
        setPredictResult(preds)
    }

    const addTrainInfo = (info: ITrainInfo): void => {
        setLogMsg(info)
    }

    const handleTrain = (): void => {
        if (!sModel || !trainSet) {
            return
        }
        // Train the model using the data.
        trainModel(sModel, trainSet)
    }

    const handleEvaluate = (): void => {
        if (!sModel || !validSet) {
            return
        }
        // Evaluate the model using the data.
        evaluateModel(sModel, validSet)
    }

    const handleDataSourceChange = (value: string): void => {
        setDataSourceName(value)
    }

    const handleModelChange = (value: string): void => {
        setModelName(value)
    }

    const handleLayerChange = (value: number): void => {
        logger('handleLayerChange', value)
        const _layer = sModel?.getLayer(undefined, value)
        setCurLayer(_layer)
    }

    const handleDrawSubmit = (data: tf.Tensor): void => {
        // logger('handleDrawSubmit', data.shape)
        const pred = sModel?.predict(data) as tf.Tensor
        logger('handleDrawSubmit', pred.dataSync())
        setDrawPred(pred)
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    /***********************
     * Render
     ***********************/

    return (
        <AIProcessTabs title={'MNIST LayerModel'} current={sTabCurrent} onChange={handleTabChange} docUrl={'/docs/rnnJena.md'}>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/mnist.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Card title='Data' style={{ margin: '8px' }} size='small'>
                    <div>
                        Select Data Source: <Select onChange={handleDataSourceChange} defaultValue={'Web'}>
                            {DATA_SOURCE.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </div>
                </Card>
                <Card title='Train' style={{ margin: '8px' }} size='small'>
                    <div style={{ color: 'red' }}>!!! ATTENTION !!! Please go to ./public/data folder, run `download_data.sh`</div>
                    <div>trainSet: {trainSet && <TfvisDatasetInfoWidget value={trainSet}/>}</div>
                    <div>validSet: {validSet && <TfvisDatasetInfoWidget value={validSet}/>}</div>
                </Card>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Card title='Model' style={{ margin: '8px' }} size='small'>
                    <div>
                        Select Model : <Select onChange={handleModelChange} defaultValue={'dense'}>
                            {Models.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                        <TfvisModelWidget model={sModel}/>
                    </div>
                    <div>
                        Select Layer : <Select onChange={handleLayerChange} defaultValue={0}>
                            {sLayersOption?.map((v) => {
                                return <Option key={v.index} value={v.index}>{v.name}</Option>
                            })}
                        </Select>
                        <TfvisLayerWidget layer={curLayer}/>
                    </div>

                    <p>backend: {tfBackend}</p>
                </Card>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                <Row>
                    <Col span={24}>
                        <Card title='Train' style={{ margin: '8px' }} size='small'>
                            <Button onClick={handleTrain} type='primary'> Train </Button>
                            <Button onClick={handleEvaluate}> Evaluate </Button>
                            <p>status: {status}</p>
                            <p>errors: {errors}</p>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Evaluate' style={{ margin: '8px' }} size='small'>
                            <SampleDataVis xDataset={predictSet?.xs as tf.Tensor} yDataset={predictSet?.ys as tf.Tensor}
                                pDataset={predictResult} xIsImage />
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Training History' style={{ margin: '8px' }} size='small'>
                            <div ref={historyRef} />
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.PREDICT}>
                <Col span={12}>
                    <DrawPanelWidget onSubmit={handleDrawSubmit} prediction={drawPred} />
                </Col>
            </TabPane>
        </AIProcessTabs>
    )
}

export default MnistKeras
