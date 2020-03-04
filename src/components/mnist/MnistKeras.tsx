import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Divider, Row, Select, Tabs } from 'antd'

import SampleDataVis from '../common/tensor/SampleDataVis'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import TfvisDatasetInfoWidget from '../common/tfvis/TfvisDatasetInfoWidget'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'
import DrawPanelWidget from '../common/tensor/DrawPanelWidget'

import { ILayerSelectOption, ITrainInfo, logger, STATUS } from '../../utils'
import { MnistGzDataset } from './dataGz'
import { MnistCoreDataset } from './dataCore'
import { addCnnLayers, addDenseLayers, addSimpleConvLayers } from './model'
import TfvisHistoryWidget from '../common/tfvis/TfvisHistoryWidget'

// // eslint-disable-next-line @typescript-eslint/no-var-requires
// const tfvis = require('@tensorflow/tfjs-vis')

const { Option } = Select
const { TabPane } = Tabs

const EPOCHS = 10
const VALID_SPLIT = 0.15

const DATA_SOURCE = ['Web', 'Gz']
const MODELS = ['dense', 'cnn-pooling', 'cnn-dropout']
const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
const BATCH_SIZES = [64, 128, 256, 512, 1024]

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

    const [sModelName, setModelName] = useState('cnn-pooling')
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [curLayer, setCurLayer] = useState<tf.layers.Layer>()

    const [sLearnRate, setLearnRate] = useState<number>(0.3)
    const [sBatchSize, setBatchSize] = useState<number>(256)

    const [logMsg, setLogMsg] = useState<ITrainInfo>()

    const [predictSet, setPredictSet] = useState<tf.TensorContainerObject>()
    const [predictResult, setPredictResult] = useState<tf.Tensor>()

    const [drawPred, setDrawPred] = useState<tf.Tensor>()

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
        if (!sModel) {
            return
        }
        logger('init model optimizer...', sLearnRate)

        // const optimizer = 'rmsprop'
        const optimizer = tf.train.sgd(sLearnRate)
        sModel.compile({ optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] })
    }, [sModel, sLearnRate])

    useEffect(() => {
        logger('init data set ...')

        setStatus(STATUS.LOADING)

        let mnistDataset: MnistGzDataset | MnistCoreDataset
        if (sDataSourceName === 'Gz') {
            mnistDataset = new MnistGzDataset()
        } else {
            mnistDataset = new MnistCoreDataset()
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

        const beginMs = performance.now()

        // Call `model.fit` to train the model.
        let trainBatchCount = 0
        let iteration = 0
        _model.fit(_trainDataset.xs as tf.Tensor, _trainDataset.ys as tf.Tensor, {
            epochs: EPOCHS,
            batchSize: sBatchSize,
            validationSplit: VALID_SPLIT,
            // callbacks: tfvis.show.fitCallbacks(historyRef.current, ['loss', 'acc', 'val_loss', 'val_acc'])
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    logger('onEpochEnd', epoch)

                    // const secPerEpoch = (performance.now() - beginMs) / (1000 * (epoch + 1))
                    logs && addTrainInfo({ iteration: iteration++, logs })
                    predictModel(_model, predictSet?.xs as tf.Tensor)

                    await tf.nextFrame()
                },
                onBatchEnd: (batch, logs) => {
                    trainBatchCount++
                    logs && addTrainInfo({ iteration: iteration++, logs })
                    if (batch % 50 === 0) {
                        logger(`onBatchEnd: ${batch.toString()} / ${trainBatchCount.toString()}`)
                        predictModel(_model, predictSet?.xs as tf.Tensor)
                    }
                    // await tf.nextFrame()
                }
            }
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
        if (!sModel) {
            return
        }

        // logger('handleDrawSubmit', data.shape)
        const pred = tf.tidy(() => sModel.predict(data)) as tf.Tensor
        logger('handleDrawSubmit', pred.dataSync())
        setDrawPred(pred)
    }

    const handleLearnRateChange = (value: number): void => {
        setLearnRate(value)
    }

    const handleBatchSizeChange = (value: number): void => {
        setBatchSize(value)
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    /***********************
     * Render
     ***********************/

    return (
        <AIProcessTabs title={'MNIST LayerModel'} current={sTabCurrent} onChange={handleTabChange} >
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
                <Card title='Data Set' style={{ margin: '8px' }} size='small'>
                    <div style={{ color: 'red' }}>!!! ATTENTION !!! Please go to ./public/data folder, run `download_data.sh`</div>
                    <div>trainSet: {trainSet && <TfvisDatasetInfoWidget value={trainSet}/>}</div>
                    <div>validSet: {validSet && <TfvisDatasetInfoWidget value={validSet}/>}</div>
                </Card>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Row>
                    <Col span={12}>
                        <Card title='Model' style={{ margin: '8px' }} size='small'>
                            <div>
                        Select Model : <Select onChange={handleModelChange} defaultValue={'cnn-pooling'}>
                                    {MODELS.map((v) => {
                                        return <Option key={v} value={v}>{v}</Option>
                                    })}
                                </Select>
                                <TfvisModelWidget model={sModel}/>
                            </div>
                            <Divider />
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
                    </Col>
                    <Col span={12}>
                        <Card title='Adjust Super Params' style={{ margin: '8px' }} size='small'>
                            <div>
                        Select Learning Rate : <Select onChange={handleLearnRateChange} defaultValue={0.3}>
                                    {LEARNING_RATES.map((v) => {
                                        return <Option key={v} value={v}>{v}</Option>
                                    })}
                                </Select>
                            </div>
                            <div>
                        Select Learning Rate : <Select onChange={handleBatchSizeChange} defaultValue={256}>
                                    {BATCH_SIZES.map((v) => {
                                        return <Option key={v} value={v}>{v}</Option>
                                    })}
                                </Select>
                            </div>
                        </Card>
                    </Col>
                </Row>
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
                            {/* <div ref={historyRef} /> */}
                            <TfvisHistoryWidget logMsg={logMsg} debug />
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
