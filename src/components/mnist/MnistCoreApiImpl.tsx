import React, { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs-core'
import { Button, Card, Col, Form, Row, Select, Slider, Tabs } from 'antd'

import { ITrainInfo, logger, loggerError, STATUS } from '../../utils'
import { layout, normalLayout, tailLayout } from '../../constant'

import SampleDataVis from '../common/tensor/SampleDataVis'
import TfvisHistoryWidget from '../common/tfvis/TfvisHistoryWidget'
import TfvisDatasetInfoWidget from '../common/tfvis/TfvisDatasetInfoWidget'
import DrawPanelWidget from '../common/tensor/DrawPanelWidget'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'

import { MnistDatasetPng } from './MnistDatasetPng'
import { MnistDatasetGz } from './MnistDatasetGz'
import * as modelCore from './modelCoreModel'

const { Option } = Select
const { TabPane } = Tabs

// Data
const DATA_SOURCE = ['mnist-png', 'mnist']
const TRAIN_STEPS = 60
const BATCH_SIZES = [64, 128, 256, 512]
const SHOW_SAMPLE = 50

// Train
const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

const MnistCoreApiImpl = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTabCurrent, setTabCurrent] = useState<number>(2)

    // General
    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>()

    // Data
    const [sDataSourceName, setDataSourceName] = useState()
    const [sDataSet, setDataSet] = useState<MnistDatasetPng | MnistDatasetGz>()
    const [sTrainSet, setTrainSet] = useState<tf.TensorContainerObject>()
    const [sTestSet, setTestSet] = useState<tf.TensorContainerObject>()

    // Predict
    const [sPredictResult, setPredictResult] = useState<tf.Tensor>()
    const [logMsg, setLogMsg] = useState<ITrainInfo>()
    const [sDrawPred, setDrawPred] = useState<tf.Tensor>()

    const [formTrain] = Form.useForm()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init data set ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.WAITING)

        let mnistDataset: MnistDatasetGz | MnistDatasetPng
        if (sDataSourceName === 'mnist') {
            mnistDataset = new MnistDatasetGz(sDataSourceName)
        } else {
            mnistDataset = new MnistDatasetPng()
        }

        let tSet: tf.TensorContainerObject
        let vSet: tf.TensorContainerObject
        mnistDataset.loadData().then(
            () => {
                setDataSet(mnistDataset)

                tSet = mnistDataset.nextTrainBatch(SHOW_SAMPLE)
                vSet = mnistDataset.nextTestBatch(SHOW_SAMPLE)
                setTrainSet(tSet)
                setTestSet(vSet)

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

    /***********************
     * useEffects only for dispose
     ***********************/

    useEffect(() => {
        // Do Nothing
        return () => {
            logger('Predict Result Dispose')
            sPredictResult?.dispose()
        }
    }, [sPredictResult])

    /***********************
     * Functions
     ***********************/

    const pushTrainingLog = (iteration: number, loss: number): void => {
        addTrainInfo({ iteration: iteration, logs: { loss } })
        predictModel(sTestSet?.xs as tf.Tensor)
    }

    const trainModel = (_dataset: MnistDatasetPng | MnistDatasetGz, steps = TRAIN_STEPS,
        batchSize = 128, learningRate = 0.01): void => {
        if (!_dataset) {
            return
        }

        setStatus(STATUS.WAITING)
        const beginMs = performance.now()
        modelCore.train(_dataset, pushTrainingLog, steps, batchSize, learningRate).then(
            () => {
                setStatus(STATUS.TRAINED)

                const secSpend = (performance.now() - beginMs) / 1000
                logger(`Spend : ${secSpend.toString()}s`)
            },
            loggerError
        )
    }

    const predictModel = (_xs: tf.Tensor): void => {
        if (!_xs) {
            return
        }
        const preds = modelCore.predict(_xs)
        setPredictResult(preds)
    }

    const addTrainInfo = (info: ITrainInfo): void => {
        setLogMsg(info)
    }

    const handleDataSourceChange = (value: string): void => {
        setDataSourceName(value)
    }

    const handleTrain = (): void => {
        if (!sDataSet) {
            return
        }
        const values = formTrain.getFieldsValue()
        const { steps, batchSize, learningRate } = values
        // Train the model using the data.
        trainModel(sDataSet, steps, batchSize, learningRate)
    }

    const handleDrawSubmit = (data: tf.Tensor): void => {
        // logger('handleDrawSubmit', data.shape)
        const pred = modelCore.predict(data)
        setDrawPred(pred)
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    /***********************
     * Render
     ***********************/

    const dataAdjustCard = (): JSX.Element => {
        return (
            <Card title='Data Source' style={{ margin: '8px' }} size='small'>
                <Form {...layout} initialValues={{
                    dataSource: 'mnist-png'
                }}>
                    <Form.Item name='dataSource' label='Select Data Source'>
                        <Select onChange={handleDataSourceChange}>
                            {DATA_SOURCE.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <div>Status: {sStatus}</div>
                    </Form.Item>
                    <Form.Item {...normalLayout}>
                        <ul>
                            <li><div style={{ color: 'red' }}>!!! 请注意 !!! 如果您是从 Github 上克隆项目，在运行之前，
                                请先前往目录 ./public/preload/data , 运行 download_mnist_data.sh 脚本，下载所需的数据。</div></li>
                            <li>如果您是在 Docker 中运行，数据已经预先放在相应的目录下。</li>
                            <li>由于数据量较大，多次加载会影响程序运行效率。</li>
                            <li><div style={{ color: 'red' }}>如果 Train Data Set 中的图片未能正常显示，表明要加载的训练集大小超过了您的内存。
                                您可以减少代码中 MnistDataset*.ts 里的 NUM_TRAIN_ELEMENTS 使用较小的数据集</div></li>
                        </ul>
                    </Form.Item>
                </Form>
            </Card>
        )
    }

    const trainAdjustCard = (): JSX.Element => {
        return (
            <Card title='Train' style={{ margin: '8px' }} size='small'>
                <Form {...layout} form={formTrain} onFinish={handleTrain} initialValues={{
                    learningRate: 0.1,
                    batchSize: 256,
                    steps: 30
                }}>
                    <Form.Item name='steps' label='Train Step'>
                        <Slider min={30} max={150} step={30} marks={{ 30: 30, 90: 90, 150: 150 }} />
                    </Form.Item>
                    <Form.Item name='batchSize' label='Batch Size'>
                        <Select >
                            {BATCH_SIZES.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item name='learningRate' label='Learning Rate'>
                        <Select >
                            {LEARNING_RATES.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <Button type='primary' htmlType={'submit'} style={{ width: '30%', margin: '0 10%' }}> Train </Button>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <div>Status: {sStatus}</div>
                        <div>Backend: {sTfBackend}</div>
                    </Form.Item>
                </Form>
            </Card>
        )
    }

    return (
        <AIProcessTabs title={'MNIST Core API'} current={sTabCurrent} onChange={handleTabChange}
            invisiblePanes={[AIProcessTabPanes.MODEL]} >
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/ai/mnist.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={8}>
                        {dataAdjustCard()}
                    </Col>
                    <Col span={8}>
                        <Card title={`Train Data Set (Only show ${SHOW_SAMPLE} samples)`} style={{ margin: '8px' }} size='small'>
                            <div>{sTrainSet && <TfvisDatasetInfoWidget value={sTrainSet}/>}</div>
                            <SampleDataVis xDataset={sTrainSet?.xs as tf.Tensor} yDataset={sTrainSet?.ys as tf.Tensor}
                                xIsImage pageSize={5} sampleCount={SHOW_SAMPLE} />
                        </Card>
                    </Col>
                    <Col span={8}>
                        <Card title={`Validate Data Set (Only show ${SHOW_SAMPLE} samples)`} style={{ margin: '8px' }} size='small'>
                            <div>{sTestSet && <TfvisDatasetInfoWidget value={sTestSet}/>}</div>
                            <SampleDataVis xDataset={sTestSet?.xs as tf.Tensor} yDataset={sTestSet?.ys as tf.Tensor}
                                xIsImage pageSize={5} sampleCount={SHOW_SAMPLE} />
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                <Row>
                    <Col span={6}>
                        {trainAdjustCard()}
                    </Col>
                    <Col span={10}>
                        <Card title='Evaluate' style={{ margin: '8px' }} size='small'>
                            <SampleDataVis xDataset={sTestSet?.xs as tf.Tensor} yDataset={sTestSet?.ys as tf.Tensor}
                                pDataset={sPredictResult} xIsImage pageSize={10} sampleCount={SHOW_SAMPLE}/>
                        </Card>
                    </Col>
                    <Col span={8}>
                        <Card title='Training History' style={{ margin: '8px' }} size='small'>
                            <TfvisHistoryWidget logMsg={logMsg} debug />
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.PREDICT}>
                <Col span={8}>
                    <DrawPanelWidget onSubmit={handleDrawSubmit} prediction={sDrawPred} />
                </Col>
            </TabPane>
        </AIProcessTabs>
    )
}

export default MnistCoreApiImpl
