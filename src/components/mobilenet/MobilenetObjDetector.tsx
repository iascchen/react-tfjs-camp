import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Form, Input, message, Row, Select, Slider, Tabs } from 'antd'

import { layout, tailLayout } from '../../constant'
import { logger, loggerError, STATUS } from '../../utils'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import TfvisDatasetInfoWidget from '../common/tfvis/TfvisDatasetInfoWidget'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'

import ObjDetectorSampleVis from './ObjDetectorSampleVis'
import { MOBILENET_IMAGE_SIZE } from './mobilenetUtils'
import { buildObjectDetectionModel, customLossFunction } from './modelObjDetector'
import ObjectDetectionImageSynthesizer from './dataObjDetector'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

const { TabPane } = Tabs
const { Option } = Select

const SEQUENTIAL_LAYER = 'sequential_1'
const SHOW_SAMPLE = 50

const TRUE_BOUNDING_BOX_LINE_WIDTH = 2
const TRUE_BOUNDING_BOX_STYLE = 'rgb(255,0,0)'
const PREDICT_BOUNDING_BOX_LINE_WIDTH = 2
const PREDICT_BOUNDING_BOX_STYLE = 'rgb(0,0,255)'

const MobilenetObjDetector = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTabCurrent, setTabCurrent] = useState<number>(4)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    // Data
    const [sDataSynth, setDataSynth] = useState<ObjectDetectionImageSynthesizer>()
    const [sTestSet, setTestSet] = useState<tf.TensorContainerObject>()
    const [sNumCircles, setNumCircles] = useState<number>(10)
    const [sNumLines, setNumLines] = useState<number>(10)

    // Model
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sFineTuningLayers, setFineTuningLayers] = useState<tf.layers.Layer[]>()
    const [sLayersOption, setLayersOption] = useState<string[]>()
    const [sCurLayer, setCurLayer] = useState<tf.layers.Layer>()

    // Train
    const [sNumExamples, setNumExamples] = useState<number>(200)

    // Predict
    const [sTrueTarget, setTrueTarget] = useState<number[]>()
    const [sPredictResult, setPredictResult] = useState<number[]>()
    const [sTrueObjectClass, setTrueObjectClass] = useState<string>()
    const [sPredictedObjectClass, setPredictedObjectClass] = useState<string>()

    const historyRef = useRef<HTMLDivElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)

    const [formData] = Form.useForm()
    const [formTrain] = Form.useForm()
    const [formPredict] = Form.useForm()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')

        setStatus(STATUS.LOADING)

        tf.backend()
        setTfBackend(tf.getBackend())

        let _model: tf.LayersModel
        buildObjectDetectionModel().then(
            ({ model, fineTuningLayers }) => {
                _model = model
                setModel(_model)
                setFineTuningLayers(fineTuningLayers)

                const _layerOptions: string[] = fineTuningLayers.map(l => l.name)
                _layerOptions.push(SEQUENTIAL_LAYER)
                setLayersOption(_layerOptions)

                setStatus(STATUS.LOADED)
            },
            (e) => {
                logger(e)
                // eslint-disable-next-line @typescript-eslint/no-floating-promises
                message.error(e.message)
            }
        )

        return () => {
            logger('Model Dispose')
            _model?.dispose()
        }
    }, [])

    useEffect(() => {
        if (!canvasRef.current) {
            return
        }
        logger('init Data Synthesizer ...')

        const synth = new ObjectDetectionImageSynthesizer(canvasRef.current)
        setDataSynth(synth)

        return () => {
            logger('Data Synthesizer Dispose')
            synth?.dispose()
        }
    }, [canvasRef])

    /***********************
     * Functions
     ***********************/

    const drawBoundingBoxes = (canvas: HTMLCanvasElement, trueBoundingBox: number[], predictBoundingBox: number[]): void => {
        tf.util.assert(
            trueBoundingBox != null && trueBoundingBox.length === 4,
            () => 'Expected boundingBoxArray to have length 4, ' + `but got [${trueBoundingBox.join(',')}] instead`)
        tf.util.assert(
            predictBoundingBox != null && predictBoundingBox.length === 4,
            () => 'Expected boundingBoxArray to have length 4, ' + `but got [${predictBoundingBox.join(',')}] instead`)

        let left = trueBoundingBox[0]
        let right = trueBoundingBox[1]
        let top = trueBoundingBox[2]
        let bottom = trueBoundingBox[3]

        const ctx = canvas.getContext('2d')
        if (!ctx) {
            return
        }

        ctx.beginPath()
        ctx.strokeStyle = TRUE_BOUNDING_BOX_STYLE
        ctx.lineWidth = TRUE_BOUNDING_BOX_LINE_WIDTH
        ctx.moveTo(left, top)
        ctx.lineTo(right, top)
        ctx.lineTo(right, bottom)
        ctx.lineTo(left, bottom)
        ctx.lineTo(left, top)
        ctx.stroke()

        ctx.font = '15px Arial'
        ctx.fillStyle = TRUE_BOUNDING_BOX_STYLE
        ctx.fillText('true', left, top)

        left = predictBoundingBox[0]
        right = predictBoundingBox[1]
        top = predictBoundingBox[2]
        bottom = predictBoundingBox[3]

        ctx.beginPath()
        ctx.strokeStyle = PREDICT_BOUNDING_BOX_STYLE
        ctx.lineWidth = PREDICT_BOUNDING_BOX_LINE_WIDTH
        ctx.moveTo(left, top)
        ctx.lineTo(right, top)
        ctx.lineTo(right, bottom)
        ctx.lineTo(left, bottom)
        ctx.lineTo(left, top)
        ctx.stroke()

        ctx.font = '15px Arial'
        ctx.fillStyle = PREDICT_BOUNDING_BOX_STYLE
        ctx.fillText('predicted', left, bottom)
    }

    const runAndVisualizeInference = async (model: tf.LayersModel): Promise<void> => {
        if (!canvasRef.current) {
            return
        }
        // Synthesize an input image and show it in the canvas.
        const synth = new ObjectDetectionImageSynthesizer(canvasRef.current)

        const numExamples = 1
        const numCircles = 10
        const numLineSegments = 10
        const { images, targets } = await synth.generateExampleBatch(numExamples, numCircles, numLineSegments) as tf.TensorContainerObject

        // const t0 = tf.util.now()
        // Runs inference with the model.
        const result = await model.predict(images as tf.Tensor) as tf.Tensor
        const modelOut = Array.from(result.dataSync())
        setPredictResult(modelOut)
        // inferenceTimeMs.textContent = `${(tf.util.now() - t0).toFixed(1)}`

        // Visualize the true and predicted bounding boxes.
        const targetsArray = Array.from((targets as tf.Tensor).dataSync())
        setTrueTarget(targetsArray)

        const boundingBoxArray = targetsArray.slice(1)
        drawBoundingBoxes(canvasRef.current, boundingBoxArray, modelOut.slice(1))
        // Display the true and predict object classes.
        const trueClassName = targetsArray[0] > 0 ? 'rectangle' : 'triangle'
        setTrueObjectClass(trueClassName)
        // trueObjectClass.textContent = trueClassName

        // The model predicts a number to indicate the predicted class
        // of the object. It is trained to predict 0 for triangle and
        // 224 (canvas.width) for rectangel. This is how the model combines
        // the class loss with the bounding-box loss to form a single loss
        // value. Therefore, at inference time, we threshold the number
        // by half of 224 (canvas.width).
        const shapeClassificationThreshold = canvasRef.current.width / 2
        const predictClassName = (modelOut[0] > shapeClassificationThreshold) ? 'rectangle' : 'triangle'
        // predictedObjectClass.textContent = predictClassName
        setPredictedObjectClass(predictClassName)

        // if (predictClassName === trueClassName) {
        //     predictedObjectClass.classList.remove('shape-class-wrong')
        //     predictedObjectClass.classList.add('shape-class-correct')
        // } else {
        //     predictedObjectClass.classList.remove('shape-class-correct')
        //     predictedObjectClass.classList.add('shape-class-wrong')
        // }

        // Tensor memory cleanup.
        tf.dispose([images, targets])
    }

    const train = async (initialTransferEpochs: number, fineTuningEpochs: number, batchSize: number, validationSplit: number): Promise<void> => {
        if (!sModel || !sFineTuningLayers || !sDataSynth) {
            return
        }
        setStatus(STATUS.TRAINING)

        const tBegin = tf.util.now()
        console.log(`Generating ${sNumExamples} training examples...`)
        const { xs, ys } = await sDataSynth.generateExampleBatch(sNumExamples, sNumCircles, sNumLines)

        sModel.compile({ loss: customLossFunction, optimizer: tf.train.rmsprop(5e-3) })
        const callbacks = tfvis.show.fitCallbacks(historyRef?.current, ['loss', 'acc', 'val_loss', 'val_acc'])

        // sDataSynth.generateExample()

        // Initial phase of transfer learning.
        console.log('Phase 1 of 2: initial transfer learning')
        await sModel.fit(xs as tf.Tensor, ys as tf.Tensor, {
            epochs: initialTransferEpochs,
            batchSize,
            validationSplit,
            callbacks
        })

        // Fine-tuning phase of transfer learning.
        // Unfreeze layers for fine-tuning.
        for (const layer of sFineTuningLayers) {
            layer.trainable = true
        }
        sModel.compile({ loss: customLossFunction, optimizer: tf.train.rmsprop(2e-3) })

        // Do fine-tuning.
        // The batch size is reduced to avoid CPU/GPU OOM. This has
        // to do with the unfreezing of the fine-tuning layers above,
        // which leads to higher memory consumption during backpropagation.
        console.log('Phase 2 of 2: fine-tuning phase')
        await sModel.fit(xs as tf.Tensor, ys as tf.Tensor, {
            epochs: fineTuningEpochs,
            batchSize: batchSize / 2,
            validationSplit,
            callbacks
        })

        for (const layer of sFineTuningLayers) {
            layer.trainable = false
        }

        logger(`Model training took ${(tf.util.now() - tBegin) / 1e3} s`)
    }

    const handlePredict = (): void => {
        if (!sModel) {
            return
        }
        setStatus(STATUS.PREDICTING)

        runAndVisualizeInference(sModel).then(
            () => {
                setStatus(STATUS.PREDICTED)
            },
            loggerError
        )
    }

    const handleLayerChange = (value: string): void => {
        logger('handleLayerChange', value)
        const _layer = sModel?.getLayer(value)
        setCurLayer(_layer)
    }

    const handleLoadModelWeight = (): void => {
        // // Save model.
        // // First make sure that the base directory dists.
        // const modelSavePath = modelSaveURL.replace('file://', '')
        // const dirName = path.dirname(modelSavePath)
        // if (!fs.existsSync(dirName)) {
        //     fs.mkdirSync(dirName)
        // }
        // await sModel.save(modelSaveURL)
    }

    const handleSaveModelWeight = (): void => {
        // TODO
    }

    const handleTrain = (values: any): void => {
        logger('handleTrain', values)
        const { initialTransferEpochs, fineTuningEpochs, batchSize, validationSplit } = values
        train(initialTransferEpochs, fineTuningEpochs, batchSize, validationSplit).then(
            () => {
                setStatus(STATUS.TRAINED)
            },
            loggerError
        )
    }

    // const handleTrainParamsChange = (): void => {
    //     const values = formTrain.getFieldsValue()
    //     // logger('handleTrainParamsChange', value)
    //     const { initialTransferEpochs, fineTuningEpochs, batchSize, validationSplit } = values
    //
    // }

    // const handleGenDataTrain = (): void => {
    //     if (!sDataSynth) {
    //         return
    //     }
    //
    //     const values = formTrain.getFieldsValue()
    //     const { numExamples } = values
    //
    //     logger(`Generating ${numExamples} training examples...`)
    //     setStatus(STATUS.LOADING)
    //     sDataSynth.generateExampleBatch(numExamples, sNumCircles, sNumLines).then(
    //         (dataSet) => {
    //             // const { images, targets } = dataSet
    //             if (dataSet) {
    //                 setTestSet(dataSet)
    //                 setStatus(STATUS.LOADED)
    //             }
    //         },
    //         loggerError
    //     )
    // }

    const handleDataParamsChange = (): void => {
        const values = formTrain.getFieldsValue()
        // logger('handleDataParamsChange', values)
        const { numCircles, numLines } = values
        setNumCircles(numCircles)
        setNumLines(numLines)
    }

    const handleGenData = (): void => {
        if (!sDataSynth) {
            return
        }

        const values = formData.getFieldsValue()
        const { numTestExamples, numCircles, numLines } = values

        logger(`Generating ${numTestExamples} testing examples...`)
        setStatus(STATUS.LOADING)
        sDataSynth.generateExampleBatch(numTestExamples, numCircles, numLines).then(
            (dataSet) => {
                // const { images, targets } = dataSet
                if (dataSet) {
                    setTestSet(dataSet)
                    setStatus(STATUS.LOADED)
                }
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

    const dataAdjustCard = (): JSX.Element => {
        return <Card title='Adjust Data' style={{ margin: '8px' }} size='small'>
            <Form {...layout} form={formData} onFinish={handleGenData} onFieldsChange={handleDataParamsChange}
                initialValues={{
                    numTestExamples: 50,
                    numCircles: 10,
                    numLines: 10
                }}>
                <Form.Item name='numTestExamples' label='Test Sample counts'>
                    <Slider min={40} max={100} step={10} marks={{ 40: 40, 70: 70, 100: 100 }} />
                </Form.Item>
                <Form.Item name='numCircles' label='Circles per sample'>
                    <Slider min={5} max={15} step={5} marks={{ 5: 5, 10: 10, 15: 15 }} />
                </Form.Item>
                <Form.Item name='numLines' label='Lines per sample'>
                    <Slider min={5} max={15} step={5} marks={{ 5: 5, 10: 10, 15: 15 }} />
                </Form.Item>
                <Form.Item {...tailLayout} >
                    <Button type='primary' htmlType={'submit'} style={{ width: '60%', margin: '0 20%' }}> Generate Test Set </Button>
                </Form.Item>
                <Form.Item {...tailLayout} >
                    <div>{sStatus}</div>
                </Form.Item>
            </Form>
        </Card>
    }

    const trainAdjustCard = (): JSX.Element => {
        return <Card title='Train' style={{ margin: '8px' }} size='small'>
            <Form {...layout} form={formTrain} onFinish={handleTrain} initialValues={{
                numExamples: 200,
                initialTransferEpochs: 50,
                fineTuningEpochs: 50,
                batchSize: 32,
                validationSplit: 0.15
            }}>
                <Form.Item name='numExamples' label='Sample counts'>
                    <Slider min={200} max={1000} step={200} marks={{ 200: 200, 600: 600, 1000: 1000 }} />
                </Form.Item>
                <Form.Item name='initialTransferEpochs' label='Transfer Epochs' >
                    <Slider min={50} max={150} step={50} marks={{ 50: 50, 100: 100, 150: 150 }} />
                </Form.Item>
                <Form.Item name='fineTuningEpochs' label='Tuning Epochs' >
                    <Slider min={50} max={150} step={50} marks={{ 50: 50, 100: 100, 150: 150 }} />
                </Form.Item>
                <Form.Item name='batchSize' label='Batch Size'>
                    <Slider min={32} max={64} step={16} marks={{ 32: 32, 48: 48, 64: 64 }} />
                </Form.Item>
                <Form.Item name='validationSplit' label='Validation Split'>
                    <Slider min={0.1} max={0.2} step={0.05} marks={{ 0.1: 0.1, 0.15: 0.15, 0.2: 0.2 }} />
                </Form.Item>
                <Form.Item {...tailLayout}>
                    <Button type='primary' htmlType='submit' style={{ width: '30%', margin: '0 10%' }}> Train </Button>
                    {/* <Button onClick={handleGenDataTrain} style={{ width: '30%', margin: '0 10%' }}> Generate </Button> */}
                </Form.Item>
                <Form.Item {...tailLayout}>
                    <Button onClick={handleSaveModelWeight} style={{ width: '30%', margin: '0 10%' }}> Save
                        Weights </Button>
                    <Button onClick={handleLoadModelWeight} style={{ width: '30%', margin: '0 10%' }}> Load
                        Weights </Button>
                </Form.Item>
                <Form.Item {...tailLayout}>
                    <div>status: {sStatus}</div>
                    <div>backend: {sTfBackend}</div>
                </Form.Item>
            </Form>
        </Card>
    }

    return (
        <>
            <canvas ref={canvasRef} style={{ border: '2px dashed lightgray', margin: '8px auto' }}
                height={MOBILENET_IMAGE_SIZE} width={MOBILENET_IMAGE_SIZE} hidden />
            <AIProcessTabs title={'Simple Object Detector based Mobilenet'} current={sTabCurrent} onChange={handleTabChange} >
                <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                    <MarkdownWidget url={'/docs/ai/mobilenet-obj-detector.md'}/>
                </TabPane>
                <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                    <Row>
                        <Col span={12}>
                            {dataAdjustCard()}
                        </Col>
                        <Col span={12}>
                            <Card title={`Test Data Set (Only show ${SHOW_SAMPLE} samples)`} style={{ margin: '8px' }} size='small'>
                                <div>{sTestSet && <TfvisDatasetInfoWidget value={sTestSet}/>}</div>
                                <ObjDetectorSampleVis xDataset={sTestSet?.xs as tf.Tensor} yDataset={sTestSet?.ys as tf.Tensor}
                                    xIsImage pageSize={5} sampleCount={SHOW_SAMPLE} />
                            </Card>
                        </Col>
                    </Row>
                </TabPane>
                <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                    <Row>
                        <Col span={12}>
                            <Card title='Model(Expand from Mobilenet)' style={{ margin: '8px' }} size='small'>
                                <h3 className='centerContainer'>预训练 Mobilenet 模型 : {sStatus}</h3>
                                <h3 className='centerContainer'>截取 Mobilenet 到 conv_pw_11_relu 层</h3>
                                <h3 className='centerContainer'>在其后面增加 sequential_1 全联接层</h3>
                                <Card title='Expand Dense Net' style={{ margin: '8px' }} size='small'>
                                    <TfvisModelWidget model={sModel}/>
                                </Card>
                            </Card>
                        </Col>
                        <Col span={12}>
                            <Card title='Model Training Process' style={{ margin: '8px' }} size='small'>
                                <h3 className='centerContainer'>Step 1: 先仅训练 sequential_1 权重</h3>
                                <h3 className='centerContainer'>Step 2: 再解锁 conv_pw_9、conv_pw_10、conv_pw_11 调优各层权重</h3>
                                <Card title='Tuning Layers' style={{ margin: '8px' }} size='small'>
                                    <Form {...layout} initialValues={{
                                        layer: SEQUENTIAL_LAYER
                                    }}>
                                        <Form.Item name='layer' label='Show Layer'>
                                            <Select onChange={handleLayerChange} >
                                                {sLayersOption?.map((name, index) => {
                                                    return <Option key={index} value={name}>{name}</Option>
                                                })}
                                            </Select>
                                        </Form.Item>
                                    </Form>
                                </Card>
                                <Card title='Layer Info' style={{ margin: '8px' }} size='small'>
                                    <TfvisLayerWidget layer={sCurLayer}/>
                                </Card>
                            </Card>
                        </Col>
                    </Row>
                </TabPane>
                <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                    <Row>
                        <Col span={8}>
                            {trainAdjustCard()}
                            {dataAdjustCard()}
                        </Col>
                        <Col span={10}>
                            <Card title={`Test Data Set (Only show ${SHOW_SAMPLE} samples)`} style={{ margin: '8px' }} size='small'>
                                <div>{sTestSet && <TfvisDatasetInfoWidget value={sTestSet}/>}</div>
                                <ObjDetectorSampleVis xDataset={sTestSet?.xs as tf.Tensor} yDataset={sTestSet?.ys as tf.Tensor}
                                    xIsImage pageSize={5} sampleCount={SHOW_SAMPLE} />
                            </Card>
                        </Col>
                        <Col span={6}>
                            <Card title='Training History' style={{ margin: '8px' }} size='small'>
                                <div ref={historyRef} />
                            </Card>
                        </Col>
                    </Row>
                </TabPane>
                <TabPane tab='&nbsp;' key={AIProcessTabPanes.PREDICT}>
                    <Col span={12}>
                        <Card title='Predict' style={{ margin: '8px' }} size='small'>
                            <Button onClick={handlePredict} style={{ width: '30%', margin: '0 10%' }}> Predict </Button>

                            <p>True: {sTrueObjectClass} VS. Predict: {sPredictedObjectClass}</p>
                            <p>True: {JSON.stringify(sTrueTarget)}</p>
                            <p>Predict {JSON.stringify(sPredictResult)}</p>
                        </Card>
                    </Col>
                </TabPane>
            </AIProcessTabs>
        </>
    )
}

export default MobilenetObjDetector
