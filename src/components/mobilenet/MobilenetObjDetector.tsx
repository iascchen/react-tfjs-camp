import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Form, Input, message, Row, Select, Tabs } from 'antd'

import { layout, tailLayout } from '../../constant'
import { ILabeledImageFileJson, ILabeledImageSet, ILayerSelectOption, logger, loggerError, STATUS } from '../../utils'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'

import { MOBILENET_IMAGE_SIZE } from './mobilenetUtils'
import { buildObjectDetectionModel, customLossFunction } from './modelObjDetector'
import ObjectDetectionImageSynthesizer from './dataObjDetector'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

const { TabPane } = Tabs
const { Option } = Select

const TRUE_BOUNDING_BOX_LINE_WIDTH = 2
const TRUE_BOUNDING_BOX_STYLE = 'rgb(255,0,0)'
const PREDICT_BOUNDING_BOX_LINE_WIDTH = 2
const PREDICT_BOUNDING_BOX_STYLE = 'rgb(0,0,255)'

const DEFAULT_TRAIN_PARAMS = {
    initialTransferEpochs: 100,
    fineTuningEpochs: 100,
    batchSize: 64,
    validationSplit: 0.15
}

const MobilenetTransferWidget = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTabCurrent, setTabCurrent] = useState<number>(4)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sNumExamples, setNumExamples] = useState<number>(512)
    const [sNumCircles, setNumCircles] = useState<number>(10)
    const [sNumLines, setNumLines] = useState<number>(10)

    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sFineTuningLayers, setFineTuningLayers] = useState<tf.layers.Layer[]>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [sCurLayer, setCurLayer] = useState<tf.layers.Layer>()

    // const [sTrainSet, setTrainSet] = useState<tf.TensorContainerObject>()

    const [sTrueTarget, setTrueTarget] = useState<number[]>()
    const [sPredictResult, setPredictResult] = useState<number[]>()
    const [sTrueObjectClass, setTrueObjectClass] = useState<string>()
    const [sPredictedObjectClass, setPredictedObjectClass] = useState<string>()

    const historyRef = useRef<HTMLDivElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)

    const [form] = Form.useForm()
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
                setModel(model)
                setFineTuningLayers(fineTuningLayers)

                const _layerOptions: ILayerSelectOption[] = _model?.layers.map((l, index) => {
                    return { name: l.name, index }
                })
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

    // useEffect(() => {
    //     if (!canvasRef.current) {
    //         return
    //     }
    //     logger('init data set ...')
    //
    //     const dataHandler = new ObjectDetectionImageSynthesizer(canvasRef.current)
    //
    //     return () => {
    //         logger('Data Dispose')
    //         dataHandler.dispose()
    //     }
    // }, [])

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
        if (!sModel || !sFineTuningLayers || !canvasRef.current) {
            return
        }

        setStatus(STATUS.TRAINING)

        const tBegin = tf.util.now()
        console.log(`Generating ${sNumExamples} training examples...`)
        const synthDataCanvas = canvasRef.current // canvasRef.current.createCanvas(CANVAS_SIZE, CANVAS_SIZE);
        const synth = new ObjectDetectionImageSynthesizer(synthDataCanvas)
        const { images, targets } = await synth.generateExampleBatch(sNumExamples, sNumCircles, sNumLines) as tf.TensorContainerObject

        sModel.compile({ loss: customLossFunction, optimizer: tf.train.rmsprop(5e-3) })
        // sModel.summary()

        const callbacks = tfvis.show.fitCallbacks(historyRef?.current, ['loss', 'acc', 'val_loss', 'val_acc'])

        // Initial phase of transfer learning.
        console.log('Phase 1 of 2: initial transfer learning')
        await sModel.fit(images as tf.Tensor, targets as tf.Tensor, {
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
        // sModel.summary()

        // Do fine-tuning.
        // The batch size is reduced to avoid CPU/GPU OOM. This has
        // to do with the unfreezing of the fine-tuning layers above,
        // which leads to higher memory consumption during backpropagation.
        console.log('Phase 2 of 2: fine-tuning phase')
        await sModel.fit(images as tf.Tensor, targets as tf.Tensor, {
            epochs: fineTuningEpochs,
            batchSize: batchSize / 2,
            validationSplit,
            callbacks
        })
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

    const handleLayerChange = (value: number): void => {
        logger('handleLayerChange', value)
        const _layer = sModel?.getLayer(undefined, value)
        setCurLayer(_layer)
    }

    const handleLabeledImagesSubmit = (value: ILabeledImageFileJson): void => {
        // logger('handleLabeledImagesSubmit', value)
        //
        // const labeledImageSetList = value.labeledImageSetList
        // setLabeledImgs(labeledImageSetList)
    }

    const handleLoadJson = (values: ILabeledImageSet[]): void => {
        // sLabeledImgs && arrayDispose(sLabeledImgs)
        // setLabeledImgs(values)
    }

    const handleLoadModelWeight = (): void => {
        // TODO : Load saved model
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        message.info('TODO: Not Implemented')
    }

    const handleSaveModelWeight = (): void => {
        // TODO
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    /***********************
     * Render
     ***********************/

    // const _tensorX = sTrainSet?.xs as tf.Tensor4D
    // const _tensorY = sTrainSet?.ys as tf.Tensor

    return (
        <AIProcessTabs title={'Simple Object Detector based Mobilenet'} current={sTabCurrent} onChange={handleTabChange} >
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/mobilenet.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={12}>
                        <Card title='Images Label Panel' style={{ margin: '8px' }} size='small'>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Train Set' style={{ margin: '8px' }} size='small'>
                            {/* <div> XShape: {_tensorX?.shape.join(',')}, YShape: {_tensorY?.shape.join(',')}</div> */}
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Row>
                    <Col span={12}>
                        <Card title='Model(Expand from Mobilenet)' style={{ margin: '8px' }} size='small'>
                            <TfvisModelWidget model={sModel}/>
                            <p>backend: {sTfBackend}</p>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Model(Expand from Mobilenet)' style={{ margin: '8px' }} size='small'>
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
                        <Card title='Mobilenet + Simple Object Detect' style={{ margin: '8px' }} size='small'>
                            <Form {...layout} form={form} onFinish={handleTrain} initialValues={DEFAULT_TRAIN_PARAMS}>
                                <Form.Item name='initialTransferEpochs' label='Transfer Epochs' >
                                    <Input />
                                </Form.Item>
                                <Form.Item name='fineTuningEpochs' label='Tuning Epochs' >
                                    <Input />
                                </Form.Item>
                                <Form.Item name='batchSize' label='Batch Size'>
                                    <Input />
                                </Form.Item>
                                <Form.Item name='validationSplit' label='Validation Split'>
                                    <Input />
                                </Form.Item>
                                <Form.Item {...tailLayout}>
                                    <Button type='primary' htmlType='submit' style={{ width: '30%', margin: '0 10%' }}> Train </Button>
                                </Form.Item>
                            </Form>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Mobilenet Simple Object Detect Train Set' style={{ margin: '8px' }} size='small'>
                            <div>
                                <Button onClick={handleSaveModelWeight} style={{ width: '30%', margin: '0 10%' }}> Save
                                    Model </Button>
                                <Button onClick={handleLoadModelWeight} style={{ width: '30%', margin: '0 10%' }}> Load
                                    Model </Button>
                                <div>status: {sStatus}</div>
                            </div>
                            <p>backend: {sTfBackend}</p>
                        </Card>

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
                        <canvas ref={canvasRef} style={{ border: '2px dashed lightgray', margin: '8px auto' }}
                            height={MOBILENET_IMAGE_SIZE} width={MOBILENET_IMAGE_SIZE} />
                        <p>True: {sTrueObjectClass} VS. Predict: {sPredictedObjectClass}</p>
                        <p>True: {JSON.stringify(sTrueTarget)}</p>
                        <p>Predict {JSON.stringify(sPredictResult)}</p>
                    </Card>
                </Col>
            </TabPane>
        </AIProcessTabs>
    )
}

export default MobilenetTransferWidget
