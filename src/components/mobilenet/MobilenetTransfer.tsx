import React, { useEffect, useReducer, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Form, message, Row, Select, Slider, Tabs, Upload } from 'antd'
import { SaveOutlined, UploadOutlined } from '@ant-design/icons'
import { RcFile, UploadChangeParam } from 'antd/es/upload'
import { UploadFile } from 'antd/es/upload/interface'

import {
    checkUploadDone,
    getUploadFileArray,
    getUploadFileBase64,
    ILabeledImage,
    ILabeledImageFileJson,
    ILabeledImageSet,
    ILabelMap,
    ILayerSelectOption,
    logger,
    loggerError,
    STATUS
} from '../../utils'
import { layout, tailLayout } from '../../constant'
import WebCamera, { IWebCameraHandler } from '../common/tensor/WebCamera'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import LabeledCaptureInputSet from '../common/tensor/LabeledCaptureInputSet'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import ImageUploadWidget from '../common/tensor/ImageUploadWidget'
import MarkdownWidget from '../common/MarkdownWidget'
import TensorImageThumbWidget from '../common/tensor/TensorImageThumbWidget'

import { decodeImageTensor, encodeImageTensor, formatImageForMobileNet, MOBILENET_IMAGE_SIZE } from './mobilenetUtils'
import { createModel, createTruncatedMobileNet } from './modelTransfer'
import { TransferDataset } from './dataTransfer'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

const { Option } = Select
const { TabPane } = Tabs

const IMAGE_HEIGHT = 100
const MOBLILENET_CONFIG = {
    // facingMode: 'user',
    resizeWidth: MOBILENET_IMAGE_SIZE,
    resizeHeight: MOBILENET_IMAGE_SIZE,
    centerCrop: false
}

// Model
const DENSE_UNITS = [10, 100, 200]

// Train
const BATCH_SIZES = [32, 64]
const LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03]

const MobileNetTransfer = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTabCurrent, setTabCurrent] = useState<number>(4)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sTruncatedModel, setTruncatedModel] = useState<tf.LayersModel>()
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sLayersOption, setLayersOption] = useState<ILayerSelectOption[]>()
    const [sCurLayer, setCurLayer] = useState<tf.layers.Layer>()

    // Data
    const [waitingPush, forceWaitingPush] = useReducer((x: number) => x + 1, 0)
    const [sUploadingJson, setUploadingJson] = useState<UploadFile>()
    const [sLabeledImgs, setLabeledImgs] = useState<ILabeledImageSet[]>()
    const [sImgUid, genImgUid] = useReducer((x: number) => x + 1, 0)
    const [sLabelsMap, setLabelsMap] = useState<ILabelMap>()
    const [sTrainSet, setTrainSet] = useState<tf.TensorContainerObject>()

    // Model
    const [sDenseUnits, setDenseUnits] = useState<number>(100)
    const [sOutputClasses, setOutputClasses] = useState<number>(4)

    // Train
    const [sEpochs, setEpochs] = useState<number>(10)
    const [sBatchSize, setBatchSize] = useState<number>(128)
    const [sLearningRate, setLearningRate] = useState<number>(0.001)

    // Predict
    const [sPredictResult, setPredictResult] = useState<tf.Tensor>()

    const webcamRef = useRef<IWebCameraHandler>(null)
    const webcamRef2 = useRef<IWebCameraHandler>(null)
    const historyRef = useRef<HTMLDivElement>(null)
    const downloadRef = useRef<HTMLAnchorElement>(null)

    const [formModel] = Form.useForm()
    const [formTrain] = Form.useForm()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init truncatedModel model ...')

        setStatus(STATUS.WAITING)

        tf.backend()
        setTfBackend(tf.getBackend())

        let truncatedModel: tf.LayersModel
        createTruncatedMobileNet().then(
            (result) => {
                truncatedModel = result
                setTruncatedModel(result)
                setStatus(STATUS.LOADED)
            },
            (e) => {
                // eslint-disable-next-line @typescript-eslint/no-floating-promises
                message.error(e.message)
            }
        )

        return () => {
            logger('TruncatedModel Dispose')
            truncatedModel?.dispose()
        }
    }, [])

    useEffect(() => {
        if (!sUploadingJson) {
            return
        }

        const timer = setInterval(async (): Promise<void> => {
            logger('Waiting upload...')
            if (checkUploadDone([sUploadingJson]) > 0) {
                forceWaitingPush()
            } else {
                clearInterval(timer)

                const buffer = await getUploadFileArray(sUploadingJson.originFileObj)
                const fileJson: ILabeledImageFileJson = JSON.parse(buffer.toString())
                const decoded = decodeImageTensor(fileJson.labeledImageSetList)
                setLabeledImgs(decoded)
                setUploadingJson(undefined)
            }
        }, 10)

        return () => {
            clearInterval(timer)
        }
    }, [waitingPush])

    useEffect(() => {
        if (!sLabeledImgs || !sTruncatedModel) {
            return
        }
        logger('init data set ...', sLabeledImgs)

        const outputClasses = sLabeledImgs.length
        setOutputClasses(outputClasses)

        const labelsArray = sLabeledImgs.map((labeled) => labeled.label)
        const labelsMap: ILabelMap = {}
        labelsArray.forEach((item, index) => {
            labelsMap[index] = item
        })
        setLabelsMap(labelsMap)

        const dataHandler = new TransferDataset(outputClasses)
        dataHandler.addExamples(sTruncatedModel, sLabeledImgs)
        setTrainSet(() => dataHandler.getData()) // when use sTrainSet, will get last records

        return () => {
            logger('Data Dispose')
            dataHandler.dispose()
        }
    }, [sLabeledImgs])

    useEffect(() => {
        logger('init model ...')
        if (!sTruncatedModel) {
            return
        }

        setStatus(STATUS.WAITING)

        const _model = createModel(sTruncatedModel, sOutputClasses, sDenseUnits)
        setModel(_model)

        const _layerOptions: ILayerSelectOption[] = _model?.layers.map((l, index) => {
            return { name: l.name, index }
        })
        setLayersOption(_layerOptions)

        setStatus(STATUS.LOADED)

        return () => {
            logger('Model Dispose')
            _model?.dispose()
        }
    }, [sTruncatedModel, sOutputClasses, sDenseUnits])

    useEffect(() => {
        logger('compile model ...')
        if (!sModel) {
            return
        }

        // Creates the optimizers which drives training of the model.
        const optimizer = tf.train.adam(sLearningRate)
        // We use categoricalCrossentropy which is the loss function we use for
        // categorical classification which measures the error between our predicted
        // probability distribution over classes (probability that an input is of each
        // class), versus the label (100% probability in the true class)>
        sModel.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' })

        return () => {
            logger('Model Dispose')
            optimizer?.dispose()
        }
    }, [sModel, sLearningRate])

    /***********************
     * Functions
     ***********************/

    const train = (_trainSet: tf.TensorContainerObject): void => {
        logger('train', _trainSet)
        if (!sModel) {
            return
        }

        setStatus(STATUS.WAITING)

        // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
        sModel.fit(_trainSet.xs as tf.Tensor, _trainSet.ys as tf.Tensor, {
            epochs: sEpochs,
            batchSize: sBatchSize,
            callbacks: tfvis.show.fitCallbacks(historyRef?.current, ['loss', 'acc', 'val_loss', 'val_acc'])
        }).then(
            () => {
                setStatus(STATUS.TRAINED)
            },
            loggerError
        )
    }

    const handleTrain = (): void => {
        sTrainSet && train(sTrainSet)
    }

    const handlePredict = (imgTensor: tf.Tensor): void => {
        if (!imgTensor) {
            return
        }
        setStatus(STATUS.WAITING)
        // logger('handlePredict', imgTensor)
        const [imgPred] = tf.tidy(() => {
            const batched = formatImageForMobileNet(imgTensor)
            const embeddings = sTruncatedModel?.predict(batched)
            const result = sModel?.predict(embeddings as tf.Tensor) as tf.Tensor
            const imgPred = result.argMax(-1)
            return [imgPred]
        })
        logger('Predict', imgPred)
        setStatus(STATUS.PREDICTED)
        setPredictResult(imgPred)
    }

    const handleLayerChange = (value: number): void => {
        logger('handleLayerChange', value)
        const _layer = sModel?.getLayer(undefined, value)
        setCurLayer(_layer)
    }

    const handleLabeledImagesSubmit = (value: ILabeledImageFileJson): void => {
        logger('handleLabeledImagesSubmit', value)

        const labeledImageSetList = value.labeledImageSetList
        setLabeledImgs(labeledImageSetList)
        setStatus(STATUS.LOADED)
    }

    const handleLabeledCapture = async (label: string): Promise<ILabeledImage | void> => {
        logger('handleLabeldCapture')
        if (webcamRef.current) {
            // eslint-disable-next-line @typescript-eslint/no-floating-promises
            const result = await webcamRef.current.capture()
            if (result) {
                genImgUid()
                const file: ILabeledImage = {
                    uid: sImgUid.toString(),
                    name: `${label}_${sImgUid.toString()}`,
                    tensor: result
                }
                return file
            }
        }
    }

    const handleModelParamsChange = (): void => {
        const values = formModel.getFieldsValue()
        // logger('handleTrainParamsChange', value)
        const { denseUnits } = values
        setDenseUnits(denseUnits)
    }

    const handleTrainParamsChange = (): void => {
        const values = formTrain.getFieldsValue()
        // logger('handleTrainParamsChange', value)
        const { learningRate, epochs, batchSize } = values
        setLearningRate(learningRate)
        setBatchSize(batchSize)
        setEpochs(epochs)
    }

    const handleJsonSave = (): void => {
        if (!sLabeledImgs) {
            return
        }

        const fileJson: ILabeledImageFileJson = { labeledImageSetList: encodeImageTensor(sLabeledImgs) }
        const a = downloadRef.current
        if (a) {
            const blob = new Blob(
                [JSON.stringify(fileJson, null, 2)],
                { type: 'application/json' })
            const blobUrl = window.URL.createObjectURL(blob)
            logger(blobUrl)

            // logger(a)
            const filename = 'labeledImages.json'
            a.href = blobUrl
            a.download = filename
            a.click()
            window.URL.revokeObjectURL(blobUrl)
        }
    }

    const handleJsonChange = ({ file }: UploadChangeParam): void => {
        logger('handleFileChange', file.name)

        setUploadingJson(file)
        forceWaitingPush()
    }

    const handleUpload = async (file: RcFile): Promise<string> => {
        setStatus(STATUS.WAITING)
        // logger(file)
        return getUploadFileBase64(file)
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    /***********************
     * Render
     ***********************/

    const _tensorX = sTrainSet?.xs as tf.Tensor4D
    const _tensorY = sTrainSet?.ys as tf.Tensor

    const dataTrainSetCard = (): JSX.Element => {
        return <Card title='MobileNet Transfer Learning Train Set' style={{ margin: '8px' }} size='small'>
            <div className='centerContainer'>
                <Upload onChange={handleJsonChange} action={handleUpload} showUploadList={false}>
                    <Button style={{ width: '300', margin: '0 10%' }}>
                        <UploadOutlined/> Load Data Set
                    </Button>
                </Upload>
                <Button style={{ width: '300', margin: '0 10%' }} onClick={handleJsonSave}>
                    <SaveOutlined/> Save Data Set
                </Button>
            </div>
            <div> Status: {sStatus}</div>
            <div> XShape: {_tensorX?.shape.join(',')}, YShape: {_tensorY?.shape.join(',')}</div>
            <a ref={downloadRef}/>

            {sLabeledImgs?.map((labeled, index) => {
                const title = `${labeled.label}(${labeled.imageList?.length.toString()})`
                return <Card key={index} title={title} style={{ margin: '8px' }} size='small'>
                    {
                        labeled.imageList?.map((imgItem: ILabeledImage) => {
                            if (imgItem.tensor) {
                                return <TensorImageThumbWidget key={imgItem.uid} data={imgItem.tensor}
                                    height={IMAGE_HEIGHT}/>
                            } else if (imgItem.img) {
                                return <img key={imgItem.uid} src={imgItem.img} alt={imgItem.name}
                                    height={IMAGE_HEIGHT} style={{ margin: 4 }}/>
                            } else {
                                return <></>
                            }
                        })
                    }
                </Card>
            })}
        </Card>
    }

    const modelAdjustCard = (): JSX.Element => {
        return (
            <Card title='Adjust Dense Net' style={{ margin: '8px' }} size='small'>
                <Form {...layout} form={formModel} onFinish={handleTrain} onFieldsChange={handleModelParamsChange}
                    initialValues={{
                        denseUnits: 100
                    }}>
                    <Form.Item name='denseUnits' label='Dense Units'>
                        <Select>
                            {DENSE_UNITS.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item label='Output Classes'>
                        <div>{sOutputClasses}</div>
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
                        learningRate: 0.0001,
                        batchSize: 32,
                        epochs: 30
                    }}>
                    <Form.Item name='epochs' label='Epochs'>
                        <Slider min={10} max={50} step={10} marks={{ 10: 10, 30: 30, 50: 50 }}/>
                    </Form.Item>
                    <Form.Item name='batchSize' label='Batch Size'>
                        <Select>
                            {BATCH_SIZES.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item name='learningRate' label='Learning Rate'>
                        <Select>
                            {LEARNING_RATES.map((v) => {
                                return <Option key={v} value={v}>{v}</Option>
                            })}
                        </Select>
                    </Form.Item>
                    <Form.Item {...tailLayout}>
                        <Button type='primary' htmlType={'submit'}
                            style={{ width: '60%', margin: '0 20%' }}> Train </Button>
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
        <AIProcessTabs title={'MobileNet Transfer Learning'} current={sTabCurrent} onChange={handleTabChange}>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/ai/mobilenet.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={12}>
                        <Card title='Captures Label Panel' size='small' style={{ margin: '8px' }}>
                            <WebCamera ref={webcamRef} config={MOBLILENET_CONFIG}/>
                            <LabeledCaptureInputSet onSave={handleLabeledImagesSubmit}
                                onCapture={handleLabeledCapture}/>
                            <div> XShape: {_tensorX?.shape.join(',')}, YShape: {_tensorY?.shape.join(',')}</div>
                        </Card>
                    </Col>
                    <Col span={12}>
                        {dataTrainSetCard()}
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Row>
                    <Col span={12}>
                        <Card title='MobileNet' style={{ margin: '8px' }} size='small'>
                            <h3 className='centerContainer'>预训练 MobileNet 模型 : {sStatus}</h3>
                            <h3 className='centerContainer'>截取 MobileNet 到 conv_pw_13_relu 层</h3>
                            <Card title='Expand Dense Net' style={{ margin: '8px' }} size='small'>
                                <TfvisModelWidget model={sTruncatedModel}/>
                            </Card>
                        </Card>
                    </Col>
                    <Col span={12}>
                        {modelAdjustCard()}
                        <Card title='+ Expand Dense Net' style={{ margin: '8px' }} size='small'>
                            <h3 className='centerContainer'>将 MobileNet 的 conv_pw_13_relu 层输出</h3>
                            <h3 className='centerContainer'>输入到 Dense 网络中</h3>
                            <Card title='Expand Dense Net' style={{ margin: '8px' }} size='small'>
                                <TfvisModelWidget model={sModel}/>
                                <p>status: {sStatus}</p>
                                <p>backend: {sTfBackend}</p>
                            </Card>
                            <Card title='Layers of expand Dense Net' style={{ margin: '8px' }} size='small'>
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
                        {dataTrainSetCard()}
                    </Col>
                    <Col span={8}>
                        {trainAdjustCard()}
                        {modelAdjustCard()}
                    </Col>
                    <Col span={8}>
                        <Card title='Training History' style={{ margin: '8px' }} size='small'>
                            <div ref={historyRef}/>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.PREDICT}>
                <Row>
                    <Col span={12}>
                        <Card title='Prediction with picture' style={{ margin: '8px' }} size='small'>
                            <ImageUploadWidget onSubmit={handlePredict} prediction={sPredictResult}
                                labelsMap={sLabelsMap}/>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Prediction with camera' style={{ margin: '8px' }} size='small'>
                            <WebCamera ref={webcamRef2} onSubmit={handlePredict} prediction={sPredictResult}
                                labelsMap={sLabelsMap} isPreview/>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
        </AIProcessTabs>
    )
}

export default MobileNetTransfer
