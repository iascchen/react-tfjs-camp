import React, { useEffect, useReducer, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as knnClassifier from '@tensorflow-models/knn-classifier'
import { Button, Card, Col, message, Row, Tabs, Upload } from 'antd'
import { SaveOutlined, UploadOutlined } from '@ant-design/icons'

import {
    checkUploadDone,
    getImageDataFromBase64, getUploadFileArray, getUploadFileBase64,
    IKnnPredictResult, ILabeledImage,
    ILabeledImageFileJson,
    ILabeledImageSet,
    logger, loggerError,
    STATUS
} from '../../utils'
import ImageUploadWidget from '../common/tensor/ImageUploadWidget'
import LabeledImageInputSet from '../common/tensor/LabeledImageInputSet'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import MarkdownWidget from '../common/MarkdownWidget'
import WebCamera, { IWebCameraHandler } from '../common/tensor/WebCamera'

import { formatImageForMobileNet, MOBILENET_IMAGE_SIZE, MOBILENET_MODEL_PATH } from './mobilenetUtils'
import { RcFile, UploadChangeParam } from 'antd/es/upload'
import { UploadFile } from 'antd/es/upload/interface'
import TfvisLayerWidget from '../common/tfvis/TfvisLayerWidget'
import VectorWidget from '../common/tensor/VectorWidget'

const { TabPane } = Tabs

const KNN_TOPK = 10
const IMAGE_HEIGHT = 100

const MobileNetKnnClassifier = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sTabCurrent, setTabCurrent] = useState<number>(2)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sKnn, setKnn] = useState<knnClassifier.KNNClassifier>()
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sCurLayer, setCurLayer] = useState<tf.layers.Layer>()

    const [sLabeledImgs, setLabeledImgs] = useState<ILabeledImageSet[]>()
    const [sPredictResult, setPredictResult] = useState<IKnnPredictResult>()
    const [sPredictFeature, setPredictFeature] = useState<tf.Tensor>()

    const [sUploadingJson, setUploadingJson] = useState<UploadFile>()
    const [waitingPush, forceWaitingPush] = useReducer((x: number) => x + 1, 0)

    const downloadRef = useRef<HTMLAnchorElement>(null)
    const webcamRef = useRef<IWebCameraHandler>(null)

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())

        setStatus(STATUS.WAITING)

        const knn = knnClassifier.create()
        setKnn(knn)

        let model: tf.LayersModel
        tf.loadLayersModel(MOBILENET_MODEL_PATH).then(
            (mobilenet) => {
                // Return a model that outputs an internal activation.
                const layer = mobilenet.getLayer('conv_preds')
                model = tf.model({ inputs: mobilenet.inputs, outputs: layer.output })

                // Warmup the model. This isn't necessary, but makes the first prediction
                // faster. Call `dispose` to release the WebGL memory allocated for the return
                // value of `predict`.
                const _temp = model.predict(tf.zeros([1, MOBILENET_IMAGE_SIZE, MOBILENET_IMAGE_SIZE, 3])) as tf.Tensor
                _temp.dispose()

                setModel(model)
                setCurLayer(layer)
                setStatus(STATUS.LOADED)
            },
            loggerError
        )

        return () => {
            logger('Model Dispose')
            model?.dispose()
            knn.dispose()
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
                setLabeledImgs(fileJson.labeledImageSetList)
                setUploadingJson(undefined)
            }
        }, 10)

        return () => {
            clearInterval(timer)
        }
    }, [waitingPush])

    /***********************
     * Functions
     ***********************/

    const train = async (imageSetList: ILabeledImageSet[]): Promise<void> => {
        logger('train', imageSetList)

        for (const imgSet of imageSetList) {
            const { label, imageList } = imgSet
            if (imageList) {
                for (const imgItem of imageList) {
                    const imgBase64 = imgItem.img
                    if (imgBase64) {
                        const _imgData = await getImageDataFromBase64(imgBase64)
                        const _imgTensor = tf.browser.fromPixels(_imgData, 3)
                        const _imgBatched = formatImageForMobileNet(_imgTensor)
                        const _imgFeature = sModel?.predict(_imgBatched) as tf.Tensor

                        // logger('sKnn.addExample', label, _imgFeature)
                        sKnn?.addExample(_imgFeature, label)
                    }
                }
            }
        }
    }

    const resetKnn = (): void => {
        sKnn?.clearAllClasses()
    }

    const handleTrain = (): void => {
        if (sLabeledImgs) {
            setStatus(STATUS.WAITING)
            train(sLabeledImgs).then(
                () => {
                    setStatus(STATUS.TRAINED)
                },
                (error) => {
                    logger(error)
                })
        }
    }

    const handleKnnReset = (): void => {
        resetKnn()
    }

    const handlePredict = async (imgTensor: tf.Tensor): Promise<void> => {
        if (!imgTensor) {
            return
        }
        const [imgFeature] = tf.tidy(() => {
            const batched = formatImageForMobileNet(imgTensor)
            const imgFeature = sModel?.predict(batched) as tf.Tensor
            // logger(imgFeature)
            return [imgFeature]
        })
        try {
            const res = await sKnn?.predictClass(imgFeature, KNN_TOPK)
            // logger('handlePredict', res)
            setPredictResult(res)
            setPredictFeature(imgFeature)
        } catch (e) {
            // logger(e)
            await message.error(e.message)
        }
    }

    const handleLabeledImagesSubmit = (value: ILabeledImageFileJson): void => {
        logger('handleLabeledImagesSubmit', value)

        const labeledImageSetList = value.labeledImageSetList
        setLabeledImgs(labeledImageSetList)
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    const handleJsonSave = (): void => {
        if (!sLabeledImgs) {
            return
        }

        const fileJson: ILabeledImageFileJson = { labeledImageSetList: sLabeledImgs }
        const a = downloadRef.current
        if (a) {
            const blob = new Blob([JSON.stringify(fileJson, null, 2)],
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
        // logger(file)
        return getUploadFileBase64(file)
    }

    /***********************
     * Render
     ***********************/

    const knnInfo = (): JSX.Element => {
        const knnNumClasses = sKnn?.getNumClasses() ?? 0
        const examples = sKnn?.getClassExampleCount()
        return (<div className='centerContainer'>
            { sKnn && knnNumClasses > 0
                ? <div>
                    <div> KNN have {knnNumClasses} classes. </div>
                    <div> {JSON.stringify(examples)} </div>
                    <div> <VectorWidget data={sKnn.getClassifierDataset()} /> </div>
                </div>
                : <div>
                    <p>KNN not trained</p>
                </div>}
        </div>)
    }

    const dataTrainSetCard = (): JSX.Element => {
        return <Card title='MobileNet + KNN Train Set' style={{ margin: '8px' }} size='small'>
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
            <a ref={downloadRef}/>

            {sLabeledImgs?.map((labeled, index) => {
                const title = `${labeled.label}(${labeled.imageList?.length.toString()})`
                return <Card key={index} title={title} style={{ margin: '8px' }} size='small'>
                    {
                        labeled.imageList?.map((imgItem: ILabeledImage) => {
                            return <img key={imgItem.uid} src={imgItem.img} alt={imgItem.name}
                                height={IMAGE_HEIGHT} style={{ margin: 4 }}/>
                        })
                    }
                </Card>
            })}
        </Card>
    }

    return (
        <AIProcessTabs title={'MobileNet + KNN'} current={sTabCurrent} onChange={handleTabChange}>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/ai/mobilenet-knn.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={12}>
                        <Card title='Images Label Panel' style={{ margin: '8px' }} size='small'>
                            <LabeledImageInputSet onSave={handleLabeledImagesSubmit} />
                        </Card>
                    </Col>
                    <Col span={12}>
                        {dataTrainSetCard()}
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Col span={12} offset={6}>
                    <Card title='MobileNet' style={{ margin: '8px' }} size='small'>
                        <h3 className='centerContainer'>预训练 MobileNet 模型 : {sStatus}</h3>
                        <h3 className='centerContainer'>MobileNet 的 conv_preds 层输出</h3>
                        <Card title='conv_preds Layer Info' style={{ margin: '8px' }} size='small'>
                            <TfvisLayerWidget layer={sCurLayer}/>
                        </Card>
                        <h3 className='centerContainer'>作为 KNN 的样本输入</h3>
                        <Card title='KNN 样本库' style={{ margin: '8px' }} size='small'>
                            {knnInfo()}
                        </Card>
                    </Card>
                </Col>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                <Row>
                    <Col span={12}>
                        {dataTrainSetCard()}
                    </Col>
                    <Col span={12}>
                        <Card title='MobileNet + KNN Train Set' style={{ margin: '8px' }} size='small'>
                            <Button onClick={handleTrain} type='primary' style={{ width: '30%', margin: '0 10%' }}>
                                Train
                            </Button>
                            <Button onClick={handleKnnReset} style={{ width: '30%', margin: '0 10%' }}> Reset
                                Model </Button>
                            {knnInfo()}
                            <p>backend: {sTfBackend}</p>
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.PREDICT}>
                <Row>
                    <Col span={12}>
                        <Card title='Prediction with picture' style={{ margin: '8px' }} size='small'>
                            <ImageUploadWidget onSubmit={handlePredict} prediction={sPredictResult}/>
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Prediction with camera' style={{ margin: '8px' }} size='small'>
                            <WebCamera ref={webcamRef} onSubmit={handlePredict} prediction={sPredictResult} isPreview/>
                        </Card>
                    </Col>
                    <Col span={24}>
                        {sKnn && <VectorWidget data={sKnn.getClassifierDataset()} predFeature={sPredictFeature}/>}
                    </Col>
                </Row>
            </TabPane>
        </AIProcessTabs>
    )
}

export default MobileNetKnnClassifier
