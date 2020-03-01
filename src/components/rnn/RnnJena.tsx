import React, { useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Col, Row, Tabs } from 'antd'

import {
    arrayDispose,
    getImageDataFromBase64,
    IKnnPredictResult,
    ILabeledImageFileJson,
    ILabeledImageSet,
    logger,
    STATUS
} from '../../utils'
import { MOBILENET_IMAGE_SIZE } from '../../constant'
import ImageUploadWidget from '../common/tensor/ImageUploadWidget'
import LabeledImageInputSet from '../common/tensor/LabeledImageInputSet'
import LabeledImageSetWidget from '../common/tensor/LabeledImageSetWidget'
import AIProcessTabs, { AIProcessTabPanes } from '../common/AIProcessTabs'
import TfvisModelWidget from '../common/tfvis/TfvisModelWidget'
import MarkdownWidget from '../common/MarkdownWidget'
import JenaModelWidget from './JenaModelWidget'

const { TabPane } = Tabs

const RnnJena = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTabCurrent, setTabCurrent] = useState<number>(3)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sNumFeatures, setNumFeatures] = useState(10)
    const [sModel, setModel] = useState<tf.LayersModel>()

    const [sLabeledImgs, setLabeledImgs] = useState<ILabeledImageSet[]>()
    const [sPredictResult, setPredictResult] = useState<tf.Tensor | IKnnPredictResult >()

    /***********************
     * useEffect
     ***********************/

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
                        // const _imgBatched = formatImageForMobilenet(_imgTensor, MOBILENET_IMAGE_SIZE)
                        // const _imgFeature = sModel?.predict(_imgBatched) as tf.Tensor
                    }
                }
            }
        }
    }

    const handleTrain = (): void => {
        if (sLabeledImgs) {
            setStatus(STATUS.TRAINING)
            train(sLabeledImgs).then(
                () => {
                    setStatus(STATUS.TRAINED)
                },
                (error) => {
                    logger(error)
                })
        }
    }

    const handlePredict = async (imgTensor: tf.Tensor): Promise<void> => {
        if (!imgTensor) {

        }
    }

    const handleLabeledImagesSubmit = (value: ILabeledImageFileJson): void => {
        logger('handleLabeledImagesSubmit', value)

        const labeledImageSetList = value.labeledImageSetList
        setLabeledImgs(labeledImageSetList)
    }

    const handleLoadJson = (values: ILabeledImageSet[]): void => {
        sLabeledImgs && arrayDispose(sLabeledImgs)
        setLabeledImgs(values)
    }

    const handleModelChange = (model: tf.LayersModel): void => {
        model && setModel(model)
    }

    const handleTabChange = (current: number): void => {
        setTabCurrent(current)
    }

    /***********************
     * Render
     ***********************/

    return (
        <AIProcessTabs title={'RNN'} current={sTabCurrent} onChange={handleTabChange} docUrl={'/docs/rnnJena.md'}>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.INFO}>
                <MarkdownWidget url={'/docs/rnnJena.md'}/>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.DATA}>
                <Row>
                    <Col span={12}>
                        <Card title='Images Label Panel' style={{ margin: '8px' }} size='small'>
                            {sModel && <LabeledImageInputSet model={sModel} onSave={handleLabeledImagesSubmit} />}
                        </Card>
                    </Col>
                    <Col span={12}>
                        <Card title='Data Set' style={{ margin: '8px' }} size='small'>
                            {sModel && <LabeledImageSetWidget model={sModel} labeledImgs={sLabeledImgs}
                                onJsonLoad={handleLoadJson}/>}
                        </Card>
                    </Col>
                </Row>
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <Col>
                    <JenaModelWidget numFeatures={sNumFeatures} onChange={handleModelChange} />
                </Col>
                <Col>
                    {/*{sModel && <TfvisModelWidget model={sModel} />}*/}
                </Col>

            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                <Card title='Mobilenet + KNN Train Set' style={{ margin: '8px' }} size='small'>
                    <div>
                        <Button onClick={handleTrain} type='primary' style={{ width: '30%', margin: '0 10%' }}> Train </Button>
                        <div>status: {sStatus}</div>
                    </div>
                    <p>backend: {sTfBackend}</p>
                </Card>
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
