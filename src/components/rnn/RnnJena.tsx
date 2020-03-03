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
import JenaModelWidget from './jena/JenaModelWidget'
import JenaDataWidget from './jena/JenaDataWidget'
import JenaTrainWidget from './jena/JenaTrainWidget'
import { JenaWeatherData } from './jena/dataJena'

const { TabPane } = Tabs

const RnnJena = (): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTabCurrent, setTabCurrent] = useState<number>(1)

    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sNumFeatures, setNumFeatures] = useState(10)
    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sDataHandler, setDataHandler] = useState<JenaWeatherData>()

    const [sLabeledImgs, setLabeledImgs] = useState<ILabeledImageSet[]>()
    const [sPredictResult, setPredictResult] = useState<tf.Tensor | IKnnPredictResult >()

    /***********************
     * useEffect
     ***********************/

    /***********************
     * Functions
     ***********************/



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

    const handleDataChange = (dataHandler: JenaWeatherData): void => {
        setDataHandler(dataHandler)

        const numFeatures = dataHandler.getDataColumnNames().length
        setNumFeatures(numFeatures)
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
                <JenaDataWidget onChange={handleDataChange} />
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.MODEL}>
                <JenaModelWidget onChange={handleModelChange} />
            </TabPane>
            <TabPane tab='&nbsp;' key={AIProcessTabPanes.TRAIN}>
                <JenaTrainWidget model={sModel} data={sDataHandler} />
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
