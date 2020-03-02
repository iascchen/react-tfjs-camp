import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, message } from 'antd'

import { logger, STATUS } from '../../../utils'

import { JenaWeatherData } from './dataJena'
import { trainModel } from './modelJena'

// cannot use import
// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

interface IProps{
    model?: tf.LayersModel
    data?: JenaWeatherData

    // onChange?: (model: JenaWeatherData) => void
}

const JenaTrainWidget = (props: IProps): JSX.Element => {
    /***********************
     * useState
     ***********************/
    const [sTfBackend, setTfBackend] = useState<string>()
    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)

    const [sModel, setModel] = useState<tf.LayersModel>()
    const [sDataHandler, setDataHandler] = useState<JenaWeatherData>()

    const [sEpochs, setEpochs] = useState<number>(10)

    const elementRef = useRef<HTMLDivElement>(null)

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        logger('init model ...')

        tf.backend()
        setTfBackend(tf.getBackend())
    }, [])

    useEffect(() => {
        props.data && setDataHandler(props.data)
    }, [props.data])

    useEffect(() => {
        props.model && setModel(props.model)
    }, [props.model])

    const surface = tfvis.visor().surface({ tab: 'Training', name: 'Model Training' })

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
            tfvis.show.fitCallbacks(surface, ['loss', 'val_loss'], {
                callbacks: ['onBatchEnd', 'onEpochEnd']
            }))
        setStatus(STATUS.TRAINED)
    }

    const handleTrain = (): void => {
        train().then()
    }

    /***********************
     * Render
     ***********************/

    // const disable = (sStatus === STATUS.TRAINING)
    return (
        <div>
            <Card title='Jena Weather' style={{ margin: '8px' }} size='small'>
                <div>
                    <Button onClick={handleTrain} type='primary' style={{ width: '30%', margin: '0 10%' }}> Train </Button>
                    <div>status: {sStatus}</div>
                </div>
                <p>backend: {sTfBackend}</p>
            </Card>
            <div ref={elementRef} />
        </div>
    )
}

export default JenaTrainWidget
