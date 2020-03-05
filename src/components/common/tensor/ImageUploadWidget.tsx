import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button } from 'antd'

import PicturesWall from '../../common/PicturesWall'
import { ImagenetClasses } from '../../mobilenet/ImagenetClasses'
import { IKnnPredictResult, ILabelMap, logger } from '../../../utils'

interface IProps {
    model?: tf.LayersModel
    prediction?: tf.Tensor | IKnnPredictResult
    labelsMap?: ILabelMap
    onSubmit?: (tensor: tf.Tensor) => void
}

const IMAGE_SIZE = 224

const ImageUploadWidget = (props: IProps): JSX.Element => {
    const [imgViewSrc, setImgViewSrc] = useState<string>('/images/cat.jpg')
    const [label, setLabel] = useState()

    const imageViewRef = useRef<HTMLImageElement>(null)

    useEffect(() => {
        if (!props.prediction) {
            return
        }

        const knnRet = props.prediction as IKnnPredictResult
        if (knnRet.label) {
            const knnRet = props.prediction as IKnnPredictResult
            setLabel(`${knnRet.label} : ${JSON.stringify(knnRet.confidences)}`)
        } else {
            // Imagenet Classes
            const imagenetRet = props.prediction as tf.Tensor
            const labelIndex = imagenetRet.arraySync() as number
            logger('labelIndex', labelIndex)
            const _label = props.labelsMap ? props.labelsMap[labelIndex] : ImagenetClasses[labelIndex]
            setLabel(`${labelIndex.toString()} : ${_label}`)
        }
    }, [props.prediction])

    const handlePreview = (file: string): void => {
        // logger('handlePreview', file)
        setImgViewSrc(file)
    }

    const handleSubmit = (): void => {
        if (!imageViewRef.current) {
            return
        }
        const _tensor = tf.browser.fromPixels(imageViewRef.current).toFloat()
        props.onSubmit && props.onSubmit(_tensor)
    }

    /***********************
     * Render
     ***********************/

    return (
        <>
            <PicturesWall onPreview={handlePreview} />
            <div>Current Image</div>
            <div><img src={imgViewSrc} height={IMAGE_SIZE} ref={imageViewRef} /></div>
            <div>
                <Button onClick={handleSubmit} type='primary' style={{ width: '30%', margin: '0 10%' }}>
                    Predict
                </Button>
            </div>
            Prediction Result : {label}
        </>
    )
}

export default ImageUploadWidget
