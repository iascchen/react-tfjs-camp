import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Row } from 'antd'

import PicturesWall from '../../common/PicturesWall'
import { ImageNetClasses } from '../../mobilenet/ImageNetClasses'
import { IKnnPredictResult, ILabelMap, logger } from '../../../utils'

interface IProps {
    prediction?: tf.Tensor | IKnnPredictResult
    labelsMap?: ILabelMap
    onSubmit?: (tensor: tf.Tensor) => void
}

const IMAGE_HEIGHT = 360

const ImageUploadWidget = (props: IProps): JSX.Element => {
    const [sImgViewSrc, setImgViewSrc] = useState<string>('/images/cat.jpg')
    const [sLabel, setLabel] = useState<string>()

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
            // ImageNet Classes
            const imagenetPred = props.prediction as tf.Tensor
            const labelIndex = imagenetPred.arraySync() as number
            logger('labelIndex', labelIndex)
            const _label = props.labelsMap ? props.labelsMap[labelIndex] : ImageNetClasses[labelIndex]
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
            <Row className='centerContainer'>
                <img src={sImgViewSrc} height={IMAGE_HEIGHT} ref={imageViewRef} />
            </Row>
            <Row className='centerContainer'>
                <Button onClick={handleSubmit} type='primary' style={{ width: '30%', margin: '8px' }}>Predict</Button>
            </Row>
            <Row className='centerContainer' >
                {sLabel && (
                    <span>{sLabel}</span>
                )}
            </Row>
            <PicturesWall onPreview={handlePreview} />
        </>
    )
}

export default ImageUploadWidget
