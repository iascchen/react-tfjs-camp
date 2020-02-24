import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card } from 'antd'

import PicturesWall from '../../common/PicturesWall'
import { ImagenetClasses } from '../../mobilenet/imagenetClasses'
import { logger } from '../../../utils'

interface IProps {
    model?: tf.LayersModel
    prediction?: tf.Tensor
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

        const labelIndex = props.prediction.arraySync() as number
        logger('labelIndex', labelIndex)
        const _label = ImagenetClasses[labelIndex]
        setLabel(`${labelIndex.toString()} : ${_label}`)
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
        <Card>
            <PicturesWall onPreview={handlePreview}/>
            <div>Current Image</div>
            <img src={imgViewSrc} height={IMAGE_SIZE} ref={imageViewRef} />
            <div>
                <Button onClick={handleSubmit} type='primary'>Submit</Button>
                Prediction : {label}
            </div>
        </Card>
    )
}

export default ImageUploadWidget
