import React, { useEffect, useRef } from 'react'
import * as tf from '@tensorflow/tfjs'
import { EyeOutlined, DeleteOutlined } from '@ant-design/icons'
import { logger, loggerError } from '../../../utils'

interface IProps {
    data: tf.Tensor3D
    uid?: string
    height?: number
    width?: number

    onPreview?: (data: tf.Tensor3D) => void
    onDelete?: (uid: string) => void
}

const TensorImageThumbWidget = (props: IProps): JSX.Element => {
    const rowCanvasRef = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        if (!props.data || !rowCanvasRef.current) {
            return
        }

        const [imgHeight, imgWidth] = props.data.shape
        // logger(imgHeight, imgWidth)
        const height = props.height
            ? props.height
            : (props.width ? Math.round(props.width / imgWidth * imgHeight) : imgHeight)
        const width = props.width
            ? props.width
            : (props.height ? Math.round(props.height / imgHeight * imgWidth) : imgWidth)
        // logger(width, height)

        const sample: tf.Tensor3D = tf.tidy(() => {
            // Normalize the image from [0, 255] to [0, 1].
            const image = props.data.toFloat().div(255)
            return tf.image.resizeBilinear(image as tf.Tensor3D, [height, width])
        })

        tf.browser.toPixels(sample, rowCanvasRef.current).then(
            () => {
                sample.dispose()
            },
            loggerError
        )
    }, [props.data, props.width, props.height])

    const handlePreview = (): void => {
        props.onPreview && props.onPreview(props.data)
    }

    const handleDelete = (): void => {
        props.onDelete && props.uid && props.onDelete(props.uid)
    }

    return <>
        <span style={{ padding: '8px', border: '1px solid lightgray', borderRadius: '2px' }} >
            <canvas ref={rowCanvasRef} />
        </span>
        {props.uid && <EyeOutlined onClick={handlePreview}/>}
        {props.uid && <DeleteOutlined onClick={handleDelete}/>}
    </>
}

export default TensorImageThumbWidget
