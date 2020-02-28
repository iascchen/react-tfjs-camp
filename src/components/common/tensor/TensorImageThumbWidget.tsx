import React, { useEffect, useRef } from 'react'
import * as tf from '@tensorflow/tfjs'

const DEFAULT_SIZE = 100

interface IProps {
    data: tf.Tensor3D
    height?: number
    width?: number
}

const TensorImageThumbWidget = (props: IProps): JSX.Element => {
    const rowCanvasRef = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        if (!props.data || !rowCanvasRef.current) {
            return
        }

        const width = props.width ?? DEFAULT_SIZE
        const height = props.height ?? DEFAULT_SIZE
        const sample = tf.image.resizeBilinear(props.data, [height, width])
        // eslint-disable-next-line @typescript-eslint/no-floating-promises
        tf.browser.toPixels(sample, rowCanvasRef.current).then()
    }, [props.data, props.width, props.height])

    return <canvas ref={rowCanvasRef} style={{ margin: 8 }}/>
}

export default TensorImageThumbWidget
