import React, { useEffect, useRef } from 'react'
import * as tf from '@tensorflow/tfjs'

// cannot use import
// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

const headers = ['DataSet', 'Shape', 'DType', 'Strides', 'Rank']

interface IProps {
    value: tf.TensorContainerObject

    debug?: boolean
}

const TfvisDatasetInfoWidget = (props: IProps): JSX.Element => {
    const elementRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (!props.value) {
            return
        }

        const xs = props.value.xs as tf.Tensor
        const ys = props.value.ys as tf.Tensor
        const _values = [
            ['xs', xs.shape, xs.dtype, xs.strides, xs.rank],
            ['ys', ys.shape, ys.dtype, ys.strides, ys.rank]
        ]
        tfvis.render.table(elementRef.current, { headers, values: _values })
    }, [props.value])

    return (
        <div ref={elementRef} />
    )
}

export default TfvisDatasetInfoWidget
