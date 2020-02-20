import React, { useEffect, useRef } from 'react'

import { ILayer } from '../../../utils'

// cannot use import
// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

interface IProps {
    layer: ILayer

    debug?: boolean
}

const TfvisLayerWidget = (props: IProps): JSX.Element => {
    const elementRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (!props.layer) {
            return
        }
        tfvis.show.layer(elementRef.current, props.layer)
    }, [props.layer])

    return (
        <div ref={elementRef} />
    )
}

export default TfvisLayerWidget
