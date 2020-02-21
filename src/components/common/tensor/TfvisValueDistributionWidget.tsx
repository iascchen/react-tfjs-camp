import React, { useEffect, useRef } from 'react'

import { ITensor } from '../../../utils'

// cannot use import
// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

interface IProps {
    value: ITensor

    debug?: boolean
}

const TfvisValueDistributionWidget = (props: IProps): JSX.Element => {
    const elementRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (!props.value) {
            return
        }
        tfvis.show.valuesDistribution(elementRef.current, props.value)
    }, [props.value])

    return (
        <div ref={elementRef} />
    )
}

export default TfvisValueDistributionWidget
