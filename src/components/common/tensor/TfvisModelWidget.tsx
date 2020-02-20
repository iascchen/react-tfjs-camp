import React, { useEffect, useRef } from 'react'

import { IModel } from '../../../utils'

// cannot use import
// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

interface IProps {
    model: IModel

    debug?: boolean
}

const TfvisModelWidget = (props: IProps): JSX.Element => {
    const elementRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (!props.model) {
            return
        }

        tfvis.show.modelSummary(elementRef.current, props.model)
    }, [props.model])

    return (
        <div ref={elementRef} />
    )
}

export default TfvisModelWidget
