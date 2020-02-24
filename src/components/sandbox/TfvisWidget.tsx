import React, { useEffect, useRef } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button } from 'antd'
import { logger } from '../../utils'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

const data = [
    { index: 0, value: 50 },
    { index: 1, value: 100 },
    { index: 2, value: 150 }
]

const logs = {
    history: { loss: [1, 2], val_loss: [1.5, 2.5] }
}

const TfvisWidget = (): JSX.Element => {
    const canvasRef = useRef<HTMLDivElement>(null)
    const canvasRef2 = useRef<HTMLDivElement>(null)

    useEffect(() => {
        draw()
    }, [])

    const draw = (): void => {
        if (!canvasRef.current) {
            return
        }
        logger(tfvis)

        // const drawable = canvasRef.current as HTMLElement
        tfvis.show.history(canvasRef.current, logs, ['loss', 'val_loss'])

        const tensor = tf.tensor1d([0, 0, 0, 0, 2, 3, 4])

        const headers = [
            'DataSet',
            'Shape',
            'dType',
            'stride'
        ]

        const values = [
            ['xs', tensor.shape, tensor.dtype, tensor.strides, JSON.stringify(tensor)], // xs
            ['xs', tensor.shape, tensor.dtype, tensor.strides, JSON.stringify(tensor)] // xs
        ]

        tfvis.render.table(canvasRef2.current, { headers, values })

        tfvis.visor().surface({
            tab: 'My Tab',
            name: 'Custom Height',
            styles: {
                height: 500
            }
        })
        const suffer = tfvis.visor().surface({
            tab: 'My Tab2',
            name: 'Custom Height 2',
            styles: {
                height: 300
            }
        })
        tfvis.render.table(suffer, { headers, values })

        const data = [
            { index: 0, value: 50 },
            { index: 1, value: 100 },
            { index: 2, value: 150 }
        ]

        // Render to visor
        const surface2 = { name: 'Bar chart', tab: 'Charts' }
        tfvis.render.barchart(surface2, data)
    }

    const handleClick = (): void => {
        // Train the model using the data.
        draw()
    }

    return (
        <div>
            {JSON.stringify(data)}
            {JSON.stringify(logs)}
            <Button onClick={handleClick}>Draw</Button>

            <div style={{ height: 400, width: 400 }} ref={canvasRef} />

            <div style={{ height: 400, width: 400 }} ref={canvasRef2} />
        </div>
    )
}

export default TfvisWidget
