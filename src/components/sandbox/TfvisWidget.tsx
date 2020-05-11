import React, { useEffect, useRef } from 'react'
import * as tf from '@tensorflow/tfjs'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const tfvis = require('@tensorflow/tfjs-vis')

const headers = ['DataSet', 'Shape', 'dType', 'stride', 'json']
const tensor = tf.tensor1d([0, 0, 0, 0, 2, 3, 4])
const values = [
    ['xs', tensor.shape, tensor.dtype, tensor.strides, JSON.stringify(tensor)], // xs
    ['ys', tensor.shape, tensor.dtype, tensor.strides, JSON.stringify(tensor)] // ys
]

const data = [
    { index: 0, value: 50 },
    { index: 1, value: 100 },
    { index: 2, value: 150 }
]

const logs = {
    history: { loss: [1, 2, 1.5], val_loss: [1.5, 2.5, 2.8] }
}

const TfvisWidget = (): JSX.Element => {
    const canvasRef = useRef<HTMLDivElement>(null)
    const canvasRef2 = useRef<HTMLDivElement>(null)

    useEffect(() => {
        drawDiv1()
        drawDiv2()

        drawSurface1()
        drawSurface2()
    }, [])

    const drawDiv1 = (): void => {
        if (!canvasRef.current) {
            return
        }
        tfvis.show.history(canvasRef.current, logs, ['loss', 'val_loss'])
    }

    const drawDiv2 = (): void => {
        if (!canvasRef2.current) {
            return
        }
        tfvis.render.table(canvasRef2.current, { headers, values })
    }

    const drawSurface1 = (): void => {
        // Render to visor
        const surface2 = { name: 'Bar chart', tab: 'My Tab1' }
        tfvis.render.barchart(surface2, data)
    }

    const drawSurface2 = (): void => {
        const suffer = tfvis.visor().surface({
            tab: 'My Tab2',
            name: 'Custom Height 2',
            styles: {
                height: 300
            }
        })
        tfvis.render.table(suffer, { headers, values })
    }

    return (
        <div>
            <div style={{ height: 400, width: 400 }} ref={canvasRef} />
            <div style={{ height: 400, width: 400 }} ref={canvasRef2} />
        </div>
    )
}

export default TfvisWidget
