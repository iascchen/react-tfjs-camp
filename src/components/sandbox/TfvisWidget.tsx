import React, { useEffect, useRef } from 'react'
// import * as tf from '@tensorflow/tfjs'
// import tfvis from '@tensorflow/tfjs-vis'
import { Button } from 'antd'
import { logger } from '../../utils'

const data = [
    { index: 0, value: 50 },
    { index: 1, value: 100 },
    { index: 2, value: 150 }
]

const logs = {
    history: { loss: [1, 2], val_loss: [1.5, 2.5] }
}

const TfvisWidget = (): JSX.Element => {
    const canvasRef = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        draw()
    }, [])

    const draw = (): void => {
        if (!canvasRef.current) {
            return
        }
        logger('in draw')

        // const drawable = canvasRef.current as HTMLElement
        // tfvis.show.history(drawable, logs, ['loss', 'val_loss']).then(
        //     () => {
        //         // ignore
        //     },
        //     (e) => {
        //         logger(e)
        //     })
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

            <canvas ref={canvasRef} ></canvas>
        </div>
    )
}

export default TfvisWidget
