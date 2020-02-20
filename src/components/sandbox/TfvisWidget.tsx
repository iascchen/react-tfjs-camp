import React, { useEffect, useRef } from 'react'
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

    useEffect(() => {
        draw()
    }, [])

    const draw = (): void => {
        if (!canvasRef.current) {
            return
        }
        console.log(tfvis)

        const drawable = canvasRef.current as HTMLElement
        tfvis.show.history(drawable, logs, ['loss', 'val_loss']).then(
            () => {
                // ignore
                console.log('drawn')
            },
            (e: any) => {
                logger(e)
            })
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

            <div style={{ height: 400, width: 400 }} ref={canvasRef} ></div>
        </div>
    )
}

export default TfvisWidget
