import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card } from 'antd'
import { logger } from '../../utils'
import RowImageWidget from '../common/tensor/RowImageWidget'

const CANVAS_WIDTH = 280
const CANVAS_HEIGHT = 280
const MNIST_SHAPE = [28, 28]

interface IPoint {
    x: number
    y: number
}

interface IProps {
    model?: tf.LayersModel
    prediction?: tf.Tensor
    onChange?: (tensor: tf.Tensor) => void
}

const DrawPanelWidget = (props: IProps): JSX.Element => {
    const [miniSample, setMiniSample] = useState<number[]>()

    const [drawing, setDrawing] = useState(false)
    const [currPos, setCurrPos] = useState<IPoint>({ x: 0, y: 0 })

    const panelRef = useRef<HTMLCanvasElement>(null)
    const currPosRef = useRef<IPoint | undefined>(currPos)

    useEffect(() => {
        const _canvas = panelRef.current
        if (!_canvas) {
            return
        }
        logger('canvasRef init')

        // Note: this implementation is a bit simplified
        _canvas?.addEventListener('mousemove', handleWindowMouseMove)
        _canvas?.addEventListener('mousedown', handleWindowMouseDown)
        _canvas?.addEventListener('mouseup', handleWindowMouseup)
        _canvas?.addEventListener('mouseleave', handleWindowMouseup)

        // _canvas?.addEventListener('touchmove', handleWindowMouseMove)
        // _canvas?.addEventListener('touchstart', handleWindowMouseDown)
        // _canvas?.addEventListener('touchend', handleWindowMouseup)

        return () => {
            logger('Dispose canvasRef')
            _canvas?.removeEventListener('mousemove', handleWindowMouseMove)
            _canvas?.removeEventListener('mousedown', handleWindowMouseDown)
            _canvas?.removeEventListener('mouseup', handleWindowMouseup)
            _canvas?.removeEventListener('mouseleave', handleWindowMouseup)

            // _canvas?.removeEventListener('touchmove', handleWindowMouseMove)
            // _canvas?.removeEventListener('touchstart', handleWindowMouseDown)
            // _canvas?.removeEventListener('touchend', handleWindowMouseup)
        }
    }, [panelRef])

    useEffect(() => {
        currPosRef.current = currPos
    })

    const draw = (from: IPoint | undefined, to: IPoint): void => {
        const _canvas = panelRef.current
        const _ctx = _canvas?.getContext('2d')

        if (!_ctx || !currPos || !from) {
            return
        }

        _ctx.beginPath()
        _ctx.moveTo(from.x, from.y)
        _ctx.lineTo(to.x, to.y)
        _ctx?.closePath()

        _ctx.strokeStyle = 'white'
        _ctx.lineWidth = 20
        _ctx.stroke()
    }

    const getMousePos = (e: MouseEvent): IPoint | null => {
        const _canvas = panelRef.current
        const bbox = _canvas?.getBoundingClientRect()
        return bbox ? {
            x: e.clientX - bbox?.left,
            y: e.clientY - bbox?.top
        } : null
    }

    const handleWindowMouseMove = (e: MouseEvent): void => {
        const _pos = getMousePos(e)
        _pos && setCurrPos(_pos)
    }

    const handleWindowMouseDown = (e: MouseEvent): void => {
        setDrawing(true)

        const _pos = getMousePos(e)
        _pos && setCurrPos(_pos)
    }

    const handleWindowMouseup = (e: MouseEvent): void => {
        setDrawing(false)

        const _pos = getMousePos(e)
        _pos && setCurrPos(_pos)
    }

    const handleClear = (): void => {
        const _canvas = panelRef.current
        if (!_canvas) {
            return
        }
        const _ctx = _canvas.getContext('2d')
        _ctx?.clearRect(0, 0, _canvas.width, _canvas.height)
    }

    const handleSubmit = (): void => {
        const _canvas = panelRef.current
        if (!_canvas) {
            return
        }
        const _ctx = _canvas.getContext('2d')
        const imageData = _ctx?.getImageData(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
        if (imageData && props.onChange) {
            // logger('imageData', imageData)
            const _tensor = tf.browser.fromPixels(imageData, 1)
            const _sample = tf.image.resizeBilinear(_tensor, [28, 28])
            setMiniSample(Array.from(_sample.dataSync()))

            props.onChange(_sample)
        }
    }

    if (drawing && currPos) {
        // logger('in ', JSON.stringify(currPos))
        // logger('currPos', currPos)
        // logger('currPos.current', currPosRef)

        const prevPos = currPosRef.current
        if (prevPos && prevPos !== currPos) {
            draw(currPos, prevPos)
        }
    }
    return <Card title={'Drawing Panel'}>
        <p> Press Down Left Mouse to Draw </p>
        <p> Mouse : {JSON.stringify(currPos)} : {drawing ? 'Drawing...' : ''}</p>
        <canvas width={CANVAS_WIDTH} height={CANVAS_HEIGHT} style={{ backgroundColor: 'black' }} ref={panelRef}/>
        <div>
            <Button onClick={handleSubmit} type='primary'>Submit</Button>
            <Button onClick={handleClear}>Clear</Button>
        </div>
        <div>
            {miniSample && <RowImageWidget data={miniSample} shape={MNIST_SHAPE as [number, number]} />}
            Prediction : {props.prediction?.arraySync()}
        </div>
    </Card>
}

export default DrawPanelWidget
