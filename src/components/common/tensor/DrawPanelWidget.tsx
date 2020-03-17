import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Row } from 'antd'
import { logger, getTensorLabel } from '../../../utils'
import RowImageWidget from './RowImageWidget'

const CANVAS_WIDTH = 280
const CANVAS_HEIGHT = 280
const MNIST_SHAPE: number[] = [28, 28]

interface IPoint {
    x: number
    y: number
}

interface IProps {
    model?: tf.LayersModel
    prediction?: tf.Tensor
    onSubmit?: (tensor: tf.Tensor) => void
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
        // _ctx.moveTo(from.x, from.y)
        // _ctx.lineTo(to.x, to.y)
        _ctx.lineWidth = 10
        _ctx.strokeStyle = 'white'
        _ctx.fillStyle = 'white'
        _ctx.arc(to.x, to.y, 8, 0, 2 * Math.PI, false)
        _ctx.fill()
        _ctx.stroke()
        _ctx.closePath()
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
        if (imageData && props.onSubmit) {
            // logger('imageData', imageData)
            const _tensor = tf.browser.fromPixels(imageData, 1)
            const _sample = tf.image.resizeBilinear(_tensor, [28, 28])
            setMiniSample(Array.from(_sample.dataSync()))

            props.onSubmit(_sample.expandDims(0))
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
    return (
        <Card title={'Drawing Panel'} size='small' style={{ margin: '8px' }} >
            <p>Press Down Left Mouse to Draw</p>
            <p>Mouse : {JSON.stringify(currPos)} : {drawing ? 'Drawing...' : ''}</p>
            <Row className='centerContainer'>
                <div style={{ width: '300px', padding: '8px' }}>
                    <canvas width={CANVAS_WIDTH} height={CANVAS_HEIGHT} style={{ backgroundColor: 'black' }} ref={panelRef}/>
                </div>
            </Row>
            <Row className='centerContainer'>
                <div style={{ width: '300px', padding: '8px' }}>
                    <Button onClick={handleSubmit} type='primary' style={{ width: '30%', margin: '0 10%' }}>Submit</Button>
                    <Button onClick={handleClear} style={{ width: '30%', margin: '0 10%' }}>Clear</Button>
                </div>
            </Row>
            <Row className='centerContainer'>
                <div style={{ width: '300px', padding: '8px' }}>
                    {miniSample && <RowImageWidget data={miniSample} shape={MNIST_SHAPE} />}
                    &nbsp; Prediction : { props.prediction && `${getTensorLabel([props.prediction]).join(', ')}` }
                </div>
            </Row>
        </Card>
    )
}

export default DrawPanelWidget
