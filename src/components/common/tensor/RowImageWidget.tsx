import React, { useEffect, useRef, useState } from 'react'

const draw = (canvas: HTMLCanvasElement | null, data: number[] | undefined, shape: [number, number]): void => {
    if (!canvas || !data || data.length === 0) {
        return
    }

    const [width, height] = shape
    canvas.width = width
    canvas.height = height

    const ctx = canvas.getContext('2d')
    const imageData = new ImageData(width, height)
    // const data = image.dataSync()
    for (let i = 0; i < height * width; ++i) {
        const j = i * 4
        imageData.data[j] = data[i] * 255
        imageData.data[j + 1] = data[i] * 255
        imageData.data[j + 2] = data[i] * 255
        imageData.data[j + 3] = 255
    }
    ctx?.putImageData(imageData, 0, 0)
}

interface IProps {
    data?: number[]
    shape?: [number, number]
}

const RowImageWidget = (props: IProps): JSX.Element => {
    const [shape, setShape] = useState<[number, number]>([28, 28])

    const canvasRef = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        if (props.shape) {
            setShape(props.shape)
        }
    }, [props.shape])

    useEffect(() => {
        if (!props.data || !canvasRef) {
            return
        }
        draw(canvasRef.current, props.data, shape)
    }, [props.data, shape])

    return <canvas width={shape[0]} height={shape[1]} ref={canvasRef} />
}

export default RowImageWidget
