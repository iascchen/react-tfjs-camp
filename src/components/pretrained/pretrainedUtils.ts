import { logger } from '../../utils'

export const drawPoint = (ctx: CanvasRenderingContext2D, y: number, x: number, r: number): void => {
    ctx.beginPath()
    ctx.arc(x, y, r, 0, 2 * Math.PI)
    ctx.fill()
}

export const drawPath = (ctx: CanvasRenderingContext2D, points: number[][], closePath: boolean): void => {
    const region = new Path2D()
    region.moveTo(points[0][0], points[0][1])
    for (let i = 1; i < points.length; i++) {
        const point = points[i]
        region.lineTo(point[0], point[1])
    }

    if (closePath) {
        region.closePath()
    }
    ctx.stroke(region)
}

export const drawSegment = (ctx: CanvasRenderingContext2D, [ay, ax]: number[],
    [by, bx]: number[], scale: number): void => {
    ctx.beginPath()
    ctx.moveTo(ax * scale, ay * scale)
    ctx.lineTo(bx * scale, by * scale)
    ctx.lineWidth = 2
    // ctx.strokeStyle = color;
    ctx.stroke()
}

export const downloadJson = (content: any, fileName: string, downloadRef: HTMLAnchorElement): void => {
    const a = downloadRef
    if (a) {
        const blob = new Blob([JSON.stringify(content, null, 2)],
            { type: 'application/json' })
        const blobUrl = window.URL.createObjectURL(blob)
        logger(blobUrl)

        a.href = blobUrl
        a.download = fileName
        a.click()
        window.URL.revokeObjectURL(blobUrl)
    }
}
