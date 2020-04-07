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
