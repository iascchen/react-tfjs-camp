import React, { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import MathJax from '@matejmazur/react-mathjax'
import RemarkMathPlugin from 'remark-math'
import { message } from 'antd'

import { fetchResource, logger } from '../../utils'

const DEFAULT_INFO = 'Please set props url or source'

const loadMD = async (url: string): Promise<string> => {
    const buffer = await fetchResource(url, false)
    return buffer.toString()
}

const math = (p: {value: string}): JSX.Element => {
    return <MathJax.Node>{p.value}</MathJax.Node>
}

const inlineMath = (p: {value: string}): JSX.Element => {
    return <MathJax.Node inline>{p.value}</MathJax.Node>
}

const renderers = {
    math, inlineMath
}

interface IProps {
    source?: string
    url?: string
    imgPathPrefix?: string
}

const MarkdownWidget = (props: IProps): JSX.Element => {
    const [sSource, setSource] = useState<string>(DEFAULT_INFO)

    useEffect(() => {
        if (!props.url) {
            return
        }
        logger('Load MD from url: ', props.url)

        // Fetch and load MD content
        loadMD(props.url).then(
            (src) => {
                const prefix = props.imgPathPrefix ?? '/docs'
                const _src = src.replace(/.\/images/g, `${prefix}/images`)
                setSource(_src)
            }, (e) => {
                // eslint-disable-next-line @typescript-eslint/no-floating-promises
                message.error(e.message)
            })
    }, [props.url])

    useEffect(() => {
        props.source && setSource(props.source)
    }, [props.source])

    return (
        <MathJax.Context>
            <ReactMarkdown source={sSource} escapeHtml={true} plugins={[RemarkMathPlugin]} renderers={renderers}/>
        </MathJax.Context>
    )
}

export default MarkdownWidget
