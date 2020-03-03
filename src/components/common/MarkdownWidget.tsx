import React, { useEffect, useState } from 'react'
import { message } from 'antd'
import ReactMarkdown from 'react-markdown'

import { fetchResource, logger } from '../../utils'

const DEFAULT_INFO = 'Please set props url or source'

interface IProps {
    url?: string
    source?: string
}

const loadMD = async (url: string): Promise<string> => {
    const buffer = await fetchResource(url, false)
    return buffer.toString()
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
                // logger('src', src)
                setSource(src)
            }, (e) => {
                // eslint-disable-next-line @typescript-eslint/no-floating-promises
                message.error(e.message)
            })
    }, [props.url])

    useEffect(() => {
        props.source && setSource(props.source)
    }, [props.source])

    return (
        <ReactMarkdown source={sSource} escapeHtml={true} />
    )
}

export default MarkdownWidget
