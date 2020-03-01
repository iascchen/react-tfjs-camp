import React, { useEffect, useState } from 'react'
import { message } from 'antd'
import ReactMarkdown from 'react-markdown'

import { fetchResource } from '../../utils'

const DEFAULT_SRC = '/docs/README.md'

interface IProps {
    url: string
}

const loadMD = async (url: string): Promise<string> => {
    const buffer = await fetchResource(url, false)
    return buffer.toString()
}

const MarkdownWidget = (props: IProps): JSX.Element => {
    const [sSource, setSource] = useState<string>(DEFAULT_SRC)

    useEffect(() => {
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

    return (
        <ReactMarkdown source={sSource} escapeHtml={true} />
    )
}

export default MarkdownWidget
