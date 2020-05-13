import React, { useEffect, useState } from 'react'
import { Col, Row, Select, Tag, Tooltip } from 'antd'

import { logger, STATUS } from '../../utils'
import { ImagenetClasses } from './ImagenetClasses'

const { Option } = Select

const LANGUAGES = ['en', 'zh']

const ImagenetTagsWidget = (): JSX.Element => {
    /***********************
     * useState
     ***********************/

    const [sStatus, setStatus] = useState<STATUS>(STATUS.INIT)
    const [sLang, setLang] = useState<string>()

    /***********************
     * useEffect
     ***********************/

    useEffect(() => {
        if (!sLang) {
            return
        }
        logger('Translate Tags ...', sLang)

        // TODO Translate
    }, [sLang])

    /***********************
     * Functions
     ***********************/

    const handleLangChange = (value: string): void => {
        setLang(value)
    }

    /***********************
     * Render
     ***********************/

    return (
        <Row>
            <Col span={2}>
                <Select onChange={handleLangChange} defaultValue={'en'}>
                    {LANGUAGES.map((v) => {
                        return <Option key={v} value={v}>{v}</Option>
                    })}
                </Select>
            </Col>
            <Col span={22}>
                <h2> 1000 Classes of ImageNet </h2>
            </Col>
            <Col span={24}>
                {Object.keys(ImagenetClasses).map((key, index) => {
                    const tag = ImagenetClasses[index]
                    const isLongTag = tag.length > 20
                    const tagElem = (
                        <Tag key={tag}>
                            {isLongTag ? `${tag.slice(0, 20)}...` : tag}
                        </Tag>
                    )
                    return isLongTag ? (
                        <Tooltip title={tag} key={tag}>
                            {tagElem}
                        </Tooltip>
                    ) : (
                        tagElem
                    )
                })}
            </Col>
        </Row>
    )
}

export default ImagenetTagsWidget
