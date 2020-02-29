import React, { ChangeEvent, useEffect, useReducer, useState } from 'react'
import { Button, Card, Input } from 'antd'
import { CameraOutlined, DeleteOutlined } from '@ant-design/icons'

import { arrayDispose, ILabeledImage, ILabeledImageSet, logger } from '../../../utils'
import TensorImageThumbWidget from './TensorImageThumbWidget'

interface IProps {
    value?: ILabeledImageSet

    onCapture?: (label: string) => Promise<ILabeledImage | void>
    onChange?: (value: ILabeledImageSet) => void
}

const LabeledCaptureInput = (props: IProps): JSX.Element => {
    const [sLabel, setLabel] = useState<string>('')
    const [sImageList, setImageList] = useState<ILabeledImage[]>([])

    const [ignore, forceUpdate] = useReducer((x: number) => x + 1, 0)

    useEffect(() => {
        if (!props.value) {
            return
        }

        const { label, imageList } = props.value
        label && setLabel(label)
        imageList && setImageList(imageList)
    }, [props.value])

    const pushToParentForm = (): void => {
        const { onChange } = props
        if (onChange) {
            const value: ILabeledImageSet = {
                label: sLabel,
                imageList: sImageList
            }
            onChange(value)
        }
    }

    const handleCapture = async (): Promise<void> => {
        // push data to LabeledCaptureWidget
        const { onCapture } = props
        if (onCapture) {
            const captured = await onCapture(sLabel)
            logger('handleCapture onCapture', captured)
            captured && sImageList.push(captured)
            pushToParentForm()
            forceUpdate()
        }
    }

    const handleReset = (): void => {
        arrayDispose(sImageList)
        pushToParentForm()
        forceUpdate()
    }

    const handleLabelChange = (e: ChangeEvent<HTMLInputElement>): void => {
        const value: string = e.target.value
        setLabel(value)
        pushToParentForm()
    }

    /***********************
     * Render
     ***********************/

    return (
        <Card>
            <Input onChange={handleLabelChange} defaultValue={sLabel} placeholder={'Label. such as: cat, dog...'} />
            <Button style={{ width: '30%', margin: '0 10%' }} size='small' onClick={handleReset}>
                <DeleteOutlined /> Drop All
            </Button>
            <Button style={{ width: '30%', margin: '0 10%' }} size='small' onClick={handleCapture}>
                <CameraOutlined /> Capture label
            </Button>
            {
                sImageList?.map((imgItem: ILabeledImage) => {
                    if (imgItem.tensor) {
                        return <TensorImageThumbWidget key={imgItem.uid} data={imgItem.tensor}/>
                    } else {
                        return <></>
                    }
                })
            }
        </Card>
    )
}

export default LabeledCaptureInput
