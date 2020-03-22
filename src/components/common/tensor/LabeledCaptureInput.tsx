import React, { ChangeEvent, useState } from 'react'
import { Button, Card, Input, Modal } from 'antd'
import { CameraOutlined, DeleteOutlined } from '@ant-design/icons'

import { arrayDispose, ILabeledImage, ILabeledImageSet, logger } from '../../../utils'
import TensorImageThumbWidget from './TensorImageThumbWidget'
import * as tf from '@tensorflow/tfjs'

const DEFAULT_HEIGHT = 100

interface IProps {
    onCapture?: (label: string) => Promise<ILabeledImage | void>
    onChange?: (value: ILabeledImageSet) => void
}

const LabeledCaptureInput = (props: IProps): JSX.Element => {
    const [sPreviewImage, setPreviewImage] = useState<tf.Tensor3D>()
    const [modelDisplay, setModalDisplay] = useState(false)

    const [sLabel, setLabel] = useState<string>('')
    const [sImageList, setImageList] = useState<ILabeledImage[]>([])

    // useEffect(() => {
    //     if (!props.value) {
    //         return
    //     }
    //
    //     const { label, imageList } = props.value
    //     label && setLabel(label)
    //     imageList && setImageList(imageList)
    // }, [props.value])

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
            // forceUpdate()
        }
    }

    const handleReset = (): void => {
        arrayDispose(sImageList)
        pushToParentForm()
        // forceUpdate()
    }

    const handleCancel = (): void => {
        setModalDisplay(false)
    }

    const handlePreview = (data: tf.Tensor3D): void => {
        if (data) {
            setPreviewImage(data)
            setModalDisplay(true)
        }
    }

    const handleDelete = (uid: string): void => {
        const index = sImageList.findIndex((value, index) => {
            return value.uid === uid
        })
        if (index >= 0) {
            sImageList.splice(index, 1)
            // forceUpdate()
        }
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
            <div className='centerContainer'>
                <Button style={{ width: '20%', margin: '0 15%' }} size='small' onClick={handleReset}>
                    <DeleteOutlined /> Drop All
                </Button>
                <Button style={{ width: '20%', margin: '0 15%' }} size='small' onClick={handleCapture}>
                    <CameraOutlined /> Capture
                </Button>
            </div>

            {sImageList?.map((imgItem: ILabeledImage, index) => {
                if (imgItem.tensor) {
                    return <TensorImageThumbWidget key={imgItem.uid} uid={imgItem.uid} data={imgItem.tensor}
                        height={DEFAULT_HEIGHT} onPreview={handlePreview} onDelete={handleDelete} />
                } else {
                    return <></>
                }
            })}

            <Modal visible={modelDisplay} footer={null} onCancel={handleCancel}>
                {sPreviewImage && <TensorImageThumbWidget data={sPreviewImage}/>}
            </Modal>
        </Card>
    )
}

export default LabeledCaptureInput
