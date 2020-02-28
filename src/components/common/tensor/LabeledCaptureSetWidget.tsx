import React, { useEffect, useReducer, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Icon, Upload } from 'antd'
import { RcFile, UploadChangeParam } from 'antd/es/upload'
import { UploadFile } from 'antd/es/upload/interface'

import {
    checkUploadDone,
    getUploadFileArray,
    getUploadFileBase64,
    ILabeledImage,
    ILabeledImageFileJson,
    ILabeledImageSet,
    logger
} from '../../../utils'
import TensorImageThumbWidget from './TensorImageThumbWidget'
import { MOBILENET_IMAGE_SIZE } from '../../../constant'

const encodeImageTensor = (labeledImgs: ILabeledImageSet[]): any[] => {
    if (!labeledImgs) {
        return []
    }

    labeledImgs.forEach((labeled, index) => {
        labeled.imageList?.forEach((imgItem: ILabeledImage) => {
            if (imgItem.tensor && !imgItem.img) {
                imgItem.tensor.print()
                const array = Array.from(imgItem.tensor.dataSync())
                imgItem.img = Buffer.from(array).toString('base64')
            }
        })
    })
    return labeledImgs
}

const decodeImageTensor = (labeledImgs: ILabeledImageSet[]): any[] => {
    logger('decodeImageTensor', labeledImgs)
    if (!labeledImgs) {
        return []
    }

    labeledImgs.forEach((labeled, index) => {
        labeled.imageList?.forEach((imgItem: ILabeledImage) => {
            if (imgItem.tensor && imgItem.img) {
                const buf = Buffer.from(imgItem.img)
                const _tensor = tf.tensor3d(buf, imgItem.tensor.shape, imgItem.tensor.dtype)
                imgItem.tensor = _tensor
                delete imgItem.img
            }
            logger(imgItem)
        })
    })
    return labeledImgs
}

interface IProps {
    model?: tf.LayersModel
    prediction?: tf.Tensor

    labeledImgs?: ILabeledImageSet[]

    onJsonLoad?: (value: ILabeledImageSet[]) => void
}

const LabeledCaptureSetWidget = (props: IProps): JSX.Element => {
    const downloadRef = useRef<HTMLAnchorElement>(null)

    const [sUploadingJson, setUploadingJson] = useState<UploadFile>()
    const [sLabeledImgs, setLabeledImgs] = useState<ILabeledImageSet[]>()

    const [waitingPush, forceWaitingPush] = useReducer((x: number) => x + 1, 0)

    useEffect(() => {
        // logger('LabeledImageSetWidget init ', props.labeledImgs)
        setLabeledImgs(props.labeledImgs)
    }, props.labeledImgs)

    useEffect(() => {
        if (!sUploadingJson) {
            return
        }

        // eslint-disable-next-line @typescript-eslint/no-misused-promises
        const timer = setInterval(async (): Promise<void> => {
            logger('Waiting upload...')
            if (checkUploadDone([sUploadingJson]) > 0) {
                forceWaitingPush()
            } else {
                clearInterval(timer)

                const buffer = await getUploadFileArray(sUploadingJson.originFileObj)
                const fileJson: ILabeledImageFileJson = JSON.parse(buffer.toString())
                const decoded = decodeImageTensor(fileJson.labeledImageSetList)
                setLabeledImgs(decoded)

                // push data to LabeledImageWidget
                const { onJsonLoad } = props
                if (onJsonLoad) {
                    logger('onJsonLoad')
                    onJsonLoad(decoded)
                }
            }
        }, 10)

        return () => {
            clearInterval(timer)
        }
    }, [waitingPush])

    /***********************
     * Event Handler
     ***********************/

    const handleJsonSave = (): void => {
        if (!sLabeledImgs) {
            return
        }

        const fileJson: ILabeledImageFileJson = { labeledImageSetList: encodeImageTensor(sLabeledImgs) }
        const a = downloadRef.current
        if (a) {
            const blob = new Blob(
                [JSON.stringify(fileJson, null, 2)],
                { type: 'application/json' })
            const blobUrl = window.URL.createObjectURL(blob)
            logger(blobUrl)

            // logger(a)
            const filename = 'labeledImages.json'
            a.href = blobUrl
            a.download = filename
            a.click()
            window.URL.revokeObjectURL(blobUrl)
        }
    }

    const handleJsonChange = ({ file }: UploadChangeParam): void => {
        logger('handleFileChange', file.name)

        setUploadingJson(file)
        forceWaitingPush()
    }

    const handleUpload = async (file: RcFile): Promise<string> => {
        // logger(file)
        return getUploadFileBase64(file)
    }

    /***********************
     * Render
     ***********************/

    return (
        <Card title={'Labeled Images'} size='small'>
            <Upload onChange={handleJsonChange} action={handleUpload} showUploadList={false} >
                <Button style={{ width: '300', margin: '0 10%' }}>
                    <Icon type='upload'/> Load Train Set
                </Button>
            </Upload>

            <a ref={downloadRef}/>
            <Button style={{ width: '300', margin: '0 10%' }} onClick={handleJsonSave}>
                <Icon type='save'/> Save Train Set
            </Button>

            {sLabeledImgs?.map((labeled, index) => {
                if (!labeled) {
                    return ''
                }

                const title = `${labeled.label}(${labeled.imageList?.length.toString()})`
                return <Card key={index} title={title} size='small'>
                    {
                        labeled.imageList?.map((imgItem: ILabeledImage) => {
                            if (imgItem.tensor) {
                                return <TensorImageThumbWidget key={imgItem.uid} data={imgItem.tensor}/>
                            } else if (imgItem.img) {
                                return <img key={imgItem.uid} src={imgItem.img} alt={imgItem.name}
                                    height={100} style={{ margin: 4 }} />
                            } else {
                                return <></>
                            }
                        })
                    }
                </Card>
            })}
        </Card>
    )
}

export default LabeledCaptureSetWidget
