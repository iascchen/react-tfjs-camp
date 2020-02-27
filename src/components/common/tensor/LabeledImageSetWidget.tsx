import React, { useEffect, useReducer, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Icon, Upload } from 'antd'
import { RcFile, UploadChangeParam } from 'antd/es/upload'

import {
    checkUploadDone,
    getUploadFileArray,
    getUploadFileBase64,
    ILabeledImage,
    ILabeledImageFileJson,
    ILabeledImageSet,
    logger
} from '../../../utils'
import { UploadFile } from 'antd/es/upload/interface'

const MOBILENET_IMAGE_SIZE = 224

interface IProps {
    model?: tf.LayersModel
    prediction?: tf.Tensor

    labeledImgs?: ILabeledImageSet[]

    onLoad?: (value: ILabeledImageSet[]) => void
}

const LabeledImageSetWidget = (props: IProps): JSX.Element => {
    const downloadRef = useRef<HTMLAnchorElement>(null)

    const [sUploadingJson, setUploadingJson] = useState<UploadFile>()
    const [sLabeledImgs, setLabeledImgs] = useState<ILabeledImageSet[]>()
    const [waitingPush, forceWaitingPush] = useReducer((x: number) => x + 1, 0)

    useEffect(() => {
        logger('LabeledImageSetWidget init ', props.labeledImgs)
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
                setLabeledImgs(fileJson.labeledImageSetList)

                // push data to LabeledImageWidget
                const { onLoad } = props
                if (onLoad) {
                    logger('onLoad')
                    onLoad(fileJson.labeledImageSetList)
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

        const fileJson: ILabeledImageFileJson = { labeledImageSetList: sLabeledImgs }
        const a = downloadRef.current
        if (a) {
            const blob = new Blob(
                [JSON.stringify(fileJson, null, 2)],
                { type: 'application/json' })
            const blobUrl = window.URL.createObjectURL(blob)
            logger(blobUrl)

            // console.log(a)
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
            {sLabeledImgs?.map((labeled, index) => {
                const title = `${labeled.label}(${labeled.imageList?.length.toString()})`
                return <Card key={index} title={title} size='small'>
                    {
                        labeled.imageList?.map((imgItem: ILabeledImage) => {
                            return <img key={imgItem.uid} src={imgItem.img} alt={imgItem.name}
                                height={MOBILENET_IMAGE_SIZE / 2} style={{ margin: 4 }} />
                        })
                    }
                </Card>
            })}

            <Upload onChange={handleJsonChange} action={handleUpload} showUploadList={false} >
                <Button style={{ width: '200', margin: '10%' }} >
                    <Icon type='upload'/> Load
                </Button>
            </Upload>

            <a ref={downloadRef}/>
            <Button style={{ width: '200', margin: '10%' }} onClick={handleJsonSave}>
                <Icon type='save'/> Save
            </Button>
        </Card>
    )
}

export default LabeledImageSetWidget