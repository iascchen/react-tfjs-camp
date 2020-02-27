import React, { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { Button, Card, Icon, Upload } from 'antd'
import { RcFile, UploadChangeParam } from 'antd/es/upload'

import {
    getUploadFileArray,
    getUploadFileBase64,
    ILabeledImage,
    ILabeledImageFileJson,
    ILabeledImageSet,
    logger
} from '../../../utils'

const MOBILENET_IMAGE_SIZE = 224

export const formatDataJson = (values: any): ILabeledImageFileJson => {
    logger(values)
    const { keys: _keys, labeledImageList } = values
    const ret = _keys.map((key: number) => JSON.parse(labeledImageList[key]))

    const formModel = { labeledImageSetList: ret }
    // logger(JSON.stringify(formModel))
    return formModel
}

interface IProps {
    model?: tf.LayersModel
    prediction?: tf.Tensor

    labeledImgs?: ILabeledImageSet[]

    // onSave?: (value: any) => void
}

const LabeledImageSetWidget = (props: IProps): JSX.Element => {
    const downloadRef = useRef<HTMLAnchorElement>(null)

    const [sLabeledImgs, setLabeledImgs] = useState<ILabeledImageSet[]>()

    useEffect(() => {
        logger('LabeledImageSetWidget init ', props.labeledImgs)
        setLabeledImgs(props.labeledImgs)
    }, props.labeledImgs)

    /***********************
     * Event Handler
     ***********************/

    const handleJsonSave = (): void => {
        const obj = formatDataJson(sLabeledImgs)
        const a = downloadRef.current
        if (a) {
            const blob = new Blob(
                [JSON.stringify(obj, null, 2)],
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

    const handleJsonChange = async ({ file }: UploadChangeParam): Promise<void> => {
        logger('handleFileChange', file.name)

        const buffer = await getUploadFileArray(file.originFileObj)
        const fileJson: ILabeledImageFileJson = JSON.parse(buffer.toString())

        const _labeledImageList = fileJson.labeledImageSetList
        if (_labeledImageList) {
            setLabeledImgs(_labeledImageList)
        }
        // TODO, push to KnnClassifier
    }

    const handleUpload = async (file: RcFile): Promise<string> => {
        // logger(file)
        return getUploadFileBase64(file)
    }

    /***********************
     * Render
     ***********************/

    return (
        <Card>
            {sLabeledImgs?.map((labeled, index) => {
                return <Card key={index} title={labeled.label} >
                    {
                        labeled.imageList?.map((imgItem: ILabeledImage) => {
                            return <div>
                                <img key={imgItem.uid} src={imgItem.img} alt={imgItem.name}
                                    width={MOBILENET_IMAGE_SIZE} />
                            </div>
                        })
                    }
                </Card>
            })}
            <a ref={downloadRef}/>

            <Upload onChange={handleJsonChange} action={handleUpload} showUploadList={false} >
                <Button>
                    <Icon type='upload'/> Load
                </Button>
            </Upload>

            <Button style={{ width: '30%', margin: '10%' }} onClick={handleJsonSave}>
                <Icon type='save'/> Save
            </Button>
        </Card>
    )
}

export default LabeledImageSetWidget
