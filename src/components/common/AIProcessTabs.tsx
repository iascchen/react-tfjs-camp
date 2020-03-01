import React, { useEffect, useState } from 'react'
import { Steps, Tabs } from 'antd'
import { Sticky, StickyContainer } from 'react-sticky'
import { CoffeeOutlined, ControlOutlined, DashboardOutlined, DotChartOutlined, FileTextOutlined } from '@ant-design/icons'
import { logger } from '../../utils'

const { Step } = Steps

export enum AIProcessTabPanes {
    INFO = '1',
    DATA = '2',
    MODEL = '3',
    TRAIN = '4',
    PREDICT = '5'
}

interface IProps {
    children?: JSX.Element[]

    title?: string | JSX.Element
    current?: number
    docUrl?: string

    onChange?: (current: number) => void
}

const AIProcessTabs = (props: IProps): JSX.Element => {
    const [sCurrent, setCurrent] = useState<number>(0)

    useEffect(() => {
        logger('init current', props.current)
        props.current && setCurrent(props.current - 1)
    }, [props.current])

    const handleChange = (current: number): void => {
        setCurrent(current)
        props.onChange && props.onChange(current + 1)
    }

    return (
        <StickyContainer>
            <Sticky>{
                ({ style }) => {
                    const _style = { zIndex: 1, backgroundColor: 'white', ...style }
                    return (<div style={_style}>
                        <h1>{props.title}</h1>
                        <Steps current={sCurrent} onChange={handleChange}>
                            <Step icon={<FileTextOutlined/>} title='问题' description='要解决的问题背景'/>
                            <Step icon={<DotChartOutlined/>} title='数据' description='加载和准备所需数据'/>
                            <Step icon={<ControlOutlined/>} title='模型' description='修改模型结构和调参'/>
                            <Step icon={<DashboardOutlined/>} title='训练' description='训练过程可视化'/>
                            <Step icon={<CoffeeOutlined/>} title='推理' description='验证训练效果'/>
                        </Steps>
                    </div>)
                }
            }
            </Sticky>

            <Tabs activeKey={(sCurrent + 1).toString()} tabBarStyle={{ height: 0 }}>
                {props.children}
            </Tabs>
        </StickyContainer>
    )
}

export default AIProcessTabs
