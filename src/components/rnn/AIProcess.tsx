import React, { useEffect, useState } from 'react'
import { Steps, Tabs } from 'antd'
import { StickyContainer, Sticky } from 'react-sticky'
import { CoffeeOutlined, ControlOutlined, DashboardOutlined, DotChartOutlined, FileTextOutlined } from '@ant-design/icons'
import { logger } from '../../utils'

const { Step } = Steps

interface IProps {
    children?: JSX.Element[]

    title?: string | JSX.Element
    current?: number

    onChange?: (current: number) => void
}

const AIProcess = (props: IProps): JSX.Element => {
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

export default AIProcess
