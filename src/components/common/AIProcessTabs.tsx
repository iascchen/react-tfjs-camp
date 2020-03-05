import React, { useEffect, useState } from 'react'
import { Steps, Tabs } from 'antd'
import { Sticky, StickyContainer } from 'react-sticky'
import {
    BoxPlotFilled,
    BoxPlotOutlined,
    ControlFilled,
    ControlOutlined,
    DashboardFilled,
    DashboardOutlined,
    FileTextFilled,
    FileTextOutlined,
    TrophyFilled,
    TrophyOutlined
} from '@ant-design/icons'
import { logger } from '../../utils'

const { Step } = Steps

export enum AIProcessTabPanes {
    INFO = '1',
    DATA = '2',
    MODEL = '3',
    TRAIN = '4',
    PREDICT = '5'
}

const INVISIBLE_PANES: AIProcessTabPanes[] = []

interface IProps {
    children?: JSX.Element[]

    title?: string | JSX.Element
    current?: number
    invisiblePanes?: AIProcessTabPanes[]

    onChange?: (current: number) => void
}

const AIProcessTabs = (props: IProps): JSX.Element => {
    const [sCurrent, setCurrent] = useState<number>(0)
    const [sInvisiblePanes, setInvisiblePanes] = useState<AIProcessTabPanes[]>(INVISIBLE_PANES)

    useEffect(() => {
        logger('init current', props.current)
        props.current && setCurrent(props.current - 1)
    }, [props.current])

    useEffect(() => {
        props.invisiblePanes && setInvisiblePanes(props.invisiblePanes)
    }, [props.invisiblePanes])

    const handleChange = (current: number): void => {
        setCurrent(current)
        props.onChange && props.onChange(current + 1)
    }

    const getStepProps = (disabled: boolean, iconEnabled: JSX.Element, iconDisabled: JSX.Element): object => {
        if (disabled) {
            return { disabled, icon: iconDisabled }
        } else {
            return { icon: iconEnabled }
        }
    }

    return (
        <StickyContainer>
            <Sticky>{
                ({ style }) => {
                    const _style = { zIndex: 1, backgroundColor: 'white', ...style }
                    return (<div style={_style}>
                        <h1>{props.title}</h1>
                        <Steps current={sCurrent} onChange={handleChange}>
                            <Step title='问题' description='要解决的问题背景'
                                {...getStepProps(sInvisiblePanes?.includes(AIProcessTabPanes.INFO), <FileTextFilled/>, <FileTextOutlined/>)}
                            />
                            <Step title='数据' description='加载和准备所需数据'
                                {...getStepProps(sInvisiblePanes?.includes(AIProcessTabPanes.DATA), <BoxPlotFilled/>, <BoxPlotOutlined/>)}
                            />
                            <Step title='模型' description='修改模型结构和调参'
                                {...getStepProps(sInvisiblePanes?.includes(AIProcessTabPanes.MODEL), <ControlFilled/>, <ControlOutlined/>)}
                            />
                            <Step title='训练' description='训练过程可视化'
                                {...getStepProps(sInvisiblePanes?.includes(AIProcessTabPanes.TRAIN), <DashboardFilled/>, <DashboardOutlined/>)}
                            />
                            <Step title='推理' description='验证训练效果'
                                {...getStepProps(sInvisiblePanes?.includes(AIProcessTabPanes.PREDICT), <TrophyFilled/>, <TrophyOutlined/>)}
                            />
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
