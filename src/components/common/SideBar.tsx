import React from 'react'
import { Link } from 'react-router-dom'
import { Menu } from 'antd'
import {
    CalculatorOutlined,
    ExperimentOutlined,
    EyeOutlined,
    HomeOutlined,
    LineChartOutlined,
    ReadOutlined
} from '@ant-design/icons'
import { ClickParam } from 'antd/es/menu'

import logo from '../../react_logo.svg'

const { Item, SubMenu } = Menu

interface IProps {
    collapsed?: boolean
    onClick?: (param: ClickParam) => void
}

const SideBar = (props: IProps): JSX.Element => {
    return (
        <div>
            <header className='App-header'>
                <img src={logo} className='App-logo' alt='logo'/>
                <h2 style={{ color: 'white' }}>RTCamp</h2>
            </header>
            <Menu theme='dark' mode='inline' onClick={props.onClick}>
                <Item key='0'>
                    <HomeOutlined/>
                    <span><Link to='/'>首页</Link></span>
                </Item>
                <SubMenu title={<span><LineChartOutlined/><span>逻辑回归 Logisttc </span></span>}>
                    <Item key='1.1'>
                        <Link to='/curve'>
                            <span> 曲线拟合 Curve </span>
                        </Link>
                    </Item>
                    <Item key='1.2'>
                        <Link to='/iris'>
                            <span>鸢尾花 IRIS</span>
                        </Link>
                    </Item>
                </SubMenu>
                <SubMenu title={<span><CalculatorOutlined/><span>手写数字识别 MNIST</span></span>}>
                    <Item key='3.1'>
                        <Link to='/mnist/keras'>
                            <span> Mnist Layer Model </span>
                        </Link>
                    </Item>
                    <Item key='3.2'>
                        <Link to='/mnist/core'>
                            <span> Mnist Core API </span>
                        </Link>
                    </Item>
                </SubMenu>
                <SubMenu title={<span><EyeOutlined/><span>使用预训练模型 Mobilenet</span></span>}>
                    <Item key='4.1'>
                        <Link to='/mobilenet/basic'>
                            <span> 图片分类器 Mobilenet </span>
                        </Link>
                    </Item>
                    <Item key='4.2'>
                        <Link to='/mobilenet/knn'>
                            <span> 结合机器学习 Teachable Machine </span>
                        </Link>
                    </Item>
                    <Item key='4.3'>
                        <Link to='/mobilenet/transfer'>
                            <span> 迁移学习：分类器 Classifier </span>
                        </Link>
                    </Item>
                    <Item key='4.4'>
                        <Link to='/mobilenet/objdetector'>
                            <span> 迁移学习：对象识别 Object Detector </span>
                        </Link>
                    </Item>
                </SubMenu>
                <SubMenu title={<span><ReadOutlined /><span>循环神经网络 RNN</span></span>}>
                    <Item key='5.1'>
                        <Link to='/rnn/jena'>
                            <span> 时序数据 Jena Weather</span>
                        </Link>
                    </Item>
                    <Item key='5.2'>
                        <Link to='/rnn/sentiment'>
                            <span> 文本理解 IMDB Sentiment </span>
                        </Link>
                    </Item>
                    <Item key='5.3'>
                        <Link to='/rnn/lstm'>
                            <span> 文本生成 LSTM </span>
                        </Link>
                    </Item>
                </SubMenu>
                <SubMenu title={<span><ExperimentOutlined /><span>Sand Box</span></span>}>
                    <Item key='9.1'>
                        <Link to='/sandbox/tfvis'>
                            <span>TfVis Widget</span>
                        </Link>
                    </Item>
                    <Item key='9.2'>
                        <Link to='/sandbox/fetch'>
                            <span>Fetch Resource File</span>
                        </Link>
                    </Item>
                    <Item key='9.3'>
                        <Link to='/sandbox/array'>
                            <span>Show Diff with [] and TypedArray</span>
                        </Link>
                    </Item>
                </SubMenu>
            </Menu>
        </div>
    )
}

export default SideBar
