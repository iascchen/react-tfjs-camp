import React from 'react'
import { Link } from 'react-router-dom'
import { Menu } from 'antd'
import { CalculatorOutlined, EyeOutlined, FileTextOutlined, HomeOutlined, LineChartOutlined } from '@ant-design/icons'
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
                <p>Tensorflow.js Playground</p>
            </header>
            <Menu theme='dark' mode='inline' onClick={props.onClick}>
                <Item key='0'>
                    <Link to='/'>
                        <span><HomeOutlined/> Home </span>
                    </Link>
                </Item>
                <SubMenu title={<span><LineChartOutlined/> 线性回归 </span>}>
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
                <SubMenu title={<span><CalculatorOutlined/> 手写数字识别 MNIST </span>}>
                    <Item key='3.1'>
                        <Link to='/mnist/web'>
                            <span> Tfjs Web 数据加载 </span>
                        </Link>
                    </Item>
                    <Item key='3.2'>
                        <Link to='/mnist/keras'>
                            <span> Tfjs Gz 数据加载 </span>
                        </Link>
                    </Item>
                    <Item key='3.3'>
                        <Link to='/mnist/core'>
                            <span> Tfjs-core 版 </span>
                        </Link>
                    </Item>
                </SubMenu>
                <SubMenu title={<span><EyeOutlined/> 使用预训练模型 Mobilenet </span>}>
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
                            <span> 迁移学习 模型修改 </span>
                        </Link>
                    </Item>
                </SubMenu>
                <SubMenu title={<span><FileTextOutlined/> RNN </span>}>
                    <Item key='5.1'>
                        <Link to='/rnn'>
                            <span> RNN </span>
                        </Link>
                    </Item>
                    <Item key='5.2'>
                        <Link to='/lstm'>
                            <span> LSTM </span>
                        </Link>
                    </Item>
                </SubMenu>
                <SubMenu title='Sand Box'>
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
