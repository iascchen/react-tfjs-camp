import React from 'react'
import { Link } from 'react-router-dom'
import { Icon, Menu } from 'antd'
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
                        <span><Icon type='home'/> Home</span>
                    </Link>
                </Item>
                <Item key='1'>
                    <Link to='/curve'>
                        <span><Icon type='line-chart' /> 曲线拟合 Curve</span>
                    </Link>
                </Item>
                <Item key='2'>
                    <Link to='/iris'>
                        <span><Icon type='dot-chart' /> 鸢尾花 IRIS</span>
                    </Link>
                </Item>
                <Item key='3'>
                    <Link to='/mnist'>
                        <span><Icon type='calculator' /> 手写数字识别 MNIST</span>
                    </Link>
                </Item>

                <SubMenu title='Sand Box'>
                    <Item key='97'>
                        <Link to='/sandbox/tfvis'>
                            <span>TfVis Widget</span>
                        </Link>
                    </Item>
                    <Item key='98'>
                        <Link to='/sandbox/fetch'>
                            <span>Fetch Resource File</span>
                        </Link>
                    </Item>
                    <Item key='99'>
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
