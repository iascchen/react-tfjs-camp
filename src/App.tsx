import React, { useState } from 'react'
import { BrowserRouter as Router } from 'react-router-dom'
import { Layout } from 'antd'

import './App.css'
import SideBar from './components/common/SideBar'
import BodyContainer from './components/common/BodyContainer'
import GitHubLogo from './components/common/GitHubLogo'

const { Content, Header, Sider, Footer } = Layout

const App = (): JSX.Element => {
    const [sCollapsed, setCollapsed] = useState(true)

    const onCollapse = (): void => {
        setCollapsed(collapsed => !collapsed)
    }

    return (
        <Layout>
            <Router>
                <Sider collapsible collapsed={sCollapsed} onCollapse={onCollapse}>
                    <SideBar/>
                </Sider>
                <Layout className='site-layout'>
                    <Header style={{ background: '#fff', padding: '0' }}>
                        <span style={{ margin: '0 8px' }}>React Tensorflow.js Camp</span>
                        <GitHubLogo/>
                    </Header>
                    <Content style={{ margin: '16px' }}>
                        <BodyContainer/>
                    </Content>
                    <Footer style={{ textAlign: 'center' }}>©2020 Created by Iasc CHEN(iascchen@gmail.com)</Footer>
                </Layout>
            </Router>
        </Layout>
    )
}

export default App
