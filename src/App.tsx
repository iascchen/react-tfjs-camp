import React, { useState } from 'react'
import { BrowserRouter as Router } from 'react-router-dom'
import { Icon, Layout } from 'antd'

import './App.css'
import SideBar from './components/common/SideBar'
import BodyContainer from './components/common/BodyContainer'
import GitHubLogo from './components/common/GitHubLogo'
import ErrorBoundary from './components/common/ErrorBoundary'

const { Header, Sider, Footer } = Layout

const App = (): JSX.Element => {
    const [collapsed, setCollapsed] = useState(false)

    const onToggle = (): void => {
        setCollapsed(collapsed => !collapsed)
    }

    return (
        <Layout>
            <Router>
                <Sider trigger={null} collapsible collapsed={collapsed}>
                    <SideBar />
                </Sider>
                <Layout>
                    <Header style={{ background: '#fff', padding: '0' }}>
                        <Icon className='trigger' type={collapsed ? 'menu-unfold' : 'menu-fold'} onClick={onToggle}/>
                        <span style={{ margin: '0 8px' }}>Tensorflow.js React-Hooks Playground</span>
                        <GitHubLogo />
                    </Header>
                    <ErrorBoundary>
                        <BodyContainer />
                    </ErrorBoundary>
                    <Footer style={{ textAlign: 'center' }}>©2020 Created by Iasc CHEN(iascchen@gmail.com)</Footer>
                </Layout>
            </Router>
        </Layout>
    )
}

export default App
