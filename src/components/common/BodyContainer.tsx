import React from 'react'
import { useLocation, Link } from 'react-router-dom'
import { renderRoutes } from 'react-router-config'
import { Breadcrumb, Layout } from 'antd'

import routes, { breadcrumbNameMap } from '../../routers'

const { Content } = Layout

const BodyContainer = (): JSX.Element => {
    const location = useLocation()

    const genBreadcrumbs = (): JSX.Element[] => {
        const pathSnippets = location.pathname.split('/').filter(i => i)
        const extraBreadcrumbItems = pathSnippets.map((_, index) => {
            const url = `/${pathSnippets.slice(0, index + 1).join('/')}`
            return (
                <Breadcrumb.Item key={url}>
                    <Link to={url}>{breadcrumbNameMap[url]}</Link>
                </Breadcrumb.Item>
            )
        })

        const breadcrumbItems = [
            <Breadcrumb.Item key='home'>
                <Link to='/'>Home</Link>
            </Breadcrumb.Item>
        ]
        breadcrumbItems.concat(extraBreadcrumbItems)

        return breadcrumbItems
    }

    return (
        <Content style={{ margin: '0 16px' }}>
            <Breadcrumb style={{ margin: '8px' }}>
                {genBreadcrumbs()}
            </Breadcrumb>
            <div style={{ padding: 24, background: '#fff', minHeight: '80vh' }}>
                {renderRoutes(routes)}
            </div>
        </Content>
    )
}

export default BodyContainer
