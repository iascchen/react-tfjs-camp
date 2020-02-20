import { RouteConfig } from 'react-router-config'

import Home from './components/common/Home'
import Curve from './components/curve/Curve'
import Iris from './components/iris/Iris'
import Mnist from './components/mnist/Mnist'

import FetchWidget from './components/sandbox/FetchWidget'
import TypedArrayWidget from './components/sandbox/TypedArrayWidget'
import TfvisWidget from './components/sandbox/TfvisWidget'

const routes: RouteConfig[] = [
    { path: '/', exact: true, component: Home },
    { path: '/curve', component: Curve },
    { path: '/iris', component: Iris },
    { path: '/mnist', component: Mnist },

    { path: '/sandbox/fetch', component: FetchWidget },
    { path: '/sandbox/array', component: TypedArrayWidget },
    { path: '/sandbox/tfvis', component: TfvisWidget }
]

interface IBreadcrumbMap {
    [index: string]: string
}

export const breadcrumbNameMap: IBreadcrumbMap = {
    '/': 'Home',
    '/curve': 'Curve',
    '/iris': 'Iris',
    '/mnist': 'MNIST',

    '/sandbox/array': 'TypedArray',
    '/sandbox/fetch': 'Fetch',
    '/sandbox/tfvis': 'TfVis Widget'
}

export default routes
