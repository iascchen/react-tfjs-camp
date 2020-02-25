import { RouteConfig } from 'react-router-config'

import Home from './components/common/Home'
import Curve from './components/curve/Curve'
import Iris from './components/iris/Iris'

import MnistKeras from './components/mnist/MnistKeras'
import MnistWeb from './components/mnist/MnistWeb'
import MnistCore from './components/mnist/MnistCore'

import MobilenetClassifier from './components/mobilenet/MobilenetClassifier'
import MobilenetKnnClassifier from './components/mobilenet/MobilenetKnnClassifier'

import FetchWidget from './components/sandbox/FetchWidget'
import TypedArrayWidget from './components/sandbox/TypedArrayWidget'
import TfvisWidget from './components/sandbox/TfvisWidget'

const routes: RouteConfig[] = [
    { path: '/', exact: true, component: Home },
    { path: '/curve', component: Curve },
    { path: '/iris', component: Iris },

    { path: '/mnist/keras', component: MnistKeras },
    { path: '/mnist/core', component: MnistCore },
    { path: '/mnist/web', component: MnistWeb },

    { path: '/mobilenet/basic', component: MobilenetClassifier },
    { path: '/mobilenet/knn', component: MobilenetKnnClassifier },
    // { path: '/mobilenet/posenet', component: PosenetWidget },

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

    '/mnist/keras': 'MNIST',
    '/mnist/core': 'MNIST tfjs-core',
    '/mnist/web': 'MNIST tfjs web',

    '/mobilenet/basic': 'Mobilenet Classifier',
    '/mobilenet/knn': 'Teachable Machine',

    '/sandbox/array': 'TypedArray',
    '/sandbox/fetch': 'Fetch',
    '/sandbox/tfvis': 'TfVis Widget'
}

export default routes
