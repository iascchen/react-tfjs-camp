import { RouteConfig } from 'react-router-config'

import Home from './components/common/Home'
import NotFound from './components/common/NotFound'

import Curve from './components/curve/Curve'
import Iris from './components/iris/Iris'

import MnistKeras from './components/mnist/MnistKeras'
import MnistCore from './components/mnist/MnistCore'

import MobilenetClassifier from './components/mobilenet/MobilenetClassifier'
import MobilenetKnnClassifier from './components/mobilenet/MobilenetKnnClassifier'
import MobilenetTransferWidget from './components/mobilenet/MobilenetTransferWidget'
import MobilenetObjDetector from './components/mobilenet/MobilenetObjDetector'

import RnnJena from './components/rnn/RnnJena'
import SentimentWidget from './components/rnn/SentimentWidget'
import TextGenLstm from './components/rnn/TextGenLstm'

import FetchWidget from './components/sandbox/FetchWidget'
import TypedArrayWidget from './components/sandbox/TypedArrayWidget'
import TfvisWidget from './components/sandbox/TfvisWidget'

const routes: RouteConfig[] = [
    { path: '/', exact: true, component: Home },

    { path: '/curve', component: Curve },
    { path: '/iris', component: Iris },

    { path: '/mnist/keras', component: MnistKeras },
    { path: '/mnist/core', component: MnistCore },

    { path: '/mobilenet/basic', component: MobilenetClassifier },
    { path: '/mobilenet/knn', component: MobilenetKnnClassifier },
    { path: '/mobilenet/transfer', component: MobilenetTransferWidget },
    { path: '/mobilenet/objdetector', component: MobilenetObjDetector },

    { path: '/rnn/jena', component: RnnJena },
    { path: '/rnn/sentiment', component: SentimentWidget },
    { path: '/rnn/lstm', component: TextGenLstm },

    { path: '/sandbox/fetch', component: FetchWidget },
    { path: '/sandbox/array', component: TypedArrayWidget },
    { path: '/sandbox/tfvis', component: TfvisWidget },

    { path: '*', component: NotFound }
]

interface IBreadcrumbMap {
    [index: string]: string
}

export const breadcrumbNameMap: IBreadcrumbMap = {
    '/': 'Home',
    '/curve': 'Curve',
    '/iris': 'Iris',

    '/mnist/keras': 'MNIST LayerModel',
    '/mnist/core': 'MNIST Core',

    '/mobilenet/basic': 'Mobilenet Classifier',
    '/mobilenet/knn': 'Teachable Machine',
    '/mobilenet/transfer': 'Transfer Learning: Classifier',
    '/mobilenet/objdetector': 'Transfer Learning: Object Detector',

    '/rnn/jena': 'Jena Weather',
    '/rnn/sentiment': 'Sentiment',
    '/rnn/lstm': 'Lstm Text',

    '/sandbox/array': 'TypedArray',
    '/sandbox/fetch': 'Fetch',
    '/sandbox/tfvis': 'TfVis Widget'
}

export default routes
