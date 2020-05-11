import { RouteConfig } from 'react-router-config'

import Home from './components/common/Home'

import Curve from './components/curve/Curve'
import Iris from './components/iris/Iris'

import MnistLayersModelImpl from './components/mnist/MnistLayersModelImpl'
import MnistCoreApiImpl from './components/mnist/MnistCoreApiImpl'

import MobilenetClassifier from './components/mobilenet/MobilenetClassifier'
import MobilenetKnnClassifier from './components/mobilenet/MobilenetKnnClassifier'
import MobilenetTransfer from './components/mobilenet/MobilenetTransfer'
import MobilenetObjDetector from './components/mobilenet/MobilenetObjDetector'

import JenaWeather from './components/rnn/JenaWeather'
import SentimentImdb from './components/rnn/SentimentImdb'
import TextGenLstm from './components/rnn/TextGenLstm'

import HandPosePanel from './components/pretrained/HandPosePanel'
import FaceMeshPanel from './components/pretrained/FaceMeshPanel'
import PoseNetPanel from './components/pretrained/PoseNetPanel'

import FetchWidget from './components/sandbox/FetchWidget'
import TypedArrayWidget from './components/sandbox/TypedArrayWidget'
import TfvisWidget from './components/sandbox/TfvisWidget'

const routes: RouteConfig[] = [
    { path: '/', exact: true, component: Home },

    { path: '/curve', component: Curve },
    { path: '/iris', component: Iris },

    { path: '/mnist/layers', component: MnistLayersModelImpl },
    { path: '/mnist/core', component: MnistCoreApiImpl },

    { path: '/mobilenet/basic', component: MobilenetClassifier },
    { path: '/mobilenet/knn', component: MobilenetKnnClassifier },
    { path: '/mobilenet/transfer', component: MobilenetTransfer },
    { path: '/mobilenet/objdetector', component: MobilenetObjDetector },

    { path: '/rnn/jena', component: JenaWeather },
    { path: '/rnn/sentiment', component: SentimentImdb },
    { path: '/rnn/lstm', component: TextGenLstm },

    { path: '/pretrained/handpose', component: HandPosePanel },
    { path: '/pretrained/facemesh', component: FaceMeshPanel },
    { path: '/pretrained/posenet', component: PoseNetPanel },

    { path: '/sandbox/fetch', component: FetchWidget },
    { path: '/sandbox/array', component: TypedArrayWidget },
    { path: '/sandbox/tfvis', component: TfvisWidget },

    { path: '*', component: Home }
]

export default routes
