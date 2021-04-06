# Neural net js

> The goal of this project is implement a neural net in javascript that is as easy to understand as possible.

This project will also look at using an [nbdev](https://github.com/fastai/nbdev/) style of library development using [tslab](https://github.com/yunabe/tslab) to provide a js kernel.

## Live demo

This [gist via bl.ocks](https://bl.ocks.org/pete88b/2aa60d189006bba7c59039f1e9d55936) shows how a model, learning in browser, converges with a 3d scatter plot.

# Quick example

Let's train a classifier using [iris.data](https://archive.ics.uci.edu/ml/datasets/iris).


```javascript
import {argmax} from './src/util.module.js';
import {head,tail,parseCsv,IRIS_CLASS_MAP,IrisRowHandler} from './src/data.module.js';
import {BinaryCrossEntropyLoss,CrossEntropyLoss,Linear,Sigmoid,ReLU,Learner} from './src/nn.module.js';
```


```javascript
let stringData=require('fs').readFileSync('data/iris.data').toString();
let data=parseCsv(stringData, new IrisRowHandler('classid')).result;
let lossFn=new CrossEntropyLoss();
let model=[new Linear(4,50), new ReLU(), new Linear(50,3)];
let learn=new Learner(model, lossFn, data);
learn.fit(25);
```

    epoch -1 valid loss 1.4739151121882226 metrics [ 0.4 ]
    epoch 0 valid loss 0.5848744757989532 metrics [ 0.6666666666666666 ]
    epoch 1 valid loss 0.49159050206731364 metrics [ 0.7666666666666667 ]
    epoch 2 valid loss 0.4229015771154563 metrics [ 0.8666666666666667 ]
    epoch 3 valid loss 0.38638241771021936 metrics [ 0.9 ]
    epoch 4 valid loss 0.3604730039888145 metrics [ 0.9333333333333333 ]
    epoch 5 valid loss 0.3398951858760056 metrics [ 0.9333333333333333 ]
    epoch 6 valid loss 0.32765219645252874 metrics [ 0.9333333333333333 ]
    epoch 7 valid loss 0.31447320048095306 metrics [ 0.9333333333333333 ]
    epoch 8 valid loss 0.309237885506544 metrics [ 0.9333333333333333 ]
    epoch 9 valid loss 0.2975130858963832 metrics [ 0.9333333333333333 ]
    epoch 10 valid loss 0.29207270307432676 metrics [ 0.9333333333333333 ]
    epoch 11 valid loss 0.28569384036078393 metrics [ 0.9333333333333333 ]
    epoch 12 valid loss 0.2719198981495626 metrics [ 0.9333333333333333 ]
    epoch 13 valid loss 0.2714915252404436 metrics [ 0.9333333333333333 ]
    epoch 14 valid loss 0.2648930707928398 metrics [ 0.9333333333333333 ]
    epoch 15 valid loss 0.2575430138085798 metrics [ 0.9333333333333333 ]
    epoch 16 valid loss 0.26139239668554426 metrics [ 0.9333333333333333 ]
    epoch 17 valid loss 0.24974931679898213 metrics [ 0.9333333333333333 ]
    epoch 18 valid loss 0.24400324123985023 metrics [ 0.9333333333333333 ]
    epoch 19 valid loss 0.23800864767053406 metrics [ 0.9666666666666667 ]
    epoch 20 valid loss 0.23783490396866186 metrics [ 0.9666666666666667 ]
    epoch 21 valid loss 0.23414740306625575 metrics [ 0.9666666666666667 ]
    epoch 22 valid loss 0.23276390444542142 metrics [ 0.9666666666666667 ]
    epoch 23 valid loss 0.22798949285307182 metrics [ 0.9666666666666667 ]
    epoch 24 valid loss 0.22188885746884998 metrics [ 0.9666666666666667 ]


We can look at predictions our trained model makes on the validation data.

For each row, `learn.predict` gives us `[preds, predicted label, actual label]`


```javascript
function yToLabelFn(y) {
    if (Array.isArray(y)) {
        y=argmax(y);
    }
    return `${y}: ${IRIS_CLASS_MAP[y]}`
}
let preds=learn.predict(learn.xValid, learn.yValid, yToLabelFn);
tail(preds,3);
```

    -3 [
      [ 5.229994592729576, -0.818909550073246, -3.375091406219404 ],
      '0: Iris-setosa',
      '0: Iris-setosa'
    ]
    -2 [
      [ -0.6277809352701051, 1.8666505942828426, 0.6384743521119428 ],
      '1: Iris-versicolor',
      '1: Iris-versicolor'
    ]
    -1 [
      [ -1.9751569068291988, 5.704614417138167, 2.861458954249989 ],
      '1: Iris-versicolor',
      '1: Iris-versicolor'
    ]


and we can easily make up some data of our own and see what the model predicts


```javascript
let rh=new IrisRowHandler();
rh.handleRow('5.5,3.5,10.4,0.2,Iris-setosa');
rh.handleRow('5.5,2.6,0.4,1.2,Iris-versicolor');
rh.handleRow('0.5,3.0,5.1,1.8,Iris-virginica');
learn.predict(...rh.result, (y=>`${argmax(y)}: ${IRIS_CLASS_MAP[argmax(y)]}`));
```

    [
      [
        [ -2.1274712852819375, 5.381204677236861, 3.6462702824166584 ],
        '1: Iris-versicolor',
        '0: Iris-setosa'
      ],
      [
        [ -1.3324676896825176, 2.382549052851048, -2.294187675498703 ],
        '1: Iris-versicolor',
        '1: Iris-versicolor'
      ],
      [
        [ 2.0427834286750493, -1.8383986297828827, -6.373072562622682 ],
        '0: Iris-setosa',
        '2: Iris-virginica'
      ]
    ]


## Setting up a development environment

This project uses the [tslab Dockerfile for running on mybinder.org](https://github.com/yunabe/tslab-examples/blob/master/Dockerfile_prebuilt).

```
git clone https://github.com/pete88b/neuralnet_js.git
cd neuralnet_js
docker build -t neuralnet_js .
docker run -d -p 8888:8888 -p 8000:8000 --mount type=bind,source="<absolute path to neuralnet_js on your machine>",target=/home/node/tslab-examples neuralnet_js
```
Check the container logs for a section that looks a bit like;
```
To access the notebook, open this file in a browser:
file:///home/node/.local/share/jupyter/runtime/nbserver-1-open.html
Or copy and paste one of these URLs:
http://4f1ee683b96c:8888/?token=b3dd4ed644617cfb795dd8eb899aafea4c2168d8dc897357
or http://127.0.0.1:8888/?token=b3dd4ed644617cfb795dd8eb899aafea4c2168d8dc897357
```
Copy the `127.0.0.1` URL and paste it into your browser.

Note: The `docker run` command above uses 2 ports;
- 8888 for jupyter and
- 8000 which we can use to view demo web pages with http://localhost:8000/demo/
    - Run `python3 -m http.server 8000` from a jupyter terminal to start the http server
    - we manually start the server so that we don't have to change Dockerfile

If you'd like to work with the tslab-examples, just docker run without the --mount;
```
docker run -d -p 8888:8888 neuralnet_js
```

## Converting notebooks to javascript files

From a jupyter terminal you can;

`$ python3 nbdev_js.py`
```
Converting 00_testutil.ipynb to src/testutil.js
Converting 00_testutil.ipynb to src/testutil.module.js
...
Converting index.ipynb to README.md
```

Note: I've added a local file `mk` so I only have to type `./mk` at the terminal.

See: `99_nbdev_js.ipynb` for notebook conversion details.
