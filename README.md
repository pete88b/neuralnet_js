# Neural net js

> The goal of this project is implement a neural net in javascript that is as easy to understand as possible.

This project will also look at using an [nbdev](https://github.com/fastai/nbdev/) style of library development using [tslab](https://github.com/yunabe/tslab) to provide a js kernel.

# Quick example

Let's train a classifier using [iris.data](https://archive.ics.uci.edu/ml/datasets/iris).


```javascript
import {argmax} from './src/util.module.js';
import {head,tail,parseCsv,IRIS_CLASS_MAP,IrisRowHandler} from './src/data.module.js';
import {BinaryCrossEntropyLoss,Linear,Sigmoid,ReLU,Learner} from './src/nn.module.js';
```


```javascript
let stringData=require('fs').readFileSync('data/iris.data').toString();
let data=parseCsv(stringData, new IrisRowHandler()).result;
let lossFn=new BinaryCrossEntropyLoss();
let model=[new Linear(4,50), new ReLU(), new Linear(50,3), new Sigmoid()];
let learn=new Learner(model, lossFn, data);
learn.fit(25);
```

    epoch -1 valid loss 0.9479352433795201 metrics [ 0.16666666666666666 ]
    epoch 0 valid loss 0.3918940015957982 metrics [ 0.7333333333333333 ]
    epoch 1 valid loss 0.30918027119353186 metrics [ 0.8333333333333334 ]
    epoch 2 valid loss 0.2746811037277132 metrics [ 0.8333333333333334 ]
    epoch 3 valid loss 0.2530131465769288 metrics [ 0.8666666666666667 ]
    epoch 4 valid loss 0.2381835103079123 metrics [ 0.8666666666666667 ]
    epoch 5 valid loss 0.2269235083808026 metrics [ 0.8666666666666667 ]
    epoch 6 valid loss 0.2182573318809938 metrics [ 0.8666666666666667 ]
    epoch 7 valid loss 0.21133856110713883 metrics [ 0.9 ]
    epoch 8 valid loss 0.2048899600128032 metrics [ 0.9 ]
    epoch 9 valid loss 0.19936697564963365 metrics [ 0.9 ]
    epoch 10 valid loss 0.19454810035687622 metrics [ 0.9 ]
    epoch 11 valid loss 0.19010850734868168 metrics [ 0.9 ]
    epoch 12 valid loss 0.18641622742173664 metrics [ 0.9 ]
    epoch 13 valid loss 0.18250157655119695 metrics [ 0.9 ]
    epoch 14 valid loss 0.17900738060361182 metrics [ 0.9 ]
    epoch 15 valid loss 0.17592690687939777 metrics [ 0.9 ]
    epoch 16 valid loss 0.17273859884997492 metrics [ 0.9 ]
    epoch 17 valid loss 0.16983281626142313 metrics [ 0.9 ]
    epoch 18 valid loss 0.16692967133988176 metrics [ 0.9 ]
    epoch 19 valid loss 0.16411886961096833 metrics [ 0.9 ]
    epoch 20 valid loss 0.16132731359970418 metrics [ 0.9 ]
    epoch 21 valid loss 0.15846197888518415 metrics [ 0.9 ]
    epoch 22 valid loss 0.1560980191410202 metrics [ 0.9 ]
    epoch 23 valid loss 0.15384031620459687 metrics [ 0.9333333333333333 ]
    epoch 24 valid loss 0.15153757745158905 metrics [ 0.9333333333333333 ]


We can look at predictions our trained model makes on the validation data.

For each row, `learn.predict` gives us `[preds, predicted label, actual label]`


```javascript
let preds=learn.predict(learn.xValid, learn.yValid, (y=>`${argmax(y)}: ${IRIS_CLASS_MAP[argmax(y)]}`));
tail(preds,3);
```

    -3 [
      [ 0.9961964007825763, 0.007737473342420738, 0.006337463666936329 ],
      '0: Iris-setosa',
      '0: Iris-setosa'
    ]
    -2 [
      [ 0.07623029536461032, 0.8285882195388641, 0.08304399546623287 ],
      '1: Iris-versicolor',
      '1: Iris-versicolor'
    ]
    -1 [
      [ 0.0004976528433462576, 0.06229985007007116, 0.9724517814948943 ],
      '2: Iris-virginica',
      '2: Iris-virginica'
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
        [ 0.016203874311635395, 0.1923948844409367, 0.025486821451765052 ],
        '1: Iris-versicolor',
        '0: Iris-setosa'
      ],
      [
        [ 0.5506634682013896, 0.33454377560614806, 0.08987844613360595 ],
        '0: Iris-setosa',
        '1: Iris-versicolor'
      ],
      [
        [ 0.5251229032595576, 0.0212231403266926, 0.6141118173008945 ],
        '2: Iris-virginica',
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
