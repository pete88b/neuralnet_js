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

    epoch -1 valid loss 0.7883572345780145 metrics [ 0.43333333333333335 ]
    epoch 0 valid loss 0.5267556633586952 metrics [ 0.5666666666666667 ]
    epoch 1 valid loss 0.44099933286406734 metrics [ 0.7 ]
    epoch 2 valid loss 0.40672909169806126 metrics [ 0.7666666666666667 ]
    epoch 3 valid loss 0.3869921034748707 metrics [ 0.7333333333333333 ]
    epoch 4 valid loss 0.3731967804273562 metrics [ 0.8 ]
    epoch 5 valid loss 0.36288733941708184 metrics [ 0.7666666666666667 ]
    epoch 6 valid loss 0.35406780802342647 metrics [ 0.8 ]
    epoch 7 valid loss 0.34640309404547903 metrics [ 0.8 ]
    epoch 8 valid loss 0.3398996707466074 metrics [ 0.8 ]
    epoch 9 valid loss 0.33421554257888836 metrics [ 0.7666666666666667 ]
    epoch 10 valid loss 0.3285108760429585 metrics [ 0.7666666666666667 ]
    epoch 11 valid loss 0.3228930031170849 metrics [ 0.8 ]
    epoch 12 valid loss 0.31745012901639436 metrics [ 0.8 ]
    epoch 13 valid loss 0.3130312981606421 metrics [ 0.8 ]
    epoch 14 valid loss 0.3086841753495582 metrics [ 0.8 ]
    epoch 15 valid loss 0.30430963697452745 metrics [ 0.8 ]
    epoch 16 valid loss 0.300084500321322 metrics [ 0.8 ]
    epoch 17 valid loss 0.29557778712689037 metrics [ 0.8 ]
    epoch 18 valid loss 0.29140629672455687 metrics [ 0.8 ]
    epoch 19 valid loss 0.28759620227811683 metrics [ 0.8 ]
    epoch 20 valid loss 0.2837103287672317 metrics [ 0.8 ]
    epoch 21 valid loss 0.28015904255922847 metrics [ 0.8 ]
    epoch 22 valid loss 0.27654524617006615 metrics [ 0.8 ]
    epoch 23 valid loss 0.27356909201221596 metrics [ 0.8 ]
    epoch 24 valid loss 0.27050508342776985 metrics [ 0.8 ]


We can look at predictions our trained model makes on the validation data.

For each row, `learn.predict` gives us `[preds, predicted label, actual label]`


```javascript
let preds=learn.predict(learn.xValid, learn.yValid, (y=>`${argmax(y)}: ${IRIS_CLASS_MAP[argmax(y)]}`));
tail(preds,3);
```

    -3 [
      [ 0.01314091483672807, 0.24587330687958442, 0.7926335199651222 ],
      '2: Iris-virginica',
      '2: Iris-virginica'
    ]
    -2 [
      [ 0.001113699016691579, 0.4366974366023484, 0.712490424150912 ],
      '2: Iris-virginica',
      '2: Iris-virginica'
    ]
    -1 [
      [ 0.9961008020334862, 0.001475307146697653, 0.0019550674874320613 ],
      '0: Iris-setosa',
      '0: Iris-setosa'
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
        [ 0.011687742115645418, 0.42806382336273874, 0.1731676214386136 ],
        '1: Iris-versicolor',
        '0: Iris-setosa'
      ],
      [
        [ 0.3528794767136167, 0.12903259032530753, 0.05141777933637007 ],
        '0: Iris-setosa',
        '1: Iris-versicolor'
      ],
      [
        [
          0.009738425285232473,
          0.006588968072670429,
          0.004163304527861473
        ],
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
