# Neural net js

> The goal of this project is implement a neural net in javascript that is as easy to understand as possible.

This project will also look at using an [nbdev](https://github.com/fastai/nbdev/) style of library development using [tslab](https://github.com/yunabe/tslab) to provide a js kernel.

## Running

This project uses the [tslab Dockerfile for running on mybinder.org](https://github.com/yunabe/tslab-examples/blob/master/Dockerfile_prebuilt).

```
git clone https://github.com/pete88b/neuralnet_js.git
cd neuralnet_js
docker build -t neuralnet_js .
docker run -d -p 8888:8888 --mount type=bind,source="<absolute path to neuralnet_js on your machine>",target=/home/node/tslab-examples neuralnet_js
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

If you'd like to work with the tslab-examples, just docker run without the --mount;
```
docker run -d -p 8888:8888 neuralnet_js
```

TODO: expose a port so we can `python3 -m http.server 8000` from the terminal and then view demo web pages with http://localhost:8000/demo/

From a jupyter terminal you can

`$ python3 nbdev_js.py`
```
Converting 00_util.ipynb to src/util.js
Converting 10_nn.ipynb to src/nn.js
Converting index.ipynb to src/ex.js
```
