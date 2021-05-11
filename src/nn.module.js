/**
Imports we need in nn.module.js
*/
import {exp,shape,transpose,dotProduct,randn,uniform,zeros,argmax,mean,round} from './util.module.js';
import {matrixSum1d,matrixSum2d,matrixSubtract1d,matrixSubtract2d,matrixMultiply1d,matrixMultiply2d} from './util.module.js';
import {head,tail,parseCsv,IRIS_CLASS_MAP,IrisRowHandler,shuffle,split,batches} from './data.module.js';

/**
yTrue can be either 2d (one-hot encoded targets) or 1d (array of class IDs).
*/
function accuracy(yPred2d,yTrue) {
    const yPredShape=shape(yPred2d);
    const yTrueShape=shape(yTrue);
    if (yPredShape[0] != yTrueShape[0]) {
        throw new Error(`Expected yPred2d.length ${yPredShape[0]} to equal yTrue.length ${yTrueShape[0]}`);
    }
    if (yTrueShape.length == 2 && yPredShape[1] != yTrueShape[1]) {
        throw new Error(`Expected shape(yPred2d)[1] ${yPredShape[1]} to equal shape(yTrue)[1] ${yTrueShape[1]}`);
    }
    let correctCount=0;
    for (let i=0; i<yPred2d.length; i++) {
        let p = argmax(yPred2d[i]);
        let t = (yTrueShape.length == 2) ? argmax(yTrue[i]) : yTrue[i];
        if (p == t) {
            correctCount++;
        }
    }
    return correctCount/yPredShape[0];
}

/**
*/
class MSE {
    forward(yPred2d,yTrue2d) {
        this.error=matrixSubtract2d(yPred2d,yTrue2d);
        return mean(this.error.map(row=>row.map(elem=>elem**2)));
    }
    backward() {
        this.grad=matrixMultiply2d(this.error, 2/this.error.length);
        return this.grad;
    }
}

/**
Takes a 2d array and returns a 1d array of the log of the sum of the exp for each row.
*/
function logsumexp(x) {
    const m = x.map(a => Math.max(...a));
    let temp = x.map((row,i) => row.map(e => e-m[i])); // x-m[:,None]
    temp = temp.map(row => row.map(e => exp(e)));      // .exp()
    temp = temp.map(row => row.reduce((a,b) => a+b))   // .sum(-1)
    temp = temp.map(a => Math.log(a));                 // .log()
    return matrixSum1d(m, temp);                       // return m + ...
}

/**
Takes a 2d array and returns a 2d array of log softmax for each element.
*/
function log_softmax(x) {
    const _logsumexp = logsumexp(x);
    return x.map((row,i) => row.map(e => e-_logsumexp[i]));
}

/**
Takes a 2d input (log softmax predictions) and a 1d array of target class IDs and returns the negative log likelihood.
*/
function nll(input, target) {
    return -mean(input.map((row,i) => row[target[i]]));
}

/**
Cross entropy with softmax.
yTrue1d is an array of target class IDs - not a 2d array of 1 hot encoded targets.
*/
class CrossEntropyLoss {
    softmax1d(a) {
        const maxValue=Math.max(...a); // normalize values for numerical stability (log sum exp)
        const temp=a.map(e => exp(e-maxValue));
        const sum=temp.reduce((a,b)=>a+b);
        return temp.map(e=>e/sum);
    }
        
    forward(yPred2d,yTrue1d) {
        this.yPred2d=yPred2d.map(yPred1d => this.softmax1d(yPred1d));
        this.yTrue1d=yTrue1d;
        const temp=this.yPred2d.map((yPred1d,i) => Math.log(yPred1d[yTrue1d[i]])); // TODO: add tiny value to avoid log(0)
        return -temp.reduce((a,b) => a+b) / temp.length;
    }
    
    backward() {
        const yTrue1d=this.yTrue1d;
        this.grad=this.yPred2d.map(yPred1d => [...yPred1d]); // copy preds
        this.grad.forEach((yPred1d,i)=>yPred1d[yTrue1d[i]]-=1);
        return this.grad;
    }
}

/**
*/
class BinaryCrossEntropyLoss {
    _forward1d(yPred1d,yTrue1d) {
        const temp=yPred1d.map((yPred,i) => Math.log((yTrue1d[i]==1.) ? yPred : 1-yPred));
        return -temp.reduce((a,b) => a+b) / temp.length;
    }
    forward(yPred2d,yTrue2d) {
        this.yPred2d=yPred2d;
        this.yTrue2d=yTrue2d;
        const lossValue1d=yPred2d.map((yPred1d,i) => this._forward1d(yPred1d,yTrue2d[i]));
        return lossValue1d.reduce((a,b) => a+b) / lossValue1d.length;
    }
    _backward1d(yPred1d,yTrue1d) {
        return yPred1d.map((yPred,i) => (yTrue1d[i]==1.) ? -1/yPred : 1/(1-yPred));
    }
    backward() {
        const yTrue2d=this.yTrue2d;
        this.grad=this.yPred2d.map((yPred1d,i) => this._backward1d(yPred1d,yTrue2d[i]));
        return this.grad;
    }
}

/**
*/
class Sigmoid {
    forward(x2d) {
        this.results=x2d.map(x1d => x1d.map(x => 1./(1.+exp(-x))));
        return this.results;
    }
    backward(gradients) {
        // `s * (1.-s)` calculates sigmoid grad, then we chain gradients passed in
        this.grad=this.results.map((result,i) => result.map((s,j) => s * (1.-s) * gradients[i][j]));
        return this.grad;
    }
}

/**
*/
class ReLU {
    forward(x2d) {
        this.gradMask=zeros(...shape(x2d));
        return x2d.map((x1d,rowIndex) => x1d.map((x,colIndex) => {
            if (x>0) {
                this.gradMask[rowIndex][colIndex]=1;
            }
            return Math.max(0,x);
        }));
    }
    backward(gradient) {
        return matrixMultiply2d(this.gradMask,gradient);
    }
}

/**
Applies a linear transformation to `x`.
*/
class Linear {
    constructor(inputDim,numHidden=1,bias=true) {
        this.inputDim=inputDim;
        this.numHidden=numHidden;
        // Kaiming Init
        this.weights=matrixMultiply2d(randn(inputDim,numHidden), Math.sqrt(2.0/inputDim));
        this.bias=zeros(numHidden)
        this.updateBias=bias;
    }
    forward(x) {
        this.x=x; // shape(bs,inputDim)
        return matrixSum2d(dotProduct(x,this.weights), this.bias);
    }
    backward(gradient) { // gradient shape(bs,numHidden)
        // weightsGradient/biasGradient need to be the same shape as weights/bias
        this.weightsGradient=dotProduct(transpose(this.x), gradient);
        // this.biasGradient=gradient.sum(axis=0)
        this.biasGradient=transpose(gradient).map(col => col.reduce((a,b) => a+b));
        this.xGradient=dotProduct(gradient,transpose(this.weights));
        return this.xGradient;
    }
    update(lr) {
        // gradient calculations in backward don't account for batch size, so we do it here
        lr=lr/this.x.length; // TODO: change gradient calc to account for batch size - all XxxLoss classes
        this.weights=matrixSubtract2d(this.weights,matrixMultiply2d(this.weightsGradient,lr));
        if (this.updateBias) {
            this.bias=matrixSubtract1d(this.bias,matrixMultiply1d(this.biasGradient,lr));
        }
    }
}

/**
Using
- `Embedding` when `x` is an array of IDs or
- `Linear` when `x` is a one-hot encoded matrix
should give the same results - but `Embedding` should be faster.
*/
class Embedding extends Linear {
    constructor(inputDim,numHidden=1,bias=true) {
        super(inputDim,numHidden,bias);
        this.weights=uniform(inputDim,numHidden,-1,1);
    }
    forward(x) {
        this.x=x;
        return matrixSum2d(x.map(i=>this.weights[i]), this.bias);
    }
    backward(gradient) { // gradient shape(bs,numHidden)
        this.weightsGradient=zeros(this.inputDim,this.numHidden);
        for (let i=0; i<this.inputDim; i++) {
            this.x.map((row, rowIndex)=>{
                if (row == i) {
                    this.weightsGradient[i]=matrixSum1d(this.weightsGradient[i],gradient[rowIndex]);
                }
            })
        }
        this.biasGradient=transpose(gradient).map(col => col.reduce((a,b) => a+b));
        this.xGradient=dotProduct(gradient,transpose(this.weights));
        return this.xGradient;
    }
    update(lr) {
        // gradient calculations in backward don't account for batch size, so we do it here
        lr=lr/this.x.length; // TODO: change gradient calc to account for batch size - all XxxLoss classes
        this.weights=matrixSubtract2d(this.weights,matrixMultiply2d(this.weightsGradient,lr));
        if (this.updateBias) {
            this.bias=matrixSubtract1d(this.bias,matrixMultiply1d(this.biasGradient,lr));
        }
    }
}

/**
*/
class Learner {
    constructor(model, lossFn, data, metrics=[accuracy]) {
        this.model=model;
        this.lossFn=lossFn;
        this.metrics=metrics;
        const splitData=split(shuffle(data));
        this.xTrain=splitData[0][0];
        this.xValid=splitData[0][1];
        this.yTrain=splitData[1][0];
        this.yValid=splitData[1][1];
        // shame that we can destructure into this. )o:
//         [[this.xTrain,this.xValid],[this.yTrain,this.yValid]]=split(data);
    }
    forward(x) {
        for (let i=0; i<this.model.length; i++) {
            x=this.model[i].forward(x);
        }
        return x;
    }
    backward(gradients) {
        for (let i=this.model.length-1; i>=0; i--) {
            gradients=this.model[i].backward(gradients);
        }
        return gradients;
    }
    step(lr) {
        this.model.forEach(m => {
            if (typeof m.update=='function') {
                m.update(lr);
            }
        });
    }
    validate(epoch) {
        const preds=this.forward(this.xValid);
        const lossValue=this.lossFn.forward(preds,this.yValid);
        const metricValues=this.metrics.map(metric=>metric(preds,this.yValid));
        console.log('epoch',epoch,'valid loss',lossValue,'metrics',metricValues);
    }
    fit(epochs, lr=0.1, bs=64) {
        this.validate(-1); // Note: we use epoch -1 to indicate before training
        for (let epoch=0; epoch<epochs; epoch++) {
            batches([this.xTrain,this.yTrain]).forEach(batch => {
                const [xb,yb]=batch;
                const preds=this.forward(xb);
                const lossValue=this.lossFn.forward(preds,yb);
                this.lossFn.backward();
                this.backward(this.lossFn.grad);
                this.step(lr);
            });
            this.validate(epoch);
        }
    }
    predict(x,y,yToLabelFn=(a=>a)) {
        const preds=this.forward(x);
        return preds.map((pred,rowIndex) => {
            const row=[pred,yToLabelFn(pred)];
            if (y!=null) {
                row.push(yToLabelFn(y[rowIndex]));
            }
            return row;
        });
    }
}

export {accuracy,Sigmoid,MSE,BinaryCrossEntropyLoss,CrossEntropyLoss,ReLU,Linear,Embedding,Learner}

