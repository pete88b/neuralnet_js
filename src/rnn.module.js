/**
Imports we need in rnn.module.js
*/
import {round,flatten,exp,shape,transpose,dotProduct,randn,full,zeros,mean,reshape,argmax} from './util.module.js';
import {normalize,identity,meanAndStandardDeviation} from './util.module.js';
import {matrixSum1d,matrixSum2d,matrixSubtract1d,matrixSubtract2d,matrixMultiply1d,matrixMultiply2d} from './util.module.js';
import {head,tail,parseCsv,IRIS_CLASS_MAP,IrisRowHandler,shuffle,split,batches} from './data.module.js';
import {accuracy,Sigmoid,MSE,ReLU,Linear,Learner} from './nn.module.js';
import {BinaryCrossEntropyLoss,CrossEntropyLoss} from './nn.module.js';

/**
A Linear that can be called multiple times during a forward pass.
*/
class MultiCallLinear extends Linear{
    constructor(inputDim,numHidden=1,bias=true) {
        super(inputDim,numHidden,bias);
        this.xHistory=[];
        this.weightsGradients=null;
        this.biasGradients=null;
    }
    forward(x) {
        this.xHistory.push(x);
        return super.forward(x);
    }
    backward(gradient) {
        if (this.xHistory.length == 0) {
            throw `this.xHistory is empty`;
        }
        this.x=this.xHistory.pop();
        super.backward(gradient);
        this.weightsGradients=(this.weightsGradients == null) 
                ? this.weightsGradient
                : matrixSum2d(this.weightsGradient,this.weightsGradients);
        this.biasGradients=(this.biasGradients == null) 
                ? this.biasGradient
                : matrixSum1d(this.biasGradient,this.biasGradients);
        // Note: we're not keeping x gradients
        return this.xGradient;
    }
    update(lr) {
        if (this.xHistory.length != 0) {
            throw `forward has been called ${this.xHistory.length} times more than backward`;
        }
        super.weightsGradient=this.weightsGradients;
        super.biasGradient=this.biasGradients;
        super.update(lr);
        this.weightsGradients=null;
        this.biasGradients=null;
    }
}

