/**
Convert a 1d array of numbers (sequence of word indices) to a 2d array of shape [sequenceLength+1, nums.length/sequenceLength].
This makes it easy to iterate over the 1st dimention of "data" to access a chunk of "nums", one timestep at a time.
*/
function toData(nums,sequenceLength) {
    const data=full(sequenceLength+1).map(e=>[]);
    const iMax=nums.length-sequenceLength;
    for (let i=0; i<iMax; i+=sequenceLength) {
        for (let j=0; j<sequenceLength+1; j++) {
            data[j].push(nums[i+j]);
        }
    }
    return data;
}

/**
A layer that can wrap a Linear or Embedding so that it can be called multiple times during a forward pass.
*/
class MultiCallLayer {
    constructor(layer) {
        this.layer=layer;
        this.xHistory=[];
        this.weightsGradients=null;
        this.biasGradients=null;
    }
    forward(x) {
        this.xHistory.push(x);
        return this.layer.forward(x);
    }
    _matrixSum2d(a,b) {
        return (b == null) ? a : matrixSum2d(a,b);
    }
    _matrixSum1d(a,b) {
        return (b == null) ? a : matrixSum1d(a,b);
    }
    backward(gradient) {
        if (this.xHistory.length == 0) {
            throw `this.xHistory is empty`;
        }
        this.x=this.xHistory.pop();
        this.layer.backward(gradient);
        this.weightsGradients=this._matrixSum2d(this.layer.weightsGradient,this.weightsGradients);
        this.biasGradients=this._matrixSum1d(this.layer.biasGradient,this.biasGradients);
        // Note: we're not keeping x gradients
        return this.layer.xGradient;
    }
    update(lr) {
        if (this.xHistory.length != 0) {
            throw `forward has been called ${this.xHistory.length} times more than backward`;
        }
        this.layer.weightsGradient=this.weightsGradients;
        this.layer.biasGradient=this.biasGradients;
        this.layer.update(lr);
        this.weightsGradients=null;
        this.biasGradients=null;
    }
}

