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

