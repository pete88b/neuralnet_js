/**
*/
class BinaryCrossEntropyLoss {
    _forward1d(yPred1d,yTrue1d) {
        const temp=[];
        yPred1d.forEach(function (yPred, i) {
            let tempValue=yPred;
            tempValue=(yTrue1d[i]==1.) ? tempValue : 1-tempValue;
            tempValue=Math.log(tempValue);
            temp.push(tempValue);
        });
        return -temp.reduce((a,b) => a+b) / temp.length;
    }
    forward(yPred2d,yTrue2d) {
        this.yPred2d=yPred2d;
        this.yTrue2d=yTrue2d;
        const lossValue1d=yPred2d.map((yPred1d,i) => this._forward1d(yPred1d,yTrue2d[i]));
        return lossValue1d.reduce((a,b) => a+b) / lossValue1d.length;
    }
    _backward1d(yPred1d,yTrue1d) {
        const temp=[];
        yPred1d.forEach(function (yPred, i) { // TODO: rewrite with map
            let tempValue=(yTrue1d[i]==1.) ? -1/yPred : 1/(1-yPred);
            temp.push(tempValue);
        });
        return temp;
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
        this.results=x2d.map(x1d => x1d.map(x => 1./(1.+Math.pow(Math.E, -x))));
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
            return Math.max(0,x)
        }));
    }
    matrixMultiply(a2d, b2d) {
        return a2d.map((a1d,rowIndex) => a1d.map((a,colIndex) => a*b2d[rowIndex][colIndex]));
    }
    backward(gradient) {
        return this.matrixMultiply(this.gradMask,gradient);
    }
}

/**
*/
class Linear {
    constructor(inputDim,numHidden=1) {
        this.inputDim=inputDim;
        this.numHidden=numHidden;
        // Kaiming Init
        this.weights=matrixMultiply2d(randn(inputDim,numHidden), Math.sqrt(2.0/inputDim));
        this.bias=zeros(numHidden)
    }
    forward(x) {
        this.x=x; // shape(bs,inputDim)
        return matrixSum(dotProduct(x,this.weights), this.bias);
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
        lr=lr/this.x.length;
        this.weights=matrixSubtract2d(this.weights,matrixMultiply2d(this.weightsGradient,lr));
        this.bias=matrixSubtract1d(this.bias,matrixMultiply1d(this.biasGradient,lr));
    }
}

/**
*/
class Learner {
    constructor(model, lossFn, data) {
        this.model=model;
        this.lossFn=lossFn;
        this.x=data[0];
        this.y=data[1];
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
    fit(epochs, lr=0.1) {
        for (let epoch=0; epoch<epochs; epoch++) {
            const preds=this.forward(this.x);
            const lossValue=this.lossFn.forward(preds, this.y);
            console.log('epoch',epoch,'lossValue',lossValue);
            this.lossFn.backward();
            this.backward(this.lossFn.grad);
            this.step(lr);
        }
    }
}

export {Sigmoid, BinaryCrossEntropyLoss, Linear}

