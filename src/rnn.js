/**
Convert a 1d array of numbers (sequence of word indices) to a 2d array of shape [sequenceLength+1, nums.length/sequenceLength].
This makes it easy to iterate over the 1st dimension of "data" to access a chunk of "nums", one timestep at a time.
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
*/
class ReLU {
    constructor() {
        this.gradMasks=[];
    }
    forward(x2d) {
        const gradMask=zeros(...shape(x2d));
        this.gradMasks.push(gradMask);
        return x2d.map((x1d,rowIndex) => x1d.map((x,colIndex) => {
            if (x>0) {
                gradMask[rowIndex][colIndex]=1;
            }
            return Math.max(0,x);
        }));
    }
    backward(gradient) {
        if (this.gradMasks.length <= 0) {
            throw `ReLU: backward has been called too many times`;
        }
        return matrixMultiply2d(this.gradMasks.pop(),gradient);
    }
    update(lr) {
        if (this.gradMasks.length != 0) {
            throw new Error(`ReLU: forward has been called ${this.gradMasks.length} times more than backward`);
        }
    }
}

/**
*/
function _matrixSum2d(a,b) {
    return (b == null) ? a : matrixSum2d(a,b);
}
function _matrixSum1d(a,b) {
    return (b == null) ? a : matrixSum1d(a,b);
}

/**
Applies a linear transformation to `x`.
*/
class Linear {
    constructor(inputDim,numHidden=1,bias=true) {
        this.inputDim=inputDim;
        this.numHidden=numHidden;
        this.weights=matrixMultiply2d(randn(inputDim,numHidden), Math.sqrt(2.0/inputDim));
        this.bias=zeros(numHidden)
        this.updateBias=bias;
        this.xHistory=[];
        this.weightsGradient=null;
        this.biasGradient=null;
        this.label=`${this.constructor.name}(${this.inputDim},${this.numHidden})`;
    }
    forward(x) {
        this.xHistory.push(x);
        return matrixSum2d(dotProduct(x,this.weights), this.bias);
    }
    backward(gradient) {
        if (this.xHistory.length <= 0) {
            throw `${this.label}: backward has been called too many times`;
        }
        let weightsGradient=dotProduct(transpose(this.xHistory.pop()), gradient);
        this.weightsGradient=_matrixSum2d(weightsGradient,this.weightsGradient);
        let biasGradient=transpose(gradient).map(col => col.reduce((a,b) => a+b));
        this.biasGradient=_matrixSum1d(biasGradient,this.biasGradient);
        return dotProduct(gradient,transpose(this.weights)); // xGradient
    }
    update(lr) {
        if (this.xHistory.length != 0) {
            throw new Error(`${this.label}: forward has been called ${this.xHistory.length} times more than backward`);
        }
        this.weights=matrixSubtract2d(this.weights,matrixMultiply2d(this.weightsGradient,lr));
        if (this.updateBias) {
            this.bias=matrixSubtract1d(this.bias,matrixMultiply1d(this.biasGradient,lr));
        }
        this.weightsGradient=null;
        this.biasGradient=null;
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
        this.xHistory.push(x);
        return matrixSum2d(x.map(i=>this.weights[i]), this.bias);
    }
    backward(gradient) {
        if (this.xHistory.length <= 0) {
            throw `${this.label}: backward has been called too many times`;
        }
        let weightsGradient=zeros(this.inputDim,this.numHidden);
        let x=this.xHistory.pop();
        for (let i=0; i<this.inputDim; i++) {
            x.map((row, rowIndex)=>{
                if (row == i) {
                    weightsGradient[i]=matrixSum1d(weightsGradient[i],gradient[rowIndex]);
                }
            })
        }
        this.weightsGradient=_matrixSum2d(weightsGradient,this.weightsGradient);
        let biasGradient=transpose(gradient).map(col => col.reduce((a,b) => a+b));
        this.biasGradient=_matrixSum1d(biasGradient,this.biasGradient);
        return dotProduct(gradient,transpose(this.weights));
    }
}

/**
*/
function groupChunks(ds,bs=64) {
    const m = Math.floor(ds[0].length/bs);
    const newDs = [...Array(ds.length).keys()].map(i=>[]);
    for (let i=0; i<m; i++) {
        for (let j=0; j<bs; j++) {
            for (let k=0; k<ds.length; k++) {
                newDs[k].push(ds[k][i + m*j]);
            }
        }
    }
    return newDs;
}

/**
*/
class Flatten {
    forward(x) {
        this.originalShape=shape(x);
        return [].concat(...x);
    }
    backward(x) {
        const result=[];
        for (let i=0; i<this.originalShape[0]; i++) {
            const startFrom=i*this.originalShape[1];
            result.push(x.slice(startFrom,startFrom+this.originalShape[1]));
        }
        return result;
    }
}

