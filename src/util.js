/**
Round `x` (or all elements of `x`) to `dp` decimal places.
*/
function round(x,dp) {
    dp = dp || 0;
    if (Array.isArray(x)) {
        return x.map(_x => round(_x,dp));
    }
    return Math.round(x*Math.pow(10,dp))/Math.pow(10,dp);
}

/**
Flatten a 2d array into a 1d array.
*/
function flatten(a2d) {
    return [].concat(...a2d);
}

/**
*/
function exp(a) {
    return Math.pow(Math.E, a);
}

/**
Returns the shape of an "n" dimentional array.
*/
function shape(m) {
    const result=[];
    while (Array.isArray(m)) {
        result.push(m.length);
        m=m[0];
    }
    return result;
}

/**
Returns the mean of all elements in a 2d array.
*/
function mean(matrix) {
    const elementCount=shape(matrix).reduce((a,b)=>a*b);
    const sum=matrix.map(row=>row.reduce((a,b)=>a+b)).reduce((a,b)=>a+b);
    return sum/elementCount;
}

/**
Return a 1d or 2d array of `fillValue`.
*/
function full(d0,d1,fillValue) {
    if (d1 == null) {
        return new Array(d0).fill(fillValue);
    }
    const result=[];
    for (let i=0; i<d0; i++) {
        result.push(new Array(d1).fill(fillValue));
    }
    return result;
}

/**
Return a 1d or 2d array of zeros.
*/
function zeros(d0,d1) {
    return full(d0,d1,0);
}

/**
Return a square array with ones on the main diagonal.
*/
function identity(n) {
    const result=zeros(n,n);
    for (let i=0; i<n; i++) {
        result[i][i]=1;
    }
    return result;
}

/**
Return the mean and population standard deviation of a 1d array.

https://stackoverflow.com/questions/7343890/standard-deviation-javascript
*/
function meanAndStandardDeviation(a1d) {
    const n=a1d.length;
    const mean=a1d.reduce((a,b)=>a+b) / n;
    return [mean,Math.sqrt(a1d.map(a => Math.pow(a - mean, 2)).reduce((a, b) => a + b) / n)];
}

/**
Returns the transpose of a 2d array.
*/
function transpose(matrix) {
    const result = [];
    matrix.forEach(function(row,rowIndex) {
        row.forEach(function(elem,columnIndex) {
            if (rowIndex==0) {
                result[columnIndex]=[elem];
            } else {
                result[columnIndex].push(elem);
            }
        });
    });
    return result;
}

/**
Returns a single value from a standard normal distribution.
*/
function randn_bm() {
    // Box-Muller transform - Max Collard - stack overflow
    var u=0, v=0;
    while(u==0) u=Math.random();
    while(v==0) v=Math.random();
    return Math.sqrt(-2.0*Math.log(u)) * Math.cos(2.0*Math.PI*v);
}

/**
Returns a 2d array filled with `randn_bm` values.
*/
function randn(d0,d1) {
    const result = [];
    for (let rowIndex = 0; rowIndex < d0; rowIndex++) {
        const row=[];
        result.push(row);
        for (let colIndex = 0; colIndex < d1; colIndex++) {
            row.push(randn_bm());
        }
    }
    return result;
}

/**
Return matrix of `newShape` if
- a is a scalar value,
- a is a 1d array with a length that matches newShape[1] or
- a is the new shape already.

`newShape` must be 2d.
*/
function reshape(a,newShape) {
    const oldShape=shape(a);
    if (oldShape.length==0) {
        return full(newShape[0],newShape[1],a);
    } else if (oldShape.length==1 && oldShape[0]==newShape[1]) {
        return new Array(newShape[0]).fill(a);
    }
    newShape.forEach((s,i) => {
        if (s!=oldShape[i]) throw `Can't reshape from [${oldShape}] to [${newShape}]`;
    });
    return a;
}

/**
Elementwise sum of a and b where a and b are 1d.
*/
function matrixSum1d(a,b) {
    return a.map((e,i) => e+b[i]);
}

/**
Elementwise sum of a2d and b, where a2d is 2d and b can be reshaped to match a.
*/
function matrixSum2d(a2d,b) {
    const b2d=reshape(b,shape(a2d));
    return a2d.map((row,i) => matrixSum1d(row, b2d[i]));
}

/**
Element wise subtraction of `b` from `a`, where a and b are 1d.
*/
function matrixSubtract1d(a,b) {
    return a.map((e,i) => e-b[i]);
}

/**
Elementwise subtraction of b from a2d, where a2d is 2d and b can be reshaped to match a.
*/
function matrixSubtract2d(a2d,b) {
    const b2d=reshape(b,shape(a2d));
    return a2d.map((row,i) => matrixSubtract1d(row,b2d[i]));
}

/**
Element wise multiplication of `b` and `a`, where a is 1d and b can be reshaped to match a.
*/
function matrixMultiply1d(a1d,b) {
    const b1d=reshape(b,shape(a1d));
    return a1d.map((e,i) => e*b1d[i]);
}

/**
Elementwise multiplication of a2d with b, where a2d is 2d and b can be reshaped to match a.
*/
function matrixMultiply2d(a2d,b) {
    const b2d=reshape(b,shape(a2d));
    return a2d.map((row,i) => matrixMultiply1d(row,b2d[i]));
}

/**
Returns the dot product of two 2d arrays.
*/
function dotProduct(a,b) {
    const bTransposed=transpose(b);
    return a.map((aRow,aRowIndex) => {
        return bTransposed.map((bRow) => {
            return matrixMultiply1d(aRow,bRow).reduce((a,b) => a+b);
        });         
    });
}

/**
Return the index of the highest value in `a`.
*/
function argmax(a) {
    return a.indexOf(Math.max(...a));
}

/**
Normalize a 2d array by subtracting its mean and dividing by its standard deviation for all elements.
*/
function normalize(a2d) {
    const [mean,std] = meanAndStandardDeviation(flatten(a2d));
    return matrixMultiply2d(matrixSubtract2d(a2d,mean), 1/std);
}

