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
Returns the dot product of two 2d arrays.
*/
function dotProduct(a,b) {
    const bTransposed=transpose(b);
    const result=[];
    a.forEach(function(aRow,aRowIndex) {
        result[aRowIndex]=[];
        bTransposed.forEach(function(bRow) {
            const mults=[];
            aRow.forEach(function(aElem,aColumnIndex) {
                const bElem=bRow[aColumnIndex];
                mults.push(aElem*bElem);
            });
            result[aRowIndex].push(mults.reduce((a, b) => a + b, 0));
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
Return a 1d or 2d array of zeros.
*/
function zeros(d0, d1) {
    if (d1 == null) {
        return new Array(d0).fill(0);
    }
    const result=[];
    for (let i=0; i<d0; i++) {
        result.push(new Array(d1).fill(0));
    }
    return result;
}

/**
Elementwise sum of a 2d and a 1d matrix.
`shape(a2d)[1]` must equal `shape(b1d)`.
*/
function matrixSum(a2d,b1d) {
    return a2d.map(row => row.map((e, i) => e+b1d[i]));
}

/**
Element wise subtraction of `b` from `a`, where a and b are 1d.
*/
function matrixSubtract1d(a,b) {
    return a.map((e,i) => e-b[i]);
}

/**
Element wise subtraction of `b` from `a`, where a and b are 2d.
*/
function matrixSubtract2d(a,b) {
    return a.map((row,i) => matrixSubtract1d(row,b[i]));
}

/**
Return 1d array multiplied by a scalar value.
*/
function matrixMultiply1d(m,scalar) {
    return m.map(e => e*scalar);
}

/**
Return 2d array multiplied by a scalar value.
*/
function matrixMultiply2d(m,scalar) {
    return m.map(row => matrixMultiply1d(row,scalar));
}

/**
Return the index of the highest value in `a`.
*/
function argmax(a) {
    return a.indexOf(Math.max(...a));
}

export {
    shape,transpose,dotProduct,randn,zeros,
    matrixSum,matrixSubtract1d,matrixSubtract2d,matrixMultiply1d,matrixMultiply2d,argmax}

