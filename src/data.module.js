/**
Imports we need in data.module.js
*/
import {argmax} from './util.module.js';

/**
Log the first `rows` of an array.
*/
function head(data,rows=10) {
    rows=Math.min(rows,data.length);
    for (let i=0; i<rows; i++) {
        console.log(i, data[i]);
    }
}

/**
Log the last `rows` of an array.
*/
function tail(data,rows=10) {
    rows=Math.min(rows,data.length);
    for (let i=-rows; i<0; i++) {
        console.log(i, data[data.length+i]);
    }
}

/**
Parse simple csv formatted strings.
*/
class RowHandler {
    constructor() {
        this.result=[]
    }
    handleRow(row,i) {
        this.result.push(row.split(','));
    }
}

function parseCsv(stringData, rowHandler, rowLimit) {
    if (rowHandler == null) {
        rowHandler = new RowHandler()
    }
    
    const rows=stringData.split('\n');
    if (rowLimit==null) {
        rowLimit=rows.length;
    }
    for (let i=0; i<rowLimit; i++) {
        const row=rows[i];
        if (row !== '') {
            rowHandler.handleRow(row);
        }
    }
    return rowHandler;
}

/**
Convert a row of the iris dataset from string values to numbers (for input features) targets.
*/
const IRIS_CLASS_MAP = {
    0: 'Iris-setosa',
    'Iris-setosa-onehot': [1,0,0],
    'Iris-setosa-classid': 0,
    1: 'Iris-versicolor',
    'Iris-versicolor-onehot': [0,1,0],
    'Iris-versicolor-classid': 1,
    2: 'Iris-virginica',
    'Iris-virginica-onehot': [0,0,1],
    'Iris-virginica-classid': 2
};
class IrisRowHandler {
    constructor(targetType) {
        this.targetType = (targetType==null) ? 'onehot' : targetType;
        this.result=[[],[]];
    }
    normalize(row) {
        return [
            (row[0]-5.843333333)/0.828066128,
            (row[1]-3.054)/0.433594311,
            (row[2]-3.758666667)/1.76442042,
            (row[3]-1.198666667)/0.763160742
        ];
    }
    handleRow(row) {
        row = row.split(',');
        // convert datatypes and normalize input features
        this.result[0].push(this.normalize(row.slice(0,4).map(a=>parseFloat(a))));
        this.result[1].push(IRIS_CLASS_MAP[`${row[4]}-${this.targetType}`]);
    }
}

/**
Shuffle any number of arrays in the same way.
*/
function shuffle(arrays) {
    var m = arrays[0].length, t, i;
    // While there remain elements to shuffle…
    while (m) {
        // Pick a remaining element…
        i = Math.floor(Math.random() * m--);
        // And swap it with the current element.
        arrays.forEach(array => {
            t = array[m];
            array[m] = array[i];
            array[i] = t;
        });
    }
    return arrays;
}

/**
Split any number of arrays returning [100-`percent`, `percent`] for each array.
*/
function split(arrays, percent=0.2) {
    const result=[];
    arrays.forEach(array => {
        const splitPos=Math.round(arrays[0].length*(1.0-percent));
        result.push([array.slice(0,splitPos), array.slice(splitPos)]);
    });
    return result;
}

/**
Shuffle any number of arrays then put them into an array of batches.
*/
function batches(arrays, bs=64) {
    shuffle(arrays);
    const result=[];
    for (let i=0; i<(arrays[0].length/bs); i++) {
        const batch=[];
        result.push(batch);
        arrays.forEach(array=>batch.push(array.slice(bs*i,bs*(i+1))))
    }
    return result;
}

export {head,tail,parseCsv,IRIS_CLASS_MAP,RowHandler,IrisRowHandler,shuffle,split,batches}

