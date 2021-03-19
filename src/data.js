/**
*/
function head(data,rows=10) {
    rows=Math.min(rows,data.length);
    for (let i=0; i<rows; i++) {
        console.log(i, data[i]);
    }
}

function tail(data,rows=10) {
    rows=Math.min(rows,data.length);
    for (let i=-rows; i<0; i++) {
        console.log(i, data[data.length+i]);
    }
}

/**
Read simple csv files.
*/
const fs = require('fs');

class RowHandler {
    constructor() {
        this.result=[]
    }
    handleRow(row,i) {
        this.result.push(row.split(','));
    }
}

function readCsv(path, rowHandler, rowLimit) {
    if (rowHandler == null) {
        rowHandler = new RowHandler()
    }
    const fileData=fs.readFileSync(path).toString();
    const rows=fileData.split('\n');
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
Convert a row of the iris dataset from string values to numbers for input features and one hot encoded targets.
*/
const IRIS_CLASS_MAP = {
    'Iris-setosa': [1,0,0],
    'Iris-versicolor': [0,1,0],
    'Iris-virginica': [0,0,1]
};
class IrisRowHandler {
    constructor() {
        this.result=[[],[]]
    }
    handleRow(row) {
        row = row.split(',');
        // convert datatypes and normalize input features
        this.result[0].push([
            (parseFloat(row[0])-5.843333333)/0.828066128,
            (parseFloat(row[1])-3.054)/0.433594311,
            (parseFloat(row[2])-3.758666667)/1.76442042,
            (parseFloat(row[3])-1.198666667)/0.763160742
        ]);
        this.result[1].push(IRIS_CLASS_MAP[row[4]])
    }
}

export {head,tail,readCsv,IRIS_CLASS_MAP,RowHandler,IrisRowHandler}

