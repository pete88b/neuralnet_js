/**
Simple "assert" function that can compare primitives and arrays.
*/
function testEq(expected,actual) {
    if (Array.isArray(expected)) {
        expected=JSON.stringify(expected);
        actual=JSON.stringify(actual);
    }
    if (expected!==actual) {
        throw Error(`Expected ${expected} but found ${actual}`);
    }
}

