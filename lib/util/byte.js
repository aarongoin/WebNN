// clamp float to interval: [-1, 1]
function clampN(float) {
    return Math.max( Math.min(float, 1), -1);
}

// clamp float to interval: [0, 1]
function clampP(float) {
    return Math.max(Math.min(float, 1), 0);
}

// maps floats on interval [-1, 1) to bytes on interval [-128, 127] with max error: 0.0078125
// behavior undefined if input does not conform to given interval
function byteFromFloat(float) {
    return float * 128 << 0;
}
// maps bytes on interval [-128, 127] to floats on interval [-1, 1)
// behavior undefined if input does not conform to given interval
function floatFromByte(byte) {
    return byte * 0.0078125;
}

function byteFromFloatArray(array) {
    let result = new Int8Array(array.length);
    for (let i in array)
        result[i] = clampN(array[i]) * 128;
}
function floatFromByteArray(array) {
    let result = new Float32Array(array.length);
    for (let i in array)
        result[i] = array[i] * 0.0078125;
}

// maps floats on interval [0, 1] to bytes on interval [0, 255] with max error: 0.003921568627451
// behavior undefined if input does not conform to given interval
function uByteFromFloat(float) {
    return float * 255 << 0;
}
// maps bytes on interval [0, 255] to floats on interval [0, 1]
// behavior undefined if input does not conform to given interval
function floatFromUByte(byte) {
    return byte * 0.003921568627451;
}

function uByteFromFloatArray(array) {
    let result = new Uint8Array(array.length);
    for (let i in array)
        result[i] = clampP(array[i]) * 255;
}

function floatFromUByteArray(array) {
    let result = new Float32Array(array.length);
    for (let i in array)
        result[i] = array[i] * 0.003921568627451;
}

module.exports = {
    clampN,
    clampP,

    byteFromFloat,
    floatFromByte,
    byteFromFloatArray,
    floatFromByteArray,

    uByteFromFloat,
    floatFromUByte,
    uByteFromFloatArray,
    floatFromUByteArray
};