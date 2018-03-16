const FS = require('fs');

const samples = 32;
let data = [];
let outstanding = [];

function shuffle(array, times) {
    let temp = null;
    let r = 0;
    while (times--) {
        for (let i in array) {
            r = (Math.random() * array.length) << 0;
            temp = array[r];
            array[r] = array[i];
            array[i] = temp;
        }
    }
    return array;
}

function prepare() {
    outstanding = Object.keys(data);
    return shuffle(outstanding, 1000);
}

// load in all the training batches
let origPoolSize = Buffer.poolSize;
Buffer.poolSize = 900;

for (let i = 0; i < 86; i++) {
    data.push(Float32Array(FS.readFileSync('./data/' + i, 'wb')));
}
Buffer.poolSize = origPoolSize;

module.exports = {
    getBatch: function(size = 32) {
        if (outstanding.length === 0) prepare();
        return data[ outstanding.pop() ];
    }
};