window.TF = require("../../../node_modules/tensorfire/src/index");
window.GL = TF.createGL();
const
Model = require("../../nn/Model"),
Pako = require('pako'),
Bytes = require('../../util/byte');

// initialize model
var run = true,
    byte_weights,
    byte_data,
    net,
    model;


// execution sequence:
// 1. getWeights()
// 2. onWeights()
// 3. getData()
// 4. onData()
// 5. onTrain()
// 6. updateWeights()
// 2. onWeights()
// 3. ...

function getData() {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status === 200) {
			onData(r.response);
		}
	};
	r.open("GET", './data');
	r.responseType = 'arraybuffer';
	r.send();
}

function getWeights() {
    var r = new XMLHttpRequest();
    r.onreadystatechange = function () {
        if (r.readyState === XMLHttpRequest.DONE && r.status === 200) {
            onWeights(r.response);
        }
    }
    r.open("GET", './update');
    r.responseType = 'arraybuffer';
    r.send();
}

function updateWeights(weights) {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status === 200) {
			onWeights(r.response);
		}
	}
	r.open("PUT", './update');
	r.responseType = 'arraybuffer';
	r.setRequestHeader("Content-Type", 'arraybuffer');
	r.send(weights);
}

// load or merge in new weights
function onWeights(arraybuffer) {

    getData();

    let unzipped = Pako.deflate(new Uint8Array(arraybuffer.slice(4))).buffer;
    let weights = byte_weights ?
        Bytes.floatFromByteArray(new Int8Array(unzipped))
    :
        new Float32Array(unzipped);

    if (model === undefined)
        model = new Model(net, weights);
    else {
        model.merge(weights);
    }
}

// train model on data
function onData(arraybuffer) {
    let learning_rate = new Float32Array(arraybuffer, 0, 1)[0];
    arraybuffer = Pako.deflate(new Uint8Array(arraybuffer.slice(4))).buffer;
    let length = new Float32Array(arraybuffer, 0, 1)[0] * model.input.layer.input;

    let data = null;
    if (byte_data)
        data = Bytes.floatFromByteArray(new Int8Array(arraybuffer, 4));
    else
        data = new Float32Array(arraybuffer, 4);

    model.train(data.subarray(0, length), data.subarray(length), learning_rate, onTrain);
}

// send updated weights back
function onTrain(weights, loss) {

    let bytes = weights;
    if (byte_weights)
        bytes = Bytes.byteFromFloatArray( new Float32Array(weights) ).buffer;

    let zipped = Pako.gzip( new Uint8Array(bytes) ).buffer;

    updateWeights(zipped);
}

// get weights for the first time
getWeights();