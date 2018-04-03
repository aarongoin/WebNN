window.TF = require("../../../node_modules/tensorfire/src/index");
window.GL = TF.createGL();
const
Model = require("../../nn/Model"),
Pako = require('pako'),
Bytes = require('../../util/byte');

// initialize model
var run = true,
    byte_data = false,
    byte_weights = false,
    learning_rate = 'learning_rate_here',
    net = 'model_here',
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
    r.setRequestHeader("client-request-date", new Date().toUTCString());
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
    r.setRequestHeader("client-request-date", new Date().toUTCString());
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
    r.setRequestHeader("client-request-date", new Date().toUTCString());
    r.send(weights);
}

function sendLoss(loss) {
    var r = new XMLHttpRequest();
    r.open("POST", './loss');
    r.setRequestHeader("Content-Type", 'arraybuffer');
    r.send(loss);
}

// load or merge in new weights
function onWeights(arraybuffer) {

    getData();

    let unzipped = Pako.inflateRaw(new Uint8Array(arraybuffer)).buffer;
    let weights = byte_weights ?
        Bytes.floatFromByteArray(new Int8Array(unzipped))
    :
        new Float32Array(unzipped);

    if (model === undefined)
        model = new Model(net, weights, learning_rate);
    else {
        model.merge(weights);
    }
}

// train model on data
function onData(arraybuffer) {
    
    arraybuffer = Pako.inflateRaw( new Uint8Array(arraybuffer) ).buffer;
    
    let data = model.readDataBatch(arraybuffer);
    // if (byte_data)
    //     data = Bytes.floatFromByteArray(new Int8Array(arraybuffer, 4));
    // else
    //     data = new Float32Array(arraybuffer, 4);

    

    model.train(data.x, data.y, data.n, onTrain);
}

// send updated weights back
function onTrain(weights, loss) {
    
    let bytes = byte_weights ?
        Bytes.byteFromFloatArray( weights ).buffer
        : weights.buffer;
    
    let zipped = Pako.deflateRaw( new Uint8Array(bytes) ).buffer;
    // let lossWrapper = new Float32Array(1);
    // lossWrapper[0] = loss;

    // console.log(loss);
    // console.log(weights);
    updateWeights(zipped);
    // sendLoss(lossWrapper.buffer);
}

// get weights for the first time
getWeights();
