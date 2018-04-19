window.TF = require("../../node_modules/tensorfire/src/index");
window.GL = TF.createGL();
const
Model = require("../Model"),
Pako = require('pako'),
Bytes = require('../util/byte');

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

function request(reqType, url, dataType, callback, payload) {
    var r = new XMLHttpRequest();
    r.open(reqType, url);
    if (typeof callback === 'function') {
        r.onreadystatechange = function () {
            if (r.readyState === XMLHttpRequest.DONE && r.status === 200) {
                callback(r.response);
            }
        }
        r.responseType = dataType;
    }
    r.setRequestHeader("client-request-date", new Date().toUTCString());
    if (payload !== undefined) {
        r.setRequestHeader("Content-Type", dataType);
        r.send(payload);
    } else {
        r.send();
    }
}
const getData = () => request('GET', './data', 'arraybuffer', onData);
const getValidation = () => request('GET', './validate', 'arraybuffer', onValidate);

const getWeights = () => request('GET', './update', 'arraybuffer', onWeights);
const updateWeights = weights => request('PUT', './update', 'arraybuffer', onWeights, weights);

const sendMetrics = (loss, accuracy, isValidation, callback, ) => {
    let metrics = new Float32Array([loss, accuracy]);
    request('PUT', isValidation ? './validate' : './metrics', 'arraybuffer', callback, metrics.buffer );
};

// load or merge in new weights
function onWeights(arraybuffer) {
    let unzipped, weights;

    // pull out steps (times trained by a client) for these weights
    let steps = new Float32Array(arraybuffer, 0, 1)[0];
    let accuracy = null;

    if (steps > -1) {
        accuracy = new Float32Array(arraybuffer, 4, 1)[0];
        // unzip the weights
        unzipped = Pako.inflateRaw(new Uint8Array(arraybuffer, 8)).buffer;
        weights = new Float32Array(unzipped);
    }

    if (steps > 0) {
        getData();
        if (model === undefined)
            model = new Model(net, weights, steps, learning_rate);
        else {
            // model.weightedMerge(weights, steps, accuracy);
            model.merge(weights, steps, accuracy);
            // model.mimicMerge(weights, steps, accuracy);
            // model.copyMerge(weights, steps, accuracy);
        }
    
    } else if (steps > -2) {
        // console.log('Client will validate...');
        getValidation();
        if (steps === 0)
            model.load(weights);
    
            // step can equal -2, which means do nothing with the weights, and continue to train on data
    } else getData();
    
}

function onValidate(arraybuffer) {
    arraybuffer = Pako.inflateRaw(new Uint8Array(arraybuffer)).buffer;
    let data = model.readDataBatch(arraybuffer);
    model.validate(data.x, data.y, data.n, (loss, accuracy) => {
        console.log("loss: " + loss + " accuracy: " + accuracy);
        sendMetrics(loss, accuracy, true);
        getData();
    });
}

// train model on data
function onData(arraybuffer) {
    arraybuffer = Pako.inflateRaw( new Uint8Array(arraybuffer) ).buffer;
    let data = model.readDataBatch(arraybuffer);
    model.train(data.x, data.y, data.n, onTrain);
}

// send updated weights back
function onTrain(weights, loss, accuracy) {
    let zipped = Pako.deflateRaw(new Uint8Array(weights.buffer) );

    let temp = new DataView(new ArrayBuffer(zipped.byteLength + 4));
    temp.setFloat32(0, accuracy);
    temp = new Uint8Array(temp.buffer);
    temp.set(zipped, 4);

    updateWeights(temp);
    sendMetrics(loss, accuracy, false, model.updateLearningRate);
}

// get weights for the first time
getWeights();


window.onbeforeunload = function() {
    request('PUT', './client', 'arraybuffer', undefined);
}