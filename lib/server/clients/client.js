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
        r.send(weights);
    } else {
        r.send();
    }
}
const getData = () => request('GET', './data', 'arraybuffer', onData);
const getWeights = () => request('GET', './update', 'arraybuffer', onWeights);
const updateWeights = weights => request('PUT', './update', 'arraybuffer', onWeights, weights);
const sendMetrics = metrics => request('POST', './metrics', 'arraybuffer', undefined, metrics);

// load or merge in new weights
function onWeights(arraybuffer) {

    getData();

    let unzipped = Pako.inflateRaw(new Uint8Array(arraybuffer)).buffer;
    let weights = new Float32Array(unzipped);

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
    model.train(data.x, data.y, data.n, onTrain);
}

// send updated weights back
function onTrain(weights, loss) {
    
    let zipped = Pako.deflateRaw(new Uint8Array(weights.buffer) ).buffer;
    // let lossWrapper = new Float32Array(1);
    // lossWrapper[0] = loss;

    // console.log(loss);
    // console.log(weights);
    updateWeights(zipped);
    // sendLoss(lossWrapper.buffer);
}

// get weights for the first time
getWeights();
