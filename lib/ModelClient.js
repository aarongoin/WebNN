const Model = require("./Model");

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

const getModel = (onDone) => request('GET', './model', 'application/json', (data) => onDone(JSON.parse(data)));
const getTrainingData = (onData) => request('GET', './data', 'arraybuffer', onData);
const getValidation = (onValidate) => request('GET', './validate', 'arraybuffer', onValidate);
const getWeights = (onWeights) => request('GET', './weights', 'arraybuffer', onWeights);
const updateWeights = (weights, onWeights) => request('POST', './weights', 'arraybuffer', onWeights, weights);
const sendMetrics = (loss, accuracy, isValidation, callback) => {
    let metrics = new Float32Array([loss, accuracy]);
    request('POST', isValidation ? './validate' : './metrics', 'arraybuffer', callback, metrics.buffer );
};

let MODEL_SINGLETON;

/**
 * This is the ModelClient class
 */
class ModelClient {
    /**
     * Creates a new model, queries server for model description and weights, and executes onDone when model is ready to use.
     * @param {Function} onReady - Callback for when model is loaded and ready
     */
    constructor(onReady) {
        if (MODEL_SINGLETON) {
            if (MODEL_SINGLETON.isReady) onReady(MODEL_SINGLETON);
            else MODEL_SINGLETON._onReady = onReady;
        } else {
            MODEL_SINGLETON = this;

            this._onReady = onReady;
            this.isReady = false;
            getModel((model) => {
                getWeights((arraybuffer) => {
                    // pull out steps (times trained by a client) for these weights
                    let steps = new Float32Array(arraybuffer, 0, 1)[0];

                    if (steps >= 0) {
                        let accuracy = new Float32Array(arraybuffer, 4, 1)[0];
                        let weights = new Float32Array(arraybuffer, 8);

                        this.name = model.name;

                        this.model = new Model(model.net, weights, steps, model.learningRate);
                        this.isReady = true;
                        if (typeof onReady === "function") onReady(MODEL_SINGLETON);
                    }
                })
            })
            window.onbeforeunload = function() {
                request('POST', './model', 'arraybuffer', undefined);
            }

        }
        return MODEL_SINGLETON;
    }

    /**
     * Train model, send updated weights to server, and execute onDone callback with loss and accuracy.
     * @param {Float32Array} input - Features input to model
     * @param {Float32Array} expect - Expected model output
     * @param {Integer} samples - Number of samples in input
     * @param {Function} onDone 
     */
    train(input, expect, samples, onDone) {
        this.model.train(input, expect, samples, (weights, trainingLoss, trainingAccuracy) => {

            // prepare weights to be sent back
            let temp = new DataView(new ArrayBuffer(weights.byteLength + 4));
            temp.setFloat32(0, trainingAccuracy);
            temp = new Uint8Array(temp.buffer);
            temp.set(weights, 4);

            // send weights to server and handle response
            updateWeights(temp, (arraybuffer) => {
                let weights;

                // pull out steps (times trained by a client) for these weights
                let steps = new Float32Array(arraybuffer, 0, 1)[0];
                let accuracy = null;

                // client must merge weights from peer
                if (steps >= 0) {
                    accuracy = new Float32Array(arraybuffer, 4, 1)[0];
                    weights = new Float32Array(arraybuffer, 8);

                    this.model.weightedMerge(weights, steps, accuracy);
                    onDone(trainingLoss, trainingAccuracy);

                // client must validate current weights
                } else if (steps == -1) {
                    getValidation((arraybuffer) => {
                        let data = this.model.readDataBatch(arraybuffer);
                        this.model.validate(data.x, data.y, data.n, (loss, accuracy) => {
                            sendMetrics(loss, accuracy, true);
                            onDone(trainingLoss, trainingAccuracy);
                        });
                    });
                } else onDone(trainingLoss, trainingAccuracy);
            });
            sendMetrics(trainingLoss, trainingAccuracy, false, this.model.updateLearningRate);
        });
    }

     /**
     * Run model on given input and execute onDone callback with model output.
     * @param {Float32Array} input - Features input to model
     * @param {Integer} samples - Number of samples in input
     * @param {Function} onDone
     */
    run(input, samples, onDone) {
        onDone( this.model.run(input, samples) );
    }

    getData(onDone) {
        getTrainingData((arraybuffer) => {
            let data = this.model.readDataBatch(arraybuffer);
            onDone(data);
        })
    }
}

module.exports = ModelClient;