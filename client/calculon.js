(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
"use strict";

var Model = require("../lib/Model"),
    TF = require("../node_modules/tensorfire/src/index"),
    GL = TF.createGL();

function GET(path, responseType, callback) {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status === 200) {
			callback(r.response);
		}
	};
	r.open("GET", path);
	r.responseType = responseType;
	r.send();
}

function PUT(path, contentType, body, callback) {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status === 200) {
			if (callback) callback(r.response);
		}
	};
	r.open("PUT", path);
	if (callback) r.responseType = contentType;
	r.setRequestHeader("Content-Type", contentType);
	r.send(body);
}

function POST(path, contentType, body) {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status !== 200) {
			// TODO - resend or save to local?
		}
	};
	r.open("POST", path);
	if (contentType !== undefined) r.setRequestHeader("Content-Type", contentType);
	if (body !== undefined) r.send(body);else r.send();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*

	1. Get model from server
	2. Get weights from server
	3. Get data from server
	4. Train and return updates


*/

(function main() {
	var run = true,
	    net,
	    model,
	    iterations,
	    times = {
		requested: null, // model weights and data request sent
		recieved: null, // model data & weights recieved from server
		loaded: null, // model weights loaded
		trained: null, // model finished training
		updated: null // model updates sent
	};

	Model = Model(TF, GL);

	function update(arraybuffer) {

		var iteration = new Float32Array(arraybuffer, 0, 1),
		    view,
		    weights,
		    data,
		    len,
		    i,
		    batch;

		times.recieved = window.performance.now();

		view = new Float32Array(arraybuffer, 4);

		if (iteration[0] >= 0) {
			// includes new weights and data
			iterations = iteration[0];
			i = model.size;
			weights = view.subarray(0, i);
			len = view[i] * net.layers[0].shape[1]; // first float is number of samples in this batch
			len += i + 1;
			batch = {
				x: view.subarray(i, len),
				y: view.subarray(len)
			};

			model.load(weights.buffer);
		} else {
			// weights are fresh, so data only
			iterations++;
			len = view[0] * net.layers[0].shape[1]; // first float is number of samples in this batch
			batch = {
				x: view.subarray(1, ++len),
				y: view.subarray(len)
			};
		}

		// TRAIN
		times.loaded = window.performance.now();
		model.train(net.learning_rate, net.iterations, batch.x, batch.y, function (model) {
			var r = 0,
			    log = "";
			times.trained = window.performance.now();
			//console.log("Time to train: " + (delta / 1000) + " seconds");
			// post results to server

			PUT("./weights/" + net.id, "arraybuffer", model.save(), update);
			r = times.updated = window.performance.now();

			log += net.weights_version + ",";
			log += model.loss + ",";
			log += times.requested + ",";
			log += times.recieved + ",";
			log += times.loaded + ",";
			log += times.trained + ",";
			log += times.updated + "\n";
			// send time and training log to server
			PUT("./log/" + net.id, "text", log);
			times.requested = r;
			net.weights_version++;
		});
	}

	//var server = io();

	// request model to train
	GET("./model", "application/json", function (jsonModel) {
		net = JSON.parse(jsonModel);

		model = new Model(net, null);
		window.onbeforeunload = function () {
			POST("./close/" + net.id, "string");
		};
		times.requested = window.performance.now();
		GET("./weights/" + net.id, "arraybuffer", update);
	});
})();

},{"../lib/Model":3,"../node_modules/tensorfire/src/index":30}],2:[function(require,module,exports){
'use strict';

var Output = require('./layers/Output'),
    Dense = require('./layers/Dense');

module.exports = function (tensorfire, glContext) {
	return {
		"dense": Dense(tensorfire, glContext),
		"output": Output(tensorfire, glContext)
	};
};

},{"./layers/Dense":6,"./layers/Output":8}],3:[function(require,module,exports){
"use strict";

var Layers = require("./Layers");

var Model = function Model(model, layers) {
	this.layers = new Array(model.layers.length);
	this.loss = 0.0;
	this.size = 0.0;
	this.model = model;
	this.load(layers);

	//console.log(JSON.stringify(this.layers[0].save()));
};
Model.prototype.run = function (input) {
	var output = input,
	    l = -1;
	while (++l < this.layers.length) {
		output = this.layers[l].run(output);
	}
};
Model.prototype.forward = function (output) {
	//console.warn("Calculon- Forward pass\n");
	// forward propogation
	var l = -1;
	while (++l < this.layers.length) {
		output = this.layers[l].run(output);
		//console.log("Calculon- output " + l + ": " + output.read().data);
	}
	return output;
};
Model.prototype.backward = function (output, learn) {
	//console.warn("Calculon- Backward pass");
	// backward propogation
	var l = this.layers.length - 1;
	while (l-- > 0) {
		output = this.layers[l].train(output, learn);
	}
};

Model.prototype.validate = function (input, expect, callback) {
	var output = input,
	    lossLayer = this.layers[this.layers.length - 1];
	output = this.forward(output);

	// calculate loss
	output = lossLayer.train(expect);
	if (typeof callback === "function") callback(lossLayer.accuracy);
};

Model.prototype.train = function (learn, iterations, input, expect, callback) {
	var output,
	    e = 0,
	    lossLayer = this.layers[this.layers.length - 1];
	while (e++ < iterations) {
		output = input;
		output = this.forward(output);

		//console.log("Calculon- output: " + output.read().data);
		// calculate loss
		output = lossLayer.train(expect);
		this.loss = lossLayer.accuracy;
		console.log("Accuracy: " + lossLayer.accuracy);

		this.backward(output, learn);

		// chance to send out data from model (metadata and log data)
		if (typeof this.afterIteration === "function") this.afterIteration(this, e);

		//console.warn("Calculon- Iteration: " + e + ", Loss: " + this.loss);
	}
	if (typeof callback === "function") callback(this);
};
Model.prototype.save = function () {
	// TypedArray to hold weights, bias, etc. from every layer of model
	var weights = new Float32Array(this.size);

	var l = -1,
	    o = 0;
	// pull out trained weights for each layer
	while (++l < this.layers.length - 1) {
		weights.set(this.layers[l].save(), o);
		o += this.layers[l].size;
	}
	//console.log("weights: " + weights);
	return weights.buffer;
};
Model.prototype.load = function (layers) {
	// construct layers
	var offset = 0,
	    layer,
	    l = -1;

	this.size = 0;
	if (layers != null) {
		layers = new Float32Array(layers);
	}
	while (++l < this.layers.length - 1) {
		layer = this.model.layers[l];
		layer = new Layers[layer.type](layer, l);
		this.size += layer.size;
		if (layers != null) offset = layer.load(layers, offset);else layer.randomWeights();
		this.layers[l] = layer;
	}
	// initialize output layer
	layer = this.model.layers[l];
	layer = new Layers[layer.type](layer, l);
	this.layers[l] = layer;
};

module.exports = function (tensorfire, glContext) {
	Layers = Layers(tensorfire, glContext);
	return Model;
};

},{"./Layers":2}],4:[function(require,module,exports){
"use strict";

module.exports = {
	Activation: {
		"linear": "o = n; \n",
		//"binary": "if (n > 0.0) { o = 0.0; } else { o = 1.0; } \n",
		"relu": "o = max(0.0, n); \n",
		"lrelu": "if (n > 0.0) { o = n; } else { o = 0.01 * n; } \n",
		"sigmoid": "o = 1.0 / (1.0 + exp(0.0 - n)); \n",
		"tanh": "o = (2.0 / (1.0 + exp(-2.0 * n))) - 1.0; \n",
		"softplus": "o = log(1.0 + exp(n)); \n",
		"softmax": "float k = 0.0; \nfor(int i = 0; i < #(O.shape).x; i++){\nk += exp(O.read(i, pos.y));\n}\no = exp(n) / k; \n"
		//"softsign": "o = n / (1.0 + abs(n)); \n"
	},
	Derivative: {
		"linear": "d = 1.0; \n",
		//"binary": "if (o == 0.0) { d = 0.0; } else { d = 0.0; } \n",
		"relu": "if (o >= 0.0) { d = 1.0; } else { d = 0.0; } \n",
		"lrelu": "if (o >= 0.0) { d = 1.0; } else { d = 0.01; } \n",
		"sigmoid": "d = o * ( 1.0 - o ); \n",
		"tanh": "d = ( 1.0 - pow(o, 2.0) ); \n",
		"softplus": "d = 1.0 - ( 1.0 / exp(o) ); \n",
		"softmax": "d = o * ( 1.0 - o ); \n" // same as sigmoid?
		//"softsign": "var = "
	}
};

},{}],5:[function(require,module,exports){
"use strict";

var ndarray = require("ndarray"),
    TF,
    GL,
    Grad = "uniform Tensor O; \n" + "uniform Tensor E; \n" + "float process(ivec4 pos) { \n" + "return 0.0 - E.read(pos) / O.read(pos); \n" + "} \n",
    Loss = "uniform Tensor O; \n" + "uniform Tensor E; \n" + "float process(ivec4 pos) { \n" + "float loss = 0.0; \n" + "for(int i = 0; i < #(O.shape).y; i++){ \n" /* iterate over each sample */
+ "float l = 0.0; \n" + "for(int j = 0; j < #(O.shape).x; j++){ \n" /* iterate over every output and calculate average */
+ "l -= E.read(j, i) * log(O.read(j, i)); \n" + "} \n" + "loss = l / float(#(O.shape).y); \n" + "} \n" + "return loss; \n" + "} \n";

function CrossEntropy() {
	// calculate loss gradients
	this.grad = Grad;

	// calculate batch average loss
	this.lossF = Loss;

	this.loss = new TF.OutputTensor(GL, [1]);
	this.output = null;
}
CrossEntropy.prototype.deltas = function (output, expect) {
	if (expect instanceof Float32Array) expect = new TF.Tensor(GL, ndarray(expect, output.shape));

	//console.log("Calculon- expected: " + expect.read().data);

	this.output = new TF.OutputTensor(GL, output.shape);
	this.output.run(this.grad, { O: output, E: expect });
	//console.log("Calculon- gradient: " + this.output.read().data);

	this.loss.run(this.lossF, { O: output, E: expect });
	//console.log(this.loss.read());

	return this.output;
};

module.exports = function (tensorfire, glContext) {
	TF = tensorfire;
	GL = glContext;
	return CrossEntropy;
};

},{"ndarray":15}],6:[function(require,module,exports){
'use strict';

var ndarray = require("ndarray"),
    TF,
    GL,
    Funcs = require('./Activations'),
    generateWeights = require('../util/generateWeights'),
    Forward = "uniform Tensor W; \n" /* weights */
+ "uniform Tensor I; \n" /* input */
+ "float process(ivec4 pos) { \n" + "float n = 0.0; \n" + "for(int i = 0; i < #(W.shape).y; i++){ \n" + "if (i == #(W.shape).y - 1) { n += W.read(pos.x, i); } \n" + "else { n += I.read(i, pos.y) * W.read(pos.x, i); } \n" + "} \n" + "return n;\n" + "} \n",
    Backward = "uniform Tensor E; \n" /* local error (from activation) */
+ "uniform Tensor W; \n" /* weights */
+ "float process(ivec4 pos) { \n" // position in partial Tensor
+ "float e = 0.0; \n" /* sum output error */
+ "for(int i = 0; i < #(E.shape).x ; i++){ \n" + "e += W.read(pos.x, i) * E.read(i, pos.y); \n" + "} \n" + "return e; \n" + "} \n",
    Weights = "uniform Tensor E; \n" /* local error (from activation) */
+ "uniform Tensor W; \n" /* weights */
+ "uniform Tensor I; \n" /* input */
+ "uniform float l; \n" /* learning rate */
+ "float process(ivec4 pos) { \n" // pos in weights Tensor
+ "float e = 0.0; \n" /* avg node batch error */
+ "for(int i = 0; i < #(E.shape).y; i++){ \n" + "if (pos.y == #(I.shape).x) { \n" /* handle bias layer ? */
+ "e += E.read(pos.x, i) / float(#(E.shape).y); \n" + "} else { \n" + "e += E.read(pos.x, i) * I.read(pos.y, i) / float(#(E.shape).y); \n" + "} \n" + "} \n" + "return W.read(pos) - (l * e); \n" + "} \n",
    ActA = "uniform Tensor O; \n" /* weighted output */
+ "float process(ivec4 pos) { \n" + "float n = O.read(pos); \n" + "float o; \n",
    ActB = "return o; \n" + "} \n",
    GradA = "uniform Tensor E; \n" + "uniform Tensor O; \n" + "uniform Tensor H; \n" + "float process(ivec4 pos) { \n" + "float d; \n" + "float o = O.read(pos); \n",
    GradB = "d *= E.read(pos); \n" + "return d; \n" + "} \n";

function Dense(layer, index) {
	this.l = index;
	// produce Output Tensor given input, weights, and bias Tensors
	this.forward = Forward;

	this.activation = ActA + Funcs.Activation[layer.activation] + ActB;

	// produce upstream error Tensor given downstream error, input, weights, bias
	this.backward = Backward;
	this.gradient = GradA + Funcs.Derivative[layer.activation] + GradB;
	// adjust weights Tensor given error and input Tensor
	this.update = Weights;

	this.shape = layer.shape;
	this.input = null;
	this.output = null;
	this.weightedOutput = null;
	this.weights = null;
	this.bias = layer.bias;
	this.size = this.shape[0] * this.shape[1] + (this.bias ? this.shape[0] : 0);
}
Dense.prototype.load = function (array, offset) {
	var length = this.size;
	// read in weights (and bias)
	this.weights = new TF.InPlaceTensor(GL, ndarray(array.subarray(offset, offset + length), [this.shape[0], this.shape[1] + (this.bias ? 1 : 0)]));
	offset += length;
	return offset;
};
Dense.prototype.randomWeights = function () {
	this.weights = new TF.InPlaceTensor(GL, ndarray(generateWeights(this.shape, this.bias ? this.shape[0] : 0), [this.shape[0], this.shape[1] + (this.bias ? 1 : 0)]));
};
Dense.prototype.save = function () {
	return this.weights.read().data;
};
Dense.prototype.run = function (input) {
	if (input instanceof Float32Array) {
		this.input = new TF.Tensor(GL, ndarray(input, [this.shape[1], input.length / this.shape[1] >> 0]));
	} else this.input = input;
	//console.log(this.input.shape);
	//console.log("Calculon- input " + this.l + ": " + this.input.read().data);
	//console.log("Calculon- weights " + this.l + ": " + this.weights.read().data);

	this.weightedOutput = new TF.OutputTensor(GL, [this.shape[0], this.input.shape[1]]);
	this.weightedOutput.run(this.forward, { W: this.weights, I: this.input });

	//console.log("Calculon- weightedOutput " + this.l + ": " + this.weightedOutput.read().data);

	this.output = new TF.OutputTensor(GL, [this.shape[0], this.input.shape[1]]);
	this.output.run(this.activation, { O: this.weightedOutput });

	//console.log("output " + this.l + ": " + this.output.read().data);
	return this.output;
};
Dense.prototype.train = function (error, learning_rate) {
	var partial = new TF.OutputTensor(GL, this.input.shape);
	var local = new TF.OutputTensor(GL, this.output.shape);

	//console.log("Calculon- error: " + error.read().data);
	//console.log("Calculon- weights " + this.l + ": " + this.weights.read().data);

	local.run(this.gradient, { E: error, O: this.output, H: this.weightedOutput });
	//console.log("Calculon- localE: " + local.read().data);

	// train weights
	this.weights.run(this.update, { W: this.weights, E: local, I: this.input, l: learning_rate });

	//console.log("Calculon- updated " + this.l + ": " + this.weights.read().data);

	// calculate upstream errors
	partial.run(this.backward, { E: error, I: this.input, W: this.weights, O: this.output });

	return partial;
};

module.exports = function (tensorfire, glContext) {
	TF = tensorfire;
	GL = glContext;
	return Dense;
};

},{"../util/generateWeights":10,"./Activations":4,"ndarray":15}],7:[function(require,module,exports){
"use strict";

var ndarray = require("ndarray"),
    TF,
    GL,
    Grad = "uniform Tensor O; \n" + "uniform Tensor E; \n" + "float process(ivec4 pos) { \n" + "return O.read(pos) - E.read(pos); \n" + "} \n",
    Loss = "uniform Tensor G; \n" + "float process(ivec4 pos) { \n" + "float loss = 0.0; \n" + "for(int i = 0; i < #(G.shape).y; i++){ \n" /* iterate over each sample */
+ "float l = 0.0; \n" + "for(int j = 0; j < #(G.shape).x; j++){ \n" /* iterate over every output and calculate average */
+ "l += pow(float(G.read(j, i)), 2.0) / float(#(G.shape).x); \n" + "} \n" + "loss += l / float(#(G.shape).y); \n" + "} \n" + "return loss; \n" + "} \n";

function MSE() {
	// calculate loss gradients
	this.grad = Grad;

	// calculate batch average loss
	this.lossF = Loss;

	this.loss = new TF.OutputTensor(GL, [1]);
	this.output = null;
}
MSE.prototype.deltas = function (output, expect) {
	if (expect instanceof Float32Array) expect = new TF.Tensor(GL, ndarray(expect, output.shape));

	//console.log("Calculon- expected: " + expect.read().data);

	this.output = new TF.OutputTensor(GL, output.shape);
	this.output.run(this.grad, { O: output, E: expect });
	//console.log("Calculon- gradient: " + this.output.read().data);

	this.loss.run(this.lossF, { G: this.output });

	return this.output;
};

module.exports = function (tensorfire, glContext) {
	TF = tensorfire;
	GL = glContext;
	return MSE;
};

},{"ndarray":15}],8:[function(require,module,exports){
'use strict';

var ndarray = require("ndarray"),
    TF,
    GL,
    Softmax = require('./Softmax'),
    MSE = require('./MSE'),
    CrossEntropy = require('./CrossEntropy'),
    Accuracy = "uniform Tensor O; \n" + "uniform Tensor E; \n" + "float process(ivec4 pos) { \n" + "int l = #(O.shape).x; \n" + "float v = 0.0; \n" + "for (int i = 0; i < #(O.shape).x; i++) { \n" /* iterate over every output and calculate average */
+ "if (O.read(i, pos.x) >= v) { \n" + "v = O.read(i, pos.y); \n" /* get largest category in output */
+ "l = i; \n" /* save index of largest value */
+ "} \n" + "} \n" + "if (E.read(l, pos.y) > 0.9) { \n" + "return 1.0; \n" + "} else { \n" + "return 0.0; \n" + "} \n" + "} \n",
    AccSum = "uniform Tensor A; \n" + "float process(ivec4 pos) { \n" + "float acc = 0.0; \n" + "for(int i = 0; i < #(A.shape).y; i++){ \n" /* iterate over each sample */
+ "acc += A.read(pos.x, i); \n" + "} \n" + "acc = acc / float(#(A.shape).y); \n" + "return acc; \n" + "} \n";

function Output(layer, index) {
	var _this = this;

	this.output = null;
	if (layer.activation === "softmax" && layer.loss === "xentropy") {
		this.output = new Softmax(layer, index);
		this.output.deltas = this.output.train;
		this.run = function (input) {
			_this._output = _this.output.run(input);
			return _this._output;
		};
	} else {
		switch (layer.loss) {
			case "xentropy":
				this.output = new CrossEntropy();
				break;
			case "mse":
				this.output = new MSE();
				break;
		}
		this.run = this.run.bind(this);
		this.train = this.train.bind(this);
	}

	this._output = null;
	this.accuracy = 0;
}
Output.prototype.run = function (input) {
	this._output = input;
	return input;
};
Output.prototype.train = function (expected) {
	if (expected instanceof Float32Array) expected = new TF.Tensor(GL, ndarray(expected, this._output.shape));

	this.batchAccuracy = new TF.OutputTensor(GL, this._output.shape);
	this._accuracy = new TF.OutputTensor(GL, [1]);
	this.batchAccuracy.run(Accuracy, { O: this._output, E: expected });
	this._accuracy.run(AccSum, { A: this.batchAccuracy });
	this.accuracy = this._accuracy.read().data[0];

	return this.output.deltas(this._output, expected);
};

module.exports = function (tensorfire, glContext) {

	TF = tensorfire;
	GL = glContext;

	Softmax = Softmax(tensorfire, glContext);
	MSE = MSE(tensorfire, glContext);
	CrossEntropy = CrossEntropy(tensorfire, glContext);

	return Output;
};

},{"./CrossEntropy":5,"./MSE":7,"./Softmax":9,"ndarray":15}],9:[function(require,module,exports){
"use strict";

var ndarray = require("ndarray"),
    TF,
    GL,
    Act = "uniform Tensor O; \n" /* input */
+ "float process(ivec4 pos) { \n" + "float n = O.read(pos); \n" + "float o; \n" + "float k = 0.0; \n" + "for(int i = 0; i < #(O.shape).x; i++){ \n" + "k += exp(O.read(i, pos.y)); \n" + "} \n" + "return exp(n) / k; \n" + "} \n",
    Grad = "uniform Tensor O; \n" + "uniform Tensor E; \n" + "float process(ivec4 pos) { \n" + "return O.read(pos) - E.read(pos); \n" + "} \n",
    Loss = "uniform Tensor G; \n" + "float process(ivec4 pos) { \n" + "float loss = 0.0; \n" + "for(int i = 0; i < #(G.shape).y; i++){ \n" /* iterate over each sample */
+ "float l = 0.0; \n" + "for(int j = 0; j < #(G.shape).x; j++){ \n" /* iterate over every output and calculate average */
+ "l += pow(float(G.read(j, i)), 2.0) / float(#(G.shape).x); \n" + "} \n" + "loss += l / float(#(G.shape).y); \n" + "} \n" + "return loss; \n" + "} \n";

function Softmax(layer, index) {
	this.l = index;

	this.activation = Act;

	this.gradient = Grad;
	this.lossF = Loss;

	this.input = null;
	this.output = null;
	this.loss = new TF.OutputTensor(GL, [1]);
}
Softmax.prototype.run = function (input) {

	this.input = input;
	this.output = new TF.OutputTensor(GL, this.input.shape);

	this.output.run(this.activation, { O: this.input });

	//console.log("output " + this.l + ": " + this.output.read().data);
	return this.output;
};
Softmax.prototype.train = function (expected) {
	var partial = new TF.OutputTensor(GL, this.input.shape);

	if (expected instanceof Float32Array) expected = new TF.Tensor(GL, ndarray(expected, this.input.shape));

	// calculate upstream errors
	partial.run(this.gradient, { O: this.output, E: expected });

	// calculate batch training loss
	this.loss.run(this.lossF, { G: partial });
	//console.log(this.loss.read());

	return partial;
};

module.exports = function (tensorfire, glContext) {
	TF = tensorfire;
	GL = glContext;
	return Softmax;
};

},{"ndarray":15}],10:[function(require,module,exports){
"use strict";

// Standard Normal variate using Box-Muller transform.

function random(mean, stdDev) {
	mean = mean || 0;
	stdDev = stdDev || 1;
	var u = 0,
	    v = 0;
	while (u === 0) {
		u = Math.random();
	} //Converting [0,1) to (0,1)
	while (v === 0) {
		v = Math.random();
	} //return 0.4;
	return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * stdDev + mean;
}

module.exports = function generateWeights(shape, bias) {
	var result = new Float32Array(shape[0] * shape[1] + bias);
	var l = -1;
	while (++l < result.length) {
		result[l] = random(0, Math.sqrt(2 / shape[1]));
	}
	//console.log(result[0]);
	return result;
};

},{}],11:[function(require,module,exports){
var sprintf = require('sprintf');
module.exports = format;

function format (x, bytes) {
    if (bytes === undefined) bytes = 8;
    var rfmt = '%' + bytes + '.' + bytes + 's';
    
    if (bytes <= 0) return undefined;
    if (isNaN(x)) return sprintf(rfmt, 'NaN');
    if (x === Infinity) {
        if (bytes === 1) return undefined;
        return sprintf(rfmt, bytes >= 9 ? 'Infinity' : ' Inf').slice(0, bytes);
    }
    if (x === -Infinity) {
        if (bytes === 1) return undefined;
        return sprintf(rfmt, bytes >= 9 ? '-Infinity' : '-Inf').slice(0, bytes);
    }
    return packf(x, bytes);
};

function sci (x, bytes) {
    var n = Math.max(1, log10f(Math.abs(x)));
    var sz = log10f(Math.abs(n));
    
    var b = Math.pow(10,bytes+1);
    if (Math.abs(x) < 1) {
        x = Math.round(x * b) / b;
    }
    else {
        var tn = Math.pow(10, n + 1);
        x = Math.round(x / tn * b) / b * tn;
    }
    
    var s;
    if (bytes - sz - 6 === -1) {
        x = Math.round(x / Math.pow(10, n));
        x = x * Math.pow(10, n);
        s = sprintf('%1e', x).replace(/\.[^e]+/, '');
    }
    else if (bytes - sz - 6 < 0) return undefined;
    else {
        s = sprintf('%.' + (bytes - sz - 6) + 'e', x);
    }
    if (x > 0) s = ' ' + s;
    return pad(s, bytes);
}

function pad (s, bytes) {
    return Array(Math.max(0, bytes - s.length + 1)).join(' ') + s;
}

function log10f (n) {
    return Math.floor(Math.log(n) / Math.log(10));
}

function packf (x, bytes) {
    var lbytes = Math.max(1, Math.floor((bytes - 2) / 2));
    var rbytes = bytes - lbytes - 2;
    
    if (x === 0 && bytes < 4) {
        return pad('0', bytes);
    }
    else if (x === 0) {
        return pad('0.' + Array(rbytes+1).join('0'), bytes);
    }
    
    if (rbytes <= 0) {
        var s = sprintf('%' + lbytes + 'f', x);
        if (x >= 0) s = ' ' + s;
        if (s.length > bytes) return undefined;
        return pad(s, bytes);
    }
    if (Math.abs(x) < Math.pow(10,1-rbytes)) return sci(x, bytes);
    
    var b = Math.pow(10,bytes-3);
    var tn = Math.pow(10, log10f(Math.abs(x)));
    var xr = Math.round(x / tn * b) / b * tn;
    
    var s = sprintf('%' + lbytes + '.' + rbytes + 'f', xr);
    if (xr > 0) s = ' ' + s;
    s = s.slice(0, bytes);
    var r = s.split('.')[1];
    if (!r || r.length < 1) return sci(xr, bytes);
    return pad(s, bytes).slice(0, bytes);
}

},{"sprintf":16}],12:[function(require,module,exports){
"use strict"

function iota(n) {
  var result = new Array(n)
  for(var i=0; i<n; ++i) {
    result[i] = i
  }
  return result
}

module.exports = iota
},{}],13:[function(require,module,exports){
/*!
 * Determine if an object is a Buffer
 *
 * @author   Feross Aboukhadijeh <feross@feross.org> <http://feross.org>
 * @license  MIT
 */

// The _isBuffer check is for Safari 5-7 support, because it's missing
// Object.prototype.constructor. Remove this eventually
module.exports = function (obj) {
  return obj != null && (isBuffer(obj) || isSlowBuffer(obj) || !!obj._isBuffer)
}

function isBuffer (obj) {
  return !!obj.constructor && typeof obj.constructor.isBuffer === 'function' && obj.constructor.isBuffer(obj)
}

// For Node v0.10 support. Remove this eventually.
function isSlowBuffer (obj) {
  return typeof obj.readFloatLE === 'function' && typeof obj.slice === 'function' && isBuffer(obj.slice(0, 0))
}

},{}],14:[function(require,module,exports){
var showf = require('fixed-width-float');
var ndarray = require('ndarray');

module.exports = function (m, opts) {
    if (!opts) opts = {};
    if (typeof opts === 'number') opts = { width: opts };
    if (!opts.width) opts.width = 8;

    if (m.dimension === undefined) {
        m = ndarray(m);
    }

    if (m.dimension === 1) return d1(m, opts);
    if (m.dimension === 2) return d2(m, opts);
    if (m.dimension === 3) return d3(m, opts);
    if (m.dimension === 4) return d4(m, opts);
};

function d1 (m, opts) {
    var terms = [];
    for (var i = 0; i < m.shape[0]; i++) {
        terms.push(showf(m.get(i), opts.width));
    }
    return terms.join(' ');
}

function d2 (m, opts) {
    var rows = [];
    for (var y = 0; y < m.shape[0]; y++) {
        rows.push(d1(m.pick(y, null), opts));
    }
    return rows.join('\n');
}

function d3 (m, opts) {
    var rows = [];
    for (var z = 0; z < m.shape[0]; z++) {
        rows.push(d2(m.pick(z, null, null), opts), '');
    }
    return rows.join('\n');
}

function d4 (m, opts) {
    var rows = [], len = 3
    for (var w = 0; w < m.shape[0]; w++) {
        var r = d3(m.pick(w, null, null, null), opts)
        rows.push(r);
        var lines = r.split('\n');
        for (var i = 0; i < lines.length; i++) {
            len = Math.max(len, lines[i].length);
        }
    }
    return rows.join('\n' + Array(len+1).join('-') + '\n\n');
}

},{"fixed-width-float":11,"ndarray":15}],15:[function(require,module,exports){
var iota = require("iota-array")
var isBuffer = require("is-buffer")

var hasTypedArrays  = ((typeof Float64Array) !== "undefined")

function compare1st(a, b) {
  return a[0] - b[0]
}

function order() {
  var stride = this.stride
  var terms = new Array(stride.length)
  var i
  for(i=0; i<terms.length; ++i) {
    terms[i] = [Math.abs(stride[i]), i]
  }
  terms.sort(compare1st)
  var result = new Array(terms.length)
  for(i=0; i<result.length; ++i) {
    result[i] = terms[i][1]
  }
  return result
}

function compileConstructor(dtype, dimension) {
  var className = ["View", dimension, "d", dtype].join("")
  if(dimension < 0) {
    className = "View_Nil" + dtype
  }
  var useGetters = (dtype === "generic")

  if(dimension === -1) {
    //Special case for trivial arrays
    var code =
      "function "+className+"(a){this.data=a;};\
var proto="+className+".prototype;\
proto.dtype='"+dtype+"';\
proto.index=function(){return -1};\
proto.size=0;\
proto.dimension=-1;\
proto.shape=proto.stride=proto.order=[];\
proto.lo=proto.hi=proto.transpose=proto.step=\
function(){return new "+className+"(this.data);};\
proto.get=proto.set=function(){};\
proto.pick=function(){return null};\
return function construct_"+className+"(a){return new "+className+"(a);}"
    var procedure = new Function(code)
    return procedure()
  } else if(dimension === 0) {
    //Special case for 0d arrays
    var code =
      "function "+className+"(a,d) {\
this.data = a;\
this.offset = d\
};\
var proto="+className+".prototype;\
proto.dtype='"+dtype+"';\
proto.index=function(){return this.offset};\
proto.dimension=0;\
proto.size=1;\
proto.shape=\
proto.stride=\
proto.order=[];\
proto.lo=\
proto.hi=\
proto.transpose=\
proto.step=function "+className+"_copy() {\
return new "+className+"(this.data,this.offset)\
};\
proto.pick=function "+className+"_pick(){\
return TrivialArray(this.data);\
};\
proto.valueOf=proto.get=function "+className+"_get(){\
return "+(useGetters ? "this.data.get(this.offset)" : "this.data[this.offset]")+
"};\
proto.set=function "+className+"_set(v){\
return "+(useGetters ? "this.data.set(this.offset,v)" : "this.data[this.offset]=v")+"\
};\
return function construct_"+className+"(a,b,c,d){return new "+className+"(a,d)}"
    var procedure = new Function("TrivialArray", code)
    return procedure(CACHED_CONSTRUCTORS[dtype][0])
  }

  var code = ["'use strict'"]

  //Create constructor for view
  var indices = iota(dimension)
  var args = indices.map(function(i) { return "i"+i })
  var index_str = "this.offset+" + indices.map(function(i) {
        return "this.stride[" + i + "]*i" + i
      }).join("+")
  var shapeArg = indices.map(function(i) {
      return "b"+i
    }).join(",")
  var strideArg = indices.map(function(i) {
      return "c"+i
    }).join(",")
  code.push(
    "function "+className+"(a," + shapeArg + "," + strideArg + ",d){this.data=a",
      "this.shape=[" + shapeArg + "]",
      "this.stride=[" + strideArg + "]",
      "this.offset=d|0}",
    "var proto="+className+".prototype",
    "proto.dtype='"+dtype+"'",
    "proto.dimension="+dimension)

  //view.size:
  code.push("Object.defineProperty(proto,'size',{get:function "+className+"_size(){\
return "+indices.map(function(i) { return "this.shape["+i+"]" }).join("*"),
"}})")

  //view.order:
  if(dimension === 1) {
    code.push("proto.order=[0]")
  } else {
    code.push("Object.defineProperty(proto,'order',{get:")
    if(dimension < 4) {
      code.push("function "+className+"_order(){")
      if(dimension === 2) {
        code.push("return (Math.abs(this.stride[0])>Math.abs(this.stride[1]))?[1,0]:[0,1]}})")
      } else if(dimension === 3) {
        code.push(
"var s0=Math.abs(this.stride[0]),s1=Math.abs(this.stride[1]),s2=Math.abs(this.stride[2]);\
if(s0>s1){\
if(s1>s2){\
return [2,1,0];\
}else if(s0>s2){\
return [1,2,0];\
}else{\
return [1,0,2];\
}\
}else if(s0>s2){\
return [2,0,1];\
}else if(s2>s1){\
return [0,1,2];\
}else{\
return [0,2,1];\
}}})")
      }
    } else {
      code.push("ORDER})")
    }
  }

  //view.set(i0, ..., v):
  code.push(
"proto.set=function "+className+"_set("+args.join(",")+",v){")
  if(useGetters) {
    code.push("return this.data.set("+index_str+",v)}")
  } else {
    code.push("return this.data["+index_str+"]=v}")
  }

  //view.get(i0, ...):
  code.push("proto.get=function "+className+"_get("+args.join(",")+"){")
  if(useGetters) {
    code.push("return this.data.get("+index_str+")}")
  } else {
    code.push("return this.data["+index_str+"]}")
  }

  //view.index:
  code.push(
    "proto.index=function "+className+"_index(", args.join(), "){return "+index_str+"}")

  //view.hi():
  code.push("proto.hi=function "+className+"_hi("+args.join(",")+"){return new "+className+"(this.data,"+
    indices.map(function(i) {
      return ["(typeof i",i,"!=='number'||i",i,"<0)?this.shape[", i, "]:i", i,"|0"].join("")
    }).join(",")+","+
    indices.map(function(i) {
      return "this.stride["+i + "]"
    }).join(",")+",this.offset)}")

  //view.lo():
  var a_vars = indices.map(function(i) { return "a"+i+"=this.shape["+i+"]" })
  var c_vars = indices.map(function(i) { return "c"+i+"=this.stride["+i+"]" })
  code.push("proto.lo=function "+className+"_lo("+args.join(",")+"){var b=this.offset,d=0,"+a_vars.join(",")+","+c_vars.join(","))
  for(var i=0; i<dimension; ++i) {
    code.push(
"if(typeof i"+i+"==='number'&&i"+i+">=0){\
d=i"+i+"|0;\
b+=c"+i+"*d;\
a"+i+"-=d}")
  }
  code.push("return new "+className+"(this.data,"+
    indices.map(function(i) {
      return "a"+i
    }).join(",")+","+
    indices.map(function(i) {
      return "c"+i
    }).join(",")+",b)}")

  //view.step():
  code.push("proto.step=function "+className+"_step("+args.join(",")+"){var "+
    indices.map(function(i) {
      return "a"+i+"=this.shape["+i+"]"
    }).join(",")+","+
    indices.map(function(i) {
      return "b"+i+"=this.stride["+i+"]"
    }).join(",")+",c=this.offset,d=0,ceil=Math.ceil")
  for(var i=0; i<dimension; ++i) {
    code.push(
"if(typeof i"+i+"==='number'){\
d=i"+i+"|0;\
if(d<0){\
c+=b"+i+"*(a"+i+"-1);\
a"+i+"=ceil(-a"+i+"/d)\
}else{\
a"+i+"=ceil(a"+i+"/d)\
}\
b"+i+"*=d\
}")
  }
  code.push("return new "+className+"(this.data,"+
    indices.map(function(i) {
      return "a" + i
    }).join(",")+","+
    indices.map(function(i) {
      return "b" + i
    }).join(",")+",c)}")

  //view.transpose():
  var tShape = new Array(dimension)
  var tStride = new Array(dimension)
  for(var i=0; i<dimension; ++i) {
    tShape[i] = "a[i"+i+"]"
    tStride[i] = "b[i"+i+"]"
  }
  code.push("proto.transpose=function "+className+"_transpose("+args+"){"+
    args.map(function(n,idx) { return n + "=(" + n + "===undefined?" + idx + ":" + n + "|0)"}).join(";"),
    "var a=this.shape,b=this.stride;return new "+className+"(this.data,"+tShape.join(",")+","+tStride.join(",")+",this.offset)}")

  //view.pick():
  code.push("proto.pick=function "+className+"_pick("+args+"){var a=[],b=[],c=this.offset")
  for(var i=0; i<dimension; ++i) {
    code.push("if(typeof i"+i+"==='number'&&i"+i+">=0){c=(c+this.stride["+i+"]*i"+i+")|0}else{a.push(this.shape["+i+"]);b.push(this.stride["+i+"])}")
  }
  code.push("var ctor=CTOR_LIST[a.length+1];return ctor(this.data,a,b,c)}")

  //Add return statement
  code.push("return function construct_"+className+"(data,shape,stride,offset){return new "+className+"(data,"+
    indices.map(function(i) {
      return "shape["+i+"]"
    }).join(",")+","+
    indices.map(function(i) {
      return "stride["+i+"]"
    }).join(",")+",offset)}")

  //Compile procedure
  var procedure = new Function("CTOR_LIST", "ORDER", code.join("\n"))
  return procedure(CACHED_CONSTRUCTORS[dtype], order)
}

function arrayDType(data) {
  if(isBuffer(data)) {
    return "buffer"
  }
  if(hasTypedArrays) {
    switch(Object.prototype.toString.call(data)) {
      case "[object Float64Array]":
        return "float64"
      case "[object Float32Array]":
        return "float32"
      case "[object Int8Array]":
        return "int8"
      case "[object Int16Array]":
        return "int16"
      case "[object Int32Array]":
        return "int32"
      case "[object Uint8Array]":
        return "uint8"
      case "[object Uint16Array]":
        return "uint16"
      case "[object Uint32Array]":
        return "uint32"
      case "[object Uint8ClampedArray]":
        return "uint8_clamped"
    }
  }
  if(Array.isArray(data)) {
    return "array"
  }
  return "generic"
}

var CACHED_CONSTRUCTORS = {
  "float32":[],
  "float64":[],
  "int8":[],
  "int16":[],
  "int32":[],
  "uint8":[],
  "uint16":[],
  "uint32":[],
  "array":[],
  "uint8_clamped":[],
  "buffer":[],
  "generic":[]
}

;(function() {
  for(var id in CACHED_CONSTRUCTORS) {
    CACHED_CONSTRUCTORS[id].push(compileConstructor(id, -1))
  }
});

function wrappedNDArrayCtor(data, shape, stride, offset) {
  if(data === undefined) {
    var ctor = CACHED_CONSTRUCTORS.array[0]
    return ctor([])
  } else if(typeof data === "number") {
    data = [data]
  }
  if(shape === undefined) {
    shape = [ data.length ]
  }
  var d = shape.length
  if(stride === undefined) {
    stride = new Array(d)
    for(var i=d-1, sz=1; i>=0; --i) {
      stride[i] = sz
      sz *= shape[i]
    }
  }
  if(offset === undefined) {
    offset = 0
    for(var i=0; i<d; ++i) {
      if(stride[i] < 0) {
        offset -= (shape[i]-1)*stride[i]
      }
    }
  }
  var dtype = arrayDType(data)
  var ctor_list = CACHED_CONSTRUCTORS[dtype]
  while(ctor_list.length <= d+1) {
    ctor_list.push(compileConstructor(dtype, ctor_list.length-1))
  }
  var ctor = ctor_list[d+1]
  return ctor(data, shape, stride, offset)
}

module.exports = wrappedNDArrayCtor

},{"iota-array":12,"is-buffer":13}],16:[function(require,module,exports){
/**
sprintf() for JavaScript 0.7-beta1
http://www.diveintojavascript.com/projects/javascript-sprintf

Copyright (c) Alexandru Marasteanu <alexaholic [at) gmail (dot] com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of sprintf() for JavaScript nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Alexandru Marasteanu BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Changelog:
2010.11.07 - 0.7-beta1-node
  - converted it to a node.js compatible module

2010.09.06 - 0.7-beta1
  - features: vsprintf, support for named placeholders
  - enhancements: format cache, reduced global namespace pollution

2010.05.22 - 0.6:
 - reverted to 0.4 and fixed the bug regarding the sign of the number 0
 Note:
 Thanks to Raphael Pigulla <raph (at] n3rd [dot) org> (http://www.n3rd.org/)
 who warned me about a bug in 0.5, I discovered that the last update was
 a regress. I appologize for that.

2010.05.09 - 0.5:
 - bug fix: 0 is now preceeded with a + sign
 - bug fix: the sign was not at the right position on padded results (Kamal Abdali)
 - switched from GPL to BSD license

2007.10.21 - 0.4:
 - unit test and patch (David Baird)

2007.09.17 - 0.3:
 - bug fix: no longer throws exception on empty paramenters (Hans Pufal)

2007.09.11 - 0.2:
 - feature: added argument swapping

2007.04.03 - 0.1:
 - initial release
**/

var sprintf = (function() {
	function get_type(variable) {
		return Object.prototype.toString.call(variable).slice(8, -1).toLowerCase();
	}
	function str_repeat(input, multiplier) {
		for (var output = []; multiplier > 0; output[--multiplier] = input) {/* do nothing */}
		return output.join('');
	}

	var str_format = function() {
		if (!str_format.cache.hasOwnProperty(arguments[0])) {
			str_format.cache[arguments[0]] = str_format.parse(arguments[0]);
		}
		return str_format.format.call(null, str_format.cache[arguments[0]], arguments);
	};

	// convert object to simple one line string without indentation or
	// newlines. Note that this implementation does not print array
	// values to their actual place for sparse arrays. 
	//
	// For example sparse array like this
	//    l = []
	//    l[4] = 1
	// Would be printed as "[1]" instead of "[, , , , 1]"
	// 
	// If argument 'seen' is not null and array the function will check for 
	// circular object references from argument.
	str_format.object_stringify = function(obj, depth, maxdepth, seen) {
		var str = '';
		if (obj != null) {
			switch( typeof(obj) ) {
			case 'function': 
				return '[Function' + (obj.name ? ': '+obj.name : '') + ']';
			    break;
			case 'object':
				if ( obj instanceof Error) { return '[' + obj.toString() + ']' };
				if (depth >= maxdepth) return '[Object]'
				if (seen) {
					// add object to seen list
					seen = seen.slice(0)
					seen.push(obj);
				}
				if (obj.length != null) { //array
					str += '[';
					var arr = []
					for (var i in obj) {
						if (seen && seen.indexOf(obj[i]) >= 0) arr.push('[Circular]');
						else arr.push(str_format.object_stringify(obj[i], depth+1, maxdepth, seen));
					}
					str += arr.join(', ') + ']';
				} else if ('getMonth' in obj) { // date
					return 'Date(' + obj + ')';
				} else { // object
					str += '{';
					var arr = []
					for (var k in obj) { 
						if(obj.hasOwnProperty(k)) {
							if (seen && seen.indexOf(obj[k]) >= 0) arr.push(k + ': [Circular]');
							else arr.push(k +': ' +str_format.object_stringify(obj[k], depth+1, maxdepth, seen)); 
						}
					}
					str += arr.join(', ') + '}';
				}
				return str;
				break;
			case 'string':				
				return '"' + obj + '"';
				break
			}
		}
		return '' + obj;
	}

	str_format.format = function(parse_tree, argv) {
		var cursor = 1, tree_length = parse_tree.length, node_type = '', arg, output = [], i, k, match, pad, pad_character, pad_length;
		for (i = 0; i < tree_length; i++) {
			node_type = get_type(parse_tree[i]);
			if (node_type === 'string') {
				output.push(parse_tree[i]);
			}
			else if (node_type === 'array') {
				match = parse_tree[i]; // convenience purposes only
				if (match[2]) { // keyword argument
					arg = argv[cursor];
					for (k = 0; k < match[2].length; k++) {
						if (!arg.hasOwnProperty(match[2][k])) {
							throw new Error(sprintf('[sprintf] property "%s" does not exist', match[2][k]));
						}
						arg = arg[match[2][k]];
					}
				}
				else if (match[1]) { // positional argument (explicit)
					arg = argv[match[1]];
				}
				else { // positional argument (implicit)
					arg = argv[cursor++];
				}

				if (/[^sO]/.test(match[8]) && (get_type(arg) != 'number')) {
					throw new Error(sprintf('[sprintf] expecting number but found %s "' + arg + '"', get_type(arg)));
				}
				switch (match[8]) {
					case 'b': arg = arg.toString(2); break;
					case 'c': arg = String.fromCharCode(arg); break;
					case 'd': arg = parseInt(arg, 10); break;
					case 'e': arg = match[7] ? arg.toExponential(match[7]) : arg.toExponential(); break;
					case 'f': arg = match[7] ? parseFloat(arg).toFixed(match[7]) : parseFloat(arg); break;
				    case 'O': arg = str_format.object_stringify(arg, 0, parseInt(match[7]) || 5); break;
					case 'o': arg = arg.toString(8); break;
					case 's': arg = ((arg = String(arg)) && match[7] ? arg.substring(0, match[7]) : arg); break;
					case 'u': arg = Math.abs(arg); break;
					case 'x': arg = arg.toString(16); break;
					case 'X': arg = arg.toString(16).toUpperCase(); break;
				}
				arg = (/[def]/.test(match[8]) && match[3] && arg >= 0 ? '+'+ arg : arg);
				pad_character = match[4] ? match[4] == '0' ? '0' : match[4].charAt(1) : ' ';
				pad_length = match[6] - String(arg).length;
				pad = match[6] ? str_repeat(pad_character, pad_length) : '';
				output.push(match[5] ? arg + pad : pad + arg);
			}
		}
		return output.join('');
	};

	str_format.cache = {};

	str_format.parse = function(fmt) {
		var _fmt = fmt, match = [], parse_tree = [], arg_names = 0;
		while (_fmt) {
			if ((match = /^[^\x25]+/.exec(_fmt)) !== null) {
				parse_tree.push(match[0]);
			}
			else if ((match = /^\x25{2}/.exec(_fmt)) !== null) {
				parse_tree.push('%');
			}
			else if ((match = /^\x25(?:([1-9]\d*)\$|\(([^\)]+)\))?(\+)?(0|'[^$])?(-)?(\d+)?(?:\.(\d+))?([b-fosOuxX])/.exec(_fmt)) !== null) {
				if (match[2]) {
					arg_names |= 1;
					var field_list = [], replacement_field = match[2], field_match = [];
					if ((field_match = /^([a-z_][a-z_\d]*)/i.exec(replacement_field)) !== null) {
						field_list.push(field_match[1]);
						while ((replacement_field = replacement_field.substring(field_match[0].length)) !== '') {
							if ((field_match = /^\.([a-z_][a-z_\d]*)/i.exec(replacement_field)) !== null) {
								field_list.push(field_match[1]);
							}
							else if ((field_match = /^\[(\d+)\]/.exec(replacement_field)) !== null) {
								field_list.push(field_match[1]);
							}
							else {
								throw new Error('[sprintf] ' + replacement_field);
							}
						}
					}
					else {
                        throw new Error('[sprintf] ' + replacement_field);
					}
					match[2] = field_list;
				}
				else {
					arg_names |= 2;
				}
				if (arg_names === 3) {
					throw new Error('[sprintf] mixing positional and named placeholders is not (yet) supported');
				}
				parse_tree.push(match);
			}
			else {
				throw new Error('[sprintf] ' + _fmt);
			}
			_fmt = _fmt.substring(match[0].length);
		}
		return parse_tree;
	};

	return str_format;
})();

var vsprintf = function(fmt, argv) {
	var argvClone = argv.slice();
	argvClone.unshift(fmt);
	return sprintf.apply(null, argvClone);
};

module.exports = sprintf;
sprintf.sprintf = sprintf;
sprintf.vsprintf = vsprintf;

},{}],17:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
	value: true
});
exports.default = {
	hard_sigmoid: 'float @activation1(float data){\n    return clamp(data * 0.2 + 0.5, 0.0, 1.0);\n}',
	linear: '#define @activation1 \n',
	relu: 'float @activation1(float data){\n    return max(data, 0.0);\n}\n',
	rgb: 'float @activation1(float data){\n    return data / 255.0; \n}',
	sigmoid: 'float @activation1(float data){\n    return (1.0/(1.0 + exp(-2.0 * \n        clamp(data,-20.0, 20.0) )));\n}\n',
	tanh: 'float @activation1(float data){\n    float e = exp(2.0 * clamp(data, -20.0, 20.0) );\n    return (e-1.0)/(e+1.0);\n}'
};

},{}],18:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
	value: true
});
exports.init = init;
exports.encode = encode;
exports.decode = decode;
var encodeShader = exports.encodeShader = 'uniform float @range;\n\nvec4 @encode1(float v) {\n\tfloat z = clamp(v / @range + 0.5, 0.0, 1.0);\n\n\treturn mod(z * vec4(\n\t\t256.0 * 256.0 * 256.0 * 256.0,\n\t\t256.0 * 256.0 * 256.0,\n\t\t256.0 * 256.0,\n\t\t256.0\n\t), vec4(256.0)) / 255.0;\n}';
var decodeShader = exports.decodeShader = 'uniform float @range;\n\nfloat @decode1(vec4 rgba) {\n\tfloat f = dot(rgba, vec4(\n\t\t255.0 / 256.0 / 256.0 / 256.0 / 256.0,\n\t\t255.0 / 256.0 / 256.0 / 256.0,\n\t\t255.0 / 256.0 / 256.0,\n\t\t255.0 / 256.0\n\t));\n\treturn (f - 0.5) * @range;\n}';

function init(shape, format) {
	return {
		range: format.range || 4096
	};
}

function encode(buf, value, info) {
	var z = Math.min(1, Math.max(0, value / info.range + 0.5));
	buf[0] = z * 256 * 256 * 256 * 256 % 256;
	buf[1] = z * 256 * 256 * 256 % 256;
	buf[2] = z * 256 * 256 % 256;
	buf[3] = z * 256 % 256;
}

function decode(buf) {
	return buf[0] / 256.0 / 256.0 / 256.0 / 256.0 + buf[1] / 256.0 / 256.0 / 256.0 + buf[2] / 256.0 / 256.0 + buf[3] / 256.0;
}

},{}],19:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
	value: true
});
exports.init = init;
exports.encode = encode;
exports.decode = decode;
var encodeShader = exports.encodeShader = '// https://github.com/mikolalysenko/glsl-read-float/blob/master/index.glsl\n\n#define FLOAT_MAX  1.70141184e38\n#define FLOAT_MIN  1.17549435e-38\n\nvec4 @encode1(float v) {\n    highp float av = abs(v);\n\n    //Handle special cases\n    if(av < FLOAT_MIN) {\n        return vec4(0.0, 0.0, 0.0, 0.0);\n    } else if(v > FLOAT_MAX) {\n        return vec4(127.0, 128.0, 0.0, 0.0) / 255.0;\n    } else if(v < -FLOAT_MAX) {\n        return vec4(255.0, 128.0, 0.0, 0.0) / 255.0;\n    }\n\n    highp vec4 c = vec4(0,0,0,0);\n\n    //Compute exponent and mantissa\n    highp float e = floor(log2(av));\n    highp float m = av * pow(2.0, -e) - 1.0;\n    \n    //Unpack mantissa\n    c[1] = floor(128.0 * m);\n    m -= c[1] / 128.0;\n    c[2] = floor(32768.0 * m);\n    m -= c[2] / 32768.0;\n    c[3] = floor(8388608.0 * m);\n    \n    //Unpack exponent\n    highp float ebias = e + 127.0;\n    c[0] = floor(ebias / 2.0);\n    ebias -= c[0] * 2.0;\n    c[1] += floor(ebias) * 128.0; \n\n    //Unpack sign bit\n    c[0] += 128.0 * step(0.0, -v);\n\n    //Scale back to range\n    return c.abgr / 255.0;\n}\n\n// TODO: compare with http://stackoverflow.com/a/7237286';
var decodeShader = exports.decodeShader = '// TODO: compare with http://stackoverflow.com/a/7237286\n\nfloat @decode1(vec4 val){\n    vec4 scl = floor(255.0 * val + 0.5);\n    float sgn = (scl.a < 128.0) ? 1.0 : -1.0;\n    float exn = mod(scl.a * 2.0, 256.0) + floor(scl.b / 128.0) - 127.0;\n    float man = 1.0 +\n        (scl.r / 8388608.0) + \n        (scl.g / 32768.0) +\n        mod(scl.b, 128.0) / 128.0;\n    return sgn * man * pow(2.0, exn);\n}\n';

function init(shape, format) {
	return {};
}

var tmp_float = new Float32Array(1),
    tmp_int = new Uint8Array(tmp_float.buffer);

function encode(buf, value) {
	tmp_float[0] = value;
	buf.set(tmp_int, 0);
}

function decode(buf) {
	tmp_int.set(buf);
	return tmp_float[0];
}

},{}],20:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
	value: true
});

var _index = require('./pack/stride/index.js');

var pack_stride = _interopRequireWildcard(_index);

var _index2 = require('./pack/tile/index.js');

var pack_tile = _interopRequireWildcard(_index2);

var _index3 = require('./codec/fixnum/index.js');

var codec_fixnum = _interopRequireWildcard(_index3);

var _index4 = require('./codec/softfloat/index.js');

var codec_softfloat = _interopRequireWildcard(_index4);

var _index5 = require('./activation/index.js');

var _index6 = _interopRequireDefault(_index5);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

exports.default = {
	pack: {
		stride: pack_stride,
		tile: pack_tile
	},

	read_shim: 'vec4 @read4(ivec4 pos){\n    int z = 4 * (pos.z / 4);\n    \n    if(@shape.z - z == 1){\n        return vec4(\n            @read(ivec4(pos.xy, z    , pos.w)), \n            0,\n            0,\n            0\n        );\n    }else if(@shape.z - z == 2){\n        return vec4(\n            @read(ivec4(pos.xy, z    , pos.w)), \n            @read(ivec4(pos.xy, z + 1, pos.w)),\n            0,\n            0\n        );\n    }else if(@shape.z - z == 3){\n        return vec4(\n            @read(ivec4(pos.xy, z    , pos.w)), \n            @read(ivec4(pos.xy, z + 1, pos.w)),\n            @read(ivec4(pos.xy, z + 2, pos.w)),\n            0\n        );\n    }\n    \n    return vec4(\n        @read(ivec4(pos.xy, z    , pos.w)),\n        @read(ivec4(pos.xy, z + 1, pos.w)),\n        @read(ivec4(pos.xy, z + 2, pos.w)),\n        @read(ivec4(pos.xy, z + 3, pos.w))\n    );\n}',
	write_shim: 'vec4 process4(ivec4 pos);\nfloat process(ivec4 pos){\n    return chsel(process4(ivec4(pos.xy, 4 * (pos.z / 4), pos.w)), imod(pos.z, 4));\n}',

	codec: {
		fixnum: codec_fixnum,
		softfloat: codec_softfloat
	},
	activations: _index6.default
};

},{"./activation/index.js":17,"./codec/fixnum/index.js":18,"./codec/softfloat/index.js":19,"./pack/stride/index.js":21,"./pack/tile/index.js":22}],21:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.writeShader = exports.readShader = undefined;
exports.init = init;
exports.pack = pack;
exports.unpack = unpack;

var _ndarray = require('ndarray');

var _ndarray2 = _interopRequireDefault(_ndarray);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var readShader = exports.readShader = 'uniform sampler2D @tex;\nuniform ivec2 @texSize;\nuniform ivec4 @shape;\nuniform vec4 @stride;\n\n\nfloat @read(ivec4 pos){\n\tint tile = int(dot(vec4(pos), @stride));\n\treturn @decode1(texture2D(@tex,\n\t\t(vec2(0.5, 0.5) + vec2(tile2vec(tile, @texSize.x))) \n\t\t/ vec2(@texSize)));\n}';
var writeShader = exports.writeShader = 'uniform ivec2 @texSize;\nuniform ivec4 @shape;\nuniform vec4 @stride;\n\n// vec4 clampify(vec4 v){\n//     return vec4(ivec4(clamp(v, vec4(0), vec4(1)) * 255.0)) / 255.0;\n// }\n\nfloat process(ivec4 pos);\nvoid main(){\n\tint tile = vec2tile(ivec2(gl_FragCoord.xy), @texSize.x);\n\tint chunks = @shape.x * @shape.y * @shape.z * @shape.w;\n\tif(tile >= chunks){ checkerboard(); return; }\n\n\tgl_FragColor = @encode1(@activation1(process(ivec4(\n\t\timod(tile, @shape.x),\n\t\timod(tile / @shape.x, @shape.y),\n\t\timod(tile / @shape.x / @shape.y, @shape.z ),\n\t\ttile / @shape.x / @shape.y / @shape.z\n\t))));\n}';

function init(shape) {
    // var length = 4 * Math.ceil(shape[2] / 4) * shape[3] * shape[1] * shape[0];
    // var cols = Math.ceil(Math.sqrt(length) / 4) * 4;

    var length = shape[2] * shape[3] * shape[1] * shape[0];
    var cols = Math.ceil(Math.sqrt(length));
    var texSize = [cols, Math.ceil(length / cols)];
    return {
        texSize: texSize,
        shape: shape,
        // vec4(1, @shape.x, @shape.x * @shape.y, @shape.x * @shape.y * @shape.z)
        stride: [1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * shape[2]]
    };
}

function pack(info, array, encode1, format) {
    // return Uint8Array or Float32Array
    array = (0, _ndarray2.default)(array.data, array.shape.concat([1, 1, 1, 1]).slice(0, 4), array.stride.concat([1, 1, 1, 1]).slice(0, 4), array.offset);

    var shape = info.shape;
    var length = info.texSize[0] * info.texSize[1] * 4;

    if (format.type === 'float32') {
        var data = new Float32Array(length);
    } else if (format.type === 'uint8') {
        var data = new Uint8Array(length);
    }

    for (var x = 0; x < shape[0]; x++) {
        for (var y = 0; y < shape[1]; y++) {
            for (var z = 0; z < shape[2]; z++) {
                for (var w = 0; w < shape[3]; w++) {
                    var tile = x + y * shape[0] + z * shape[0] * shape[1] + w * shape[0] * shape[1] * shape[2];

                    encode1(data.subarray(4 * tile, 4 * tile + 4), array.get(x, y, z, w), info);
                }
            }
        }
    }

    return data;
}

function unpack(info, data, decode1, type) {
    if (type != 'float32') throw new Error('not impl');

    var shape = info.shape;
    var length = shape.reduce(function (a, b) {
        return a * b;
    });

    var array = (0, _ndarray2.default)(new Float32Array(length), shape.concat([1, 1, 1, 1]).slice(0, 4));

    for (var x = 0; x < shape[0]; x++) {
        for (var y = 0; y < shape[1]; y++) {
            for (var z = 0; z < shape[2]; z++) {
                for (var w = 0; w < shape[3]; w++) {
                    var tile = x + y * shape[0] + z * shape[0] * shape[1] + w * shape[0] * shape[1] * shape[2];

                    array.set(x, y, z, w, decode1(data.subarray(4 * tile, 4 * tile + 4), info));
                }
            }
        }
    }
    return array;
}

},{"ndarray":15}],22:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.writeShader = exports.readShader = undefined;
exports.init = init;
exports.pack = pack;
exports.unpack = unpack;

var _ndarray = require('ndarray');

var _ndarray2 = _interopRequireDefault(_ndarray);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var readShader = exports.readShader = 'uniform sampler2D @tex;\nuniform ivec2 @texSize;\nuniform ivec4 @shape;\nuniform int @cols;\n\nfloat @read(ivec4 pos){\n    return @decode1(texture2D(@tex, (\n        vec2(tile2vec(\n            vec2tile(pos.zw, @shape.z)\n        , @cols) * ivec2(@shape.xy)) +\n        vec2(pos.xy) + vec2(0.5, 0.5)\n    ) / vec2(@texSize)));\n}\n';
var writeShader = exports.writeShader = 'uniform ivec2 @texSize;\nuniform ivec4 @shape;\nuniform int @cols;\n\n// vec4 clampify(vec4 v){\n//     return vec4(ivec4(clamp(v, vec4(0), vec4(1)) * 255.0)) / 255.0;\n// }\n\nfloat process(ivec4 pos);\nvoid main(){\n    int tile = vec2tile(ivec2(gl_FragCoord.xy) / @shape.xy, @cols);\n    if(tile >= @shape.z * @shape.w){ checkerboard(); return; }\n\n    gl_FragColor = @encode1(@activation1(process(ivec4(\n    \timod(gl_FragCoord.x, @shape.x),\n    \timod(gl_FragCoord.y, @shape.y),\n        // mod(vec2(gl_FragCoord.xy), vec2(@shape.xy)), \n        tile2vec(tile, @shape.z)))));\n}';
function init(shape) {
    var width = shape[0];
    // we pick the number of columns so we can keep
    // the texture as square as possible, with the
    // minimal amount of wasted space.

    var tiles = shape[2] * shape[3],
        cols = Math.max(1, Math.min(tiles, Math.ceil(Math.sqrt(shape[0] * shape[1] * tiles) / width)));

    var texSize = [width * cols, shape[1] * Math.ceil(tiles / cols)];

    return {
        texSize: texSize,
        cols: cols,
        shape: shape
    };
}

function pack(info, ndarray) {
    // return Uint8Array or Float32Array


    // uniform sampler2D @_tex;
    // uniform ivec2 @_texSize;
    // uniform ivec4 @_shape;
    // uniform int @_cols;

    // return {
    //  tex:
    //  texSize:
    //  shape:
    //  cols:
    // }
    throw new Error("not implemented: format/1-4/pack/tile/index.js:pack");
}

function unpack(info, arr) {
    // return ndarray
    throw new Error("not implemented: format/1-4/pack/tile/index.js:unpack");
}

},{"ndarray":15}],23:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
	value: true
});
exports.default = {
	hard_sigmoid: 'vec4 @activation4(vec4 data){\n    return clamp(data * vec4(0.2,0.2,0.2,0.2) + \n        vec4(.5,.5,.5,.5), vec4(0,0,0,0), vec4(1,1,1,1));\n}',
	linear: '#define @activation4 \n',
	relu: 'vec4 @activation4(vec4 data){\n    return max(data, vec4(0, 0, 0, 0));\n}\n',
	rgb: 'vec4 @activation4(vec4 data){\n    return data / 255.0; \n}',
	sigmoid: 'vec4 @activation4(vec4 data){\n    return (vec4(1,1,1,1)/(vec4(1,1,1,1) + exp(-2.0 * \n        clamp(data,vec4(-20,-20,-20,-20), vec4(20,20,20,20)) )));\n}\n',
	tanh: 'vec4 @activation4(vec4 data){\n    vec4 e = exp(2.0 * clamp(data, vec4(-20,-20,-20,-20), vec4(20,20,20,20)) );\n    return (e-vec4(1, 1, 1, 1))/(e+vec4(1, 1, 1, 1));\n}'
};

},{}],24:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
	value: true
});
exports.init = init;
exports.encode = encode;
exports.decode = decode;
var encodeShader = exports.encodeShader = 'uniform vec2 @range;\n\nvec4 @encode4(vec4 v){\n\treturn (v - vec4(@range.x)) / vec4(@range.y - @range.x);\n}';
var decodeShader = exports.decodeShader = 'uniform vec2 @range;\n\nvec4 @decode4(vec4 v){\n\treturn v * vec4(@range.y - @range.x) + vec4(@range.x);\n}';

function init(shape, format) {
	return {
		range: [isFinite(format.min) ? format.min : 0, isFinite(format.max) ? format.max : 1]
		// max: ,
		// min: ,
	};
}

function encode(data, r, g, b, a, info) {

	data[0] = Math.round(255 * Math.min(1, Math.max(0, (r - info.range[0]) / (info.range[1] - info.range[0]))));
	data[1] = Math.round(255 * Math.min(1, Math.max(0, (g - info.range[0]) / (info.range[1] - info.range[0]))));
	data[2] = Math.round(255 * Math.min(1, Math.max(0, (b - info.range[0]) / (info.range[1] - info.range[0]))));
	data[3] = Math.round(255 * Math.min(1, Math.max(0, (a - info.range[0]) / (info.range[1] - info.range[0]))));
	// console.log(data[0], data[1], data[2])
}

function decode(data, r, g, b, a, info) {
	data[0] = r / 255 * (info.range[1] - info.range[0]) + info.range[0];
	data[1] = g / 255 * (info.range[1] - info.range[0]) + info.range[0];
	data[2] = b / 255 * (info.range[1] - info.range[0]) + info.range[0];
	data[3] = a / 255 * (info.range[1] - info.range[0]) + info.range[0];
}

},{}],25:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
	value: true
});
exports.init = init;
exports.encode = encode;
exports.decode = decode;
var encodeShader = exports.encodeShader = '#define @encode4 \n ';
var decodeShader = exports.decodeShader = '#define @decode4 \n';

function init(shape, format) {
	return {};
}

function encode(data, r, g, b, a) {
	data[0] = r;
	data[1] = g;
	data[2] = b;
	data[3] = a;
}

function decode(data, r, g, b, a) {
	data[0] = r;
	data[1] = g;
	data[2] = b;
	data[3] = a;
}

},{}],26:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
	value: true
});

var _index = require('./pack/stride/index.js');

var pack_stride = _interopRequireWildcard(_index);

var _index2 = require('./pack/tile/index.js');

var pack_tile = _interopRequireWildcard(_index2);

var _index3 = require('./codec/raw/index.js');

var codec_raw = _interopRequireWildcard(_index3);

var _index4 = require('./codec/linquant/index.js');

var codec_linquant = _interopRequireWildcard(_index4);

var _index5 = require('./activation/index.js');

var _index6 = _interopRequireDefault(_index5);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _interopRequireWildcard(obj) { if (obj && obj.__esModule) { return obj; } else { var newObj = {}; if (obj != null) { for (var key in obj) { if (Object.prototype.hasOwnProperty.call(obj, key)) newObj[key] = obj[key]; } } newObj.default = obj; return newObj; } }

exports.default = {
	pack: {
		stride: pack_stride,
		tile: pack_tile
	},

	read_shim: 'float @read(ivec4 pos){\n    return chsel(@read4(pos), imod(pos.z, 4));\n}\n',
	write_shim: 'float process(ivec4 pos);\nvec4 process4(ivec4 pos){\n    int z = 4 * (pos.z / 4);\n\n    if(@shape.z - z == 1){\n        return vec4(\n            process(ivec4(pos.xy, z    , pos.w)), \n            0,\n            0,\n            0\n        );\n    }else if(@shape.z - z == 2){\n        return vec4(\n            process(ivec4(pos.xy, z    , pos.w)), \n            process(ivec4(pos.xy, z + 1, pos.w)),\n            0,\n            0\n        );\n    }else if(@shape.z - z == 3){\n        return vec4(\n            process(ivec4(pos.xy, z    , pos.w)), \n            process(ivec4(pos.xy, z + 1, pos.w)),\n            process(ivec4(pos.xy, z + 2, pos.w)),\n            0\n        );\n    }\n    \n    return vec4(\n        process(ivec4(pos.xy, z    , pos.w)),\n        process(ivec4(pos.xy, z + 1, pos.w)),\n        process(ivec4(pos.xy, z + 2, pos.w)),\n        process(ivec4(pos.xy, z + 3, pos.w))\n    );\n}',

	codec: {
		raw: codec_raw,
		linquant: codec_linquant
	},
	activations: _index6.default
};

},{"./activation/index.js":23,"./codec/linquant/index.js":24,"./codec/raw/index.js":25,"./pack/stride/index.js":27,"./pack/tile/index.js":28}],27:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.writeShader = exports.readShader = undefined;

var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

exports.init = init;
exports.pack = pack;
exports.unpack = unpack;

var _ndarray = require('ndarray');

var _ndarray2 = _interopRequireDefault(_ndarray);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var readShader = exports.readShader = 'uniform sampler2D @tex;\nuniform ivec2 @texSize;\nuniform ivec4 @shape;\nuniform vec4 @stride;\n\n\nvec4 @read4(ivec4 pos){\n\tint tile = int(dot(vec4(pos), @stride));\n\treturn @decode4(texture2D(@tex,\n\t\t(vec2(0.5, 0.5) + vec2(tile2vec(tile, @texSize.x))) \n\t\t/ vec2(@texSize)));\n}';
var writeShader = exports.writeShader = 'uniform ivec2 @texSize;\nuniform ivec4 @shape;\nuniform vec4 @stride;\n\nvec4 process4(ivec4 pos);\nvoid main(){\n\tint tile = @texSize.x * int(gl_FragCoord.y) + int(gl_FragCoord.x);\n\tint shapez = ceildiv(@shape.z, 4);\n\tif(tile >= int(@stride.w) * @shape.w){ checkerboard(); return; }\n\n\tgl_FragColor = @encode4(@activation4(process4(ivec4(\n\t\timod(tile, @shape.x),\n\t\timod(tile / @shape.x, @shape.y),\n\t\t4 * imod(tile / @shape.x / @shape.y, shapez),\n\t\ttile / @shape.x / @shape.y / shapez\n\t))));\n}\n';
function init(shape) {
    var length = Math.ceil(shape[2] / 4) * shape[3] * shape[1] * shape[0];
    var cols = Math.ceil(Math.sqrt(length));
    var texSize = [cols, Math.ceil(length / cols)];

    console.assert(texSize[0] * texSize[1] >= length);
    return {
        texSize: texSize,
        shape: shape,

        stride: [1, shape[0], shape[0] * shape[1] / 4, // the /4 is because of the color channel
        shape[0] * shape[1] * Math.ceil(shape[2] / 4)]
        // decvec: [1, shape[0], shape[0] * shape[1], shape[0] * shape[1] * Math.ceil(shape[2] / 4)]
    };
}

function pack(info, array, encode4, format) {
    // return Uint8Array or Float32Array

    array = (0, _ndarray2.default)(array.data, array.shape.concat([1, 1, 1, 1]).slice(0, 4), array.stride.concat([1, 1, 1, 1]).slice(0, 4), array.offset);

    var _info$texSize = _slicedToArray(info.texSize, 2),
        width = _info$texSize[0],
        height = _info$texSize[1],
        length = width * height * 4;

    var shape = info.shape;

    if (format.type === 'float32') {
        var data = new Float32Array(length);
    } else if (format.type === 'uint8') {
        var data = new Uint8Array(length);
    }

    var chans = Math.ceil(info.shape[2] / 4);

    for (var i = 0; i < info.shape[0]; i++) {
        for (var j = 0; j < info.shape[1]; j++) {
            for (var k = 0; k < chans; k++) {
                var b = Math.min(k * 4 + 4, shape[2]) - k * 4;
                for (var w = 0; w < info.shape[3]; w++) {

                    var tile = i + j * shape[0] + k * shape[0] * shape[1] + w * shape[0] * shape[1] * chans;

                    var pos = 4 * tile;
                    encode4(data.subarray(pos, pos + 4), b < 1 ? 0 : array.get(i, j, 4 * k + 0, w), b < 2 ? 0 : array.get(i, j, 4 * k + 1, w), b < 3 ? 0 : array.get(i, j, 4 * k + 2, w), b < 4 ? 0 : array.get(i, j, 4 * k + 3, w), info);
                }
            }
        }
    }

    return data;
}

function unpack(info, data, decode4, type) {

    var shape = info.shape;
    var shapelength = shape.reduce(function (a, b) {
        return a * b;
    });

    var _info$texSize2 = _slicedToArray(info.texSize, 2),
        width = _info$texSize2[0],
        height = _info$texSize2[1],
        length = width * height * 4;

    var chans = Math.ceil(info.shape[2] / 4);

    // if(type === 'float32'){
    var array = (0, _ndarray2.default)(new Float32Array(shapelength), shape);
    var buf = new Float32Array(4);
    // }else if(type == 'uint8'){
    //     var array = ndarray(new Uint8Array(shapelength), shape)
    //     var buf = new Uint8Array(4);
    // }else throw new Error('unimplemented type');


    for (var i = 0; i < info.shape[0]; i++) {
        for (var j = 0; j < info.shape[1]; j++) {
            for (var k = 0; k < chans; k++) {
                var b = Math.min(k * 4 + 4, shape[2]) - k * 4;
                for (var w = 0; w < info.shape[3]; w++) {

                    var tile = i + j * shape[0] + k * shape[0] * shape[1] + w * shape[0] * shape[1] * chans;

                    decode4(buf, data[4 * tile + 0], data[4 * tile + 1], data[4 * tile + 2], data[4 * tile + 3], info);

                    for (var x = 0; x < b; x++) {
                        array.set(i, j, 4 * k + x, w, buf[x]);
                    }
                }
            }
        }
    }

    return array;
}

},{"ndarray":15}],28:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.writeShader = exports.readShader = undefined;

var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

exports.init = init;
exports.pack = pack;
exports.unpack = unpack;

var _ndarray = require('ndarray');

var _ndarray2 = _interopRequireDefault(_ndarray);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var readShader = exports.readShader = 'uniform sampler2D @tex;\nuniform ivec2 @texSize;\nuniform ivec4 @shape;\nuniform int @cols;\n\n\nvec4 @read4(ivec4 pos){\n\tfloat tile = float(vec2tile(pos.zw / ivec2(4, 1), ceildiv(@shape.z, 4)));\n    return @decode4(texture2D(@tex, (\n        vec2(\n        \tmod(tile, float(@cols)),\n        \tfloor(tile / float(@cols))\n        ) * vec2(@shape.xy) +\n        vec2(pos.xy) + vec2(0.5, 0.5)\n    ) / vec2(@texSize)));\n}\n';
var writeShader = exports.writeShader = 'uniform ivec2 @texSize;\nuniform ivec4 @shape;\nuniform int @cols;\n\nvec4 process4(ivec4 pos);\nvoid main(){\n    int tile = vec2tile(ivec2(gl_FragCoord.xy) / @shape.xy, @cols);\n    int chunks = ceildiv(@shape.z, 4);\n    if(tile * 4 >= @shape.z * @shape.w){ checkerboard(); return; }\n    gl_FragColor = @encode4(@activation4(process4(ivec4(\n        mod(gl_FragCoord.xy, vec2(@shape.xy)), \n        tile2vec(tile, chunks) * ivec2(4, 1)))));\n}\n\n';

function init(shape) {
    var width = shape[0]; // var width = shape[0] * 4;    
    // we pick the number of columns so we can keep
    // the texture as square as possible, with the
    // minimal amount of wasted space.

    var tiles = Math.ceil(shape[2] / 4) * shape[3],
        cols = Math.max(1, Math.min(tiles, Math.round(Math.sqrt(shape[0] * shape[1] * tiles) / width)));

    var texSize = [width * cols, shape[1] * Math.ceil(tiles / cols)];

    return {
        texSize: texSize,
        cols: cols,
        shape: shape
    };
}

function pack(info, array, encode4, format) {
    array = (0, _ndarray2.default)(array.data, array.shape.concat([1, 1, 1, 1]).slice(0, 4), array.stride.concat([1, 1, 1, 1]).slice(0, 4), array.offset);

    var shape = array.shape,
        tiles = Math.ceil(shape[2] / 4) * shape[3],
        tw = shape[0],
        th = shape[1],
        cols = info.cols,
        _info$texSize = _slicedToArray(info.texSize, 2),
        width = _info$texSize[0],
        height = _info$texSize[1],
        chunks = Math.ceil(shape[2] / 4),
        length = width * height * 4;


    if (format.type === 'float32') {
        var data = new Float32Array(length);
    } else if (format.type === 'uint8') {
        var data = new Uint8Array(length);
    }

    for (var z = 0; z < chunks; z++) {
        for (var w = 0; w < shape[3]; w++) {
            var tile = w * chunks + z;
            var b = Math.min(z * 4 + 4, shape[2]) - z * 4;

            var ih = th * Math.floor(tile / cols);
            var jw = tw * (tile % cols);

            for (var i = 0; i < tw; i++) {
                for (var j = 0; j < th; j++) {

                    var pos = 4 * ((ih + j) * width + jw + i);
                    encode4(data.subarray(pos, pos + 4), b < 1 ? 0 : array.get(i, j, 4 * z + 0, w), b < 2 ? 0 : array.get(i, j, 4 * z + 1, w), b < 3 ? 0 : array.get(i, j, 4 * z + 2, w), b < 4 ? 0 : array.get(i, j, 4 * z + 3, w), info);
                }
            }
        }
    }
    return data;
}

function unpack(info, data, decode4, type) {
    throw new Error("not implemented: format/4-4/pack/tile/index.js:unpack");
}

},{"ndarray":15}],29:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
	value: true
});

var _index = require('./4-4/index.js');

var _index2 = _interopRequireDefault(_index);

var _index3 = require('./1-4/index.js');

var _index4 = _interopRequireDefault(_index3);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

exports.default = {
	'4:4': _index2.default,
	'1:4': _index4.default
};

},{"./1-4/index.js":20,"./4-4/index.js":26}],30:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _index = require('./tensor/index.js');

Object.defineProperty(exports, 'Tensor', {
  enumerable: true,
  get: function get() {
    return _index.Tensor;
  }
});
Object.defineProperty(exports, 'OutputTensor', {
  enumerable: true,
  get: function get() {
    return _index.OutputTensor;
  }
});
Object.defineProperty(exports, 'InPlaceTensor', {
  enumerable: true,
  get: function get() {
    return _index.InPlaceTensor;
  }
});

var _index2 = require('./runtime/index.js');

Object.defineProperty(exports, 'Run', {
  enumerable: true,
  get: function get() {
    return _index2.Run;
  }
});
Object.defineProperty(exports, 'Compile', {
  enumerable: true,
  get: function get() {
    return _index2.Compile;
  }
});

var _util = require('./util.js');

Object.defineProperty(exports, 'createGL', {
  enumerable: true,
  get: function get() {
    return _util.createGL;
  }
});

},{"./runtime/index.js":33,"./tensor/index.js":40,"./util.js":42}],31:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.checkLinkError = checkLinkError;
exports.checkShaderError = checkShaderError;
exports.checkFramebufferError = checkFramebufferError;
// code for pretty printing shader errors from regl

function checkLinkError(gl, program, fragShader, vertShader, command) {
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        var errLog = gl.getProgramInfoLog(program);
        var fragParse = parseSource(fragShader, command);
        var vertParse = parseSource(vertShader, command);

        var header = 'Error linking program with vertex shader, "' + vertParse[0].name + '", and fragment shader "' + fragParse[0].name + '"';

        if (typeof document !== 'undefined') {
            console.log('%c' + header + '\n%c' + errLog, 'color:red;text-decoration:underline;font-weight:bold', 'color:red');
        } else {
            console.log(header + '\n' + errLog);
        }

        console.log(fragShader);

        throw new Error(header);
    }
}

function checkShaderError(gl, shader, source, type, command) {
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        var errLog = gl.getShaderInfoLog(shader);
        var typeName = type === gl.FRAGMENT_SHADER ? 'fragment' : 'vertex';
        // checkCommandType(source, 'string', typeName + ' shader source must be a string', command)

        var files = parseSource(source, command);
        var errors = parseErrorLog(errLog);
        annotateFiles(files, errors);

        Object.keys(files).forEach(function (fileNumber) {
            var file = files[fileNumber];
            if (!file.hasErrors) {
                return;
            }

            var strings = [''];
            var styles = [''];

            function push(str, style) {
                strings.push(str);
                styles.push(style || '');
            }

            push('file number ' + fileNumber + ': ' + file.name + '\n', 'color:red;text-decoration:underline;font-weight:bold');

            file.lines.forEach(function (line) {
                if (line.errors.length > 0) {
                    push(leftPad(line.number, 4) + '|  ', 'background-color:yellow; font-weight:bold');
                    push(line.line + '\n', 'color:red; background-color:yellow; font-weight:bold');

                    // try to guess token
                    var offset = 0;
                    line.errors.forEach(function (error) {
                        var message = error.message;
                        var token = /^\s*\'(.*)\'\s*\:\s*(.*)$/.exec(message);
                        if (token) {
                            var tokenPat = token[1];
                            message = token[2];
                            switch (tokenPat) {
                                case 'assign':
                                    tokenPat = '=';
                                    break;
                            }
                            offset = Math.max(line.line.indexOf(tokenPat, offset), 0);
                        } else {
                            offset = 0;
                        }

                        push(leftPad('| ', 6));
                        push(leftPad('^^^', offset + 3) + '\n', 'font-weight:bold');
                        push(leftPad('| ', 6));
                        push(message + '\n', 'font-weight:bold');
                    });
                    push(leftPad('| ', 6) + '\n');
                } else {
                    push(leftPad(line.number, 4) + '|  ');
                    push(line.line + '\n', 'color:red');
                }
            });
            if (typeof document !== 'undefined') {
                styles[0] = strings.join('%c');
                console.log.apply(console, styles);
            } else {
                console.log(strings.join(''));
            }
        });

        throw new Error('Error compiling ' + typeName + ' shader, ' + files[0].name);
    }
}

function checkFramebufferError(gl) {

    var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status != gl.FRAMEBUFFER_COMPLETE) {
        var statusCode = {};
        statusCode[gl.FRAMEBUFFER_COMPLETE] = 'complete';
        statusCode[gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT] = 'incomplete attachment';
        statusCode[gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS] = 'incomplete dimensions';
        statusCode[gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT] = 'incomplete, missing attachment';
        statusCode[gl.FRAMEBUFFER_UNSUPPORTED] = 'unsupported';
        throw new Error('framebuffer configuration not supported, status = ' + statusCode[status]);
    }
}

function leftPad(str, n) {
    str = str + '';
    while (str.length < n) {
        str = ' ' + str;
    }
    return str;
}

function ShaderFile() {
    this.name = 'unknown';
    this.lines = [];
    this.index = {};
    this.hasErrors = false;
}

function ShaderLine(number, line) {
    this.number = number;
    this.line = line;
    this.errors = [];
}

function ShaderError(fileNumber, lineNumber, message) {
    this.file = fileNumber;
    this.line = lineNumber;
    this.message = message;
}

function parseSource(source, command) {
    var lines = source.split('\n');
    var lineNumber = 1;
    var fileNumber = 0;
    var files = {
        unknown: new ShaderFile(),
        0: new ShaderFile()
    };
    files.unknown.name = files[0].name = 'unknown';
    files.unknown.lines.push(new ShaderLine(0, ''));
    for (var i = 0; i < lines.length; ++i) {
        var line = lines[i];
        var parts = /^\s*\#\s*(\w+)\s+(.+)\s*$/.exec(line);
        if (parts) {
            switch (parts[1]) {
                case 'line':
                    var lineNumberInfo = /(\d+)(\s+\d+)?/.exec(parts[2]);
                    if (lineNumberInfo) {
                        lineNumber = lineNumberInfo[1] | 0;
                        if (lineNumberInfo[2]) {
                            fileNumber = lineNumberInfo[2] | 0;
                            if (!(fileNumber in files)) {
                                files[fileNumber] = new ShaderFile();
                            }
                        }
                    }
                    break;
                case 'define':
                    var nameInfo = /SHADER_NAME(_B64)?\s+(.*)$/.exec(parts[2]);
                    if (nameInfo) {
                        files[fileNumber].name = nameInfo[1] ? decodeB64(nameInfo[2]) : nameInfo[2];
                    }
                    break;
            }
        }
        files[fileNumber].lines.push(new ShaderLine(lineNumber++, line));
    }
    Object.keys(files).forEach(function (fileNumber) {
        var file = files[fileNumber];
        file.lines.forEach(function (line) {
            file.index[line.number] = line;
        });
    });
    return files;
}

function parseErrorLog(errLog) {
    var result = [];
    errLog.split('\n').forEach(function (errMsg) {
        if (errMsg.length < 5) {
            return;
        }
        var parts = /^ERROR\:\s+(\d+)\:(\d+)\:\s*(.*)$/.exec(errMsg);
        if (parts) {
            result.push(new ShaderError(parts[1] | 0, parts[2] | 0, parts[3].trim()));
        } else if (errMsg.length > 0) {
            result.push(new ShaderError('unknown', 0, errMsg));
        }
    });
    return result;
}

function annotateFiles(files, errors) {
    errors.forEach(function (error) {
        var file = files[error.file];
        if (file) {
            var line = file.index[error.line];
            if (line) {
                line.errors.push(error);
                file.hasErrors = true;
                return;
            }
        }
        files.unknown.hasErrors = true;
        files.unknown.lines[0].errors.push(error);
    });
}

},{}],32:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.default = assembleFragmentShader;

var _base = require('../tensor/base.js');

var _base2 = _interopRequireDefault(_base);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var TENSOR_FRAGMENT_HEADER = 'precision highp int;\nprecision highp float;\n\nint   imod(int f, int p){ return f - p * (f / p); }\nint   vec2tile(ivec2 v, int rows){ return rows * v.y + v.x; }\nivec2 tile2vec(int f, int rows){ return ivec2(imod(f, rows), f / rows); }\nint   ceildiv(int a, int b){ return (a - 1) / b + 1; }\nvoid  checkerboard(){ gl_FragColor = vec4(mod(gl_FragCoord.x - gl_FragCoord.y, 2.0), 0.2, 0.1, 1); }\n\nfloat chsel(vec4 val, int ch){\n\tif(ch == 0) return val.r;\n\tif(ch == 1) return val.g;\n\tif(ch == 2) return val.b;\n\treturn val.a;\n}\n'; // import { Tensor, OutputTensor, InPlaceTensor } from '../tensor/index.js'
function assembleFragmentShader(shaderGen, output, uniforms) {
    var tensorShader = shaderGen(uniforms, output);

    var fragmentShader = TENSOR_FRAGMENT_HEADER;
    for (var uniform in uniforms) {
        if (uniforms[uniform] instanceof _base2.default) {
            var tensor = uniforms[uniform];

            fragmentShader += tensor._format.codec.decodeShader.replace(/@/g, uniform + '_') + '\n';
            fragmentShader += tensor._format.pack.readShader.replace(/@/g, uniform + '_') + '\n';

            if (tensor.format.density == '1:4' && new RegExp(uniform + '_read4\\b').test(tensorShader) || tensor.format.density == '4:4' && new RegExp(uniform + '_read\\b').test(tensorShader)) {
                fragmentShader += tensor._format.read_shim.replace(/@/g, uniform + '_') + '\n';
            }
        }
    }

    var activation = typeof uniforms._activation == 'string' && uniforms._activation != 'linear' ? uniforms._activation.toLowerCase() : 'linear';

    if (!(activation in output._format.activations)) throw new Error('Unknown activation type ' + activation);

    fragmentShader += output._format.activations[activation].replace(/@/g, 'out_') + '\n';
    fragmentShader += output._format.codec.encodeShader.replace(/@/g, 'out_') + '\n';
    fragmentShader += output._format.pack.writeShader.replace(/@/g, 'out_') + '\n';

    if (output.format.density == '1:4' && /process4\b/.test(tensorShader) || output.format.density == '4:4' && /process\b/.test(tensorShader)) {
        fragmentShader += output._format.write_shim.replace(/@/g, 'out_') + '\n';
    }

    fragmentShader += tensorShader.replace(/@/g, 'out_');

    // console.log(fragmentShader)

    return fragmentShader;
}

},{"../tensor/base.js":37}],33:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.Compile = Compile;
exports.Run = Run;

var _program = require('./program.js');

var _program2 = _interopRequireDefault(_program);

var _frag = require('./frag.js');

var _frag2 = _interopRequireDefault(_frag);

var _index = require('../tensor/index.js');

var _check = require('./check.js');

var _tnsl = require('./tnsl.js');

var _tnsl2 = _interopRequireDefault(_tnsl);

var _timer = require('./timer.js');

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function Compile(shaderGen, output) {
    var uniforms = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};

    var startTime = (0, _timer.now)();
    if (!(output instanceof _index.OutputTensor)) throw new Error("First argument must be an instance of OutputTensor");

    if (typeof shaderGen === 'string') shaderGen = (0, _tnsl2.default)(shaderGen);

    var gl = output.gl;
    var program = (0, _program2.default)(gl, (0, _frag2.default)(shaderGen, output, uniforms));
    var compileTime = (0, _timer.now)() - startTime;
    // console.log('Compile Time', compileTime)
    return program;
}

function Run(shaderGen, output) {
    var uniforms = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
    var callback = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : null;

    var tp = Compile(shaderGen, output, uniforms);

    var gl = output.gl;

    if (callback && typeof callback != 'function') throw new Error('Callback must be a function');
    if (callback) {
        (0, _timer.beginTimer)(gl, {
            shader: shaderGen,
            output: output
        });
    }

    gl.useProgram(tp.program);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);

    var setUniform = tp.setUniform,
        texIndex = 0,
        mustSwap = false;

    for (var name in uniforms) {
        if (name.startsWith('_')) continue;

        if (name + '_tex' in tp.uniformTypes) {
            var tensor = uniforms[name];
            if (tensor.gl !== output.gl) throw new Error('Uniforms must belong to same GL context as output');
            if (tensor === output) mustSwap = true;

            for (var uniform in tensor.info) {
                setUniform(name + '_' + uniform, tensor.info[uniform]);
            }

            gl.activeTexture(gl['TEXTURE' + texIndex]);
            gl.bindTexture(gl.TEXTURE_2D, tensor.tex);
            setUniform(name + '_tex', texIndex);

            texIndex++;
        } else if (name in tp.uniformTypes) {
            setUniform(name, uniforms[name]);
        } else {
            throw new Error("Unknown uniform " + name);
        }
    }

    // Ordinarily we can't write to the same texture that we're using as
    // an input, as this could lead to all sorts of terrible race conditions,
    // undefined behavior, and invalid state. InPlaceTensors actually consist
    // of a pair of textures which are swapped for these in-place operations. 
    if (mustSwap) output.swap();

    for (var _uniform in output.info) {
        setUniform('out_' + _uniform, output.info[_uniform]);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, output.fbo);
    gl.viewport(0, 0, output.info.texSize[0], output.info.texSize[1]);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); // draw to framebuffer

    (0, _check.checkFramebufferError)(gl);

    // var runTime = now() - startTime;
    // timer.end()
    if (callback) {
        (0, _timer.endTimer)(gl, function (info) {
            // console.log('GPU time: ', info)
            callback(info);
        });
    }
    // console.log('CPU Run Time', runTime)

    return output;
}

},{"../tensor/index.js":40,"./check.js":31,"./frag.js":32,"./program.js":34,"./timer.js":35,"./tnsl.js":36}],34:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.default = getTensorProgram;
exports.bindAttributeBuffer = bindAttributeBuffer;
exports.createShaderProgram = createShaderProgram;

var _check = require('./check.js');

var TENSOR_VERTEX_SHADER = '\n    precision highp float;\n    attribute vec2 a_position;\n    void main() {\n        gl_Position = vec4(a_position, 0, 1);\n    }\n';

var UNIFORM_SETTERS = { vec4: '4fv', vec3: '3fv', vec2: '2fv', float: '1f',
    ivec4: '4iv', ivec3: '3iv', ivec2: '2iv', int: '1i',
    sampler2D: '1i' };

function getTensorProgram(gl, fragmentShader) {
    if (!gl._tensorPrograms) gl._tensorPrograms = {};
    if (fragmentShader in gl._tensorPrograms) {
        return gl._tensorPrograms[fragmentShader];
    }
    var program = createTensorProgram(gl, fragmentShader);
    gl._tensorPrograms[fragmentShader] = program;
    return program;
}

function createTensorProgram(gl, fragmentShader) {
    var program = createShaderProgram(gl, TENSOR_VERTEX_SHADER, fragmentShader);

    gl.useProgram(program);
    bindAttributeBuffer(gl, program);

    var uniformTypes = extractUniformDeclarations(fragmentShader),
        uniformLocs = {};

    function addUniform(name, type) {
        uniformLocs[name] = { loc: gl.getUniformLocation(program, name), type: type };
    }

    for (var name in uniformTypes) {
        var type = uniformTypes[name];
        if (type in UNIFORM_SETTERS) {
            addUniform(name, type);
        } else throw new Error("Unknown uniform type " + type);
    }

    function setUniform(name, value) {
        if (!(name in uniformLocs)) {
            throw new Error("Could not find uniform " + name);
        }
        gl['uniform' + UNIFORM_SETTERS[uniformLocs[name].type]](uniformLocs[name].loc, value);
    }

    return {
        program: program,
        uniformLocs: uniformLocs,
        uniformTypes: uniformTypes,
        setUniform: setUniform
    };
}

function bindAttributeBuffer(gl, program) {
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    var positionLocation = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
}

function extractUniformDeclarations(str) {
    var uniforms = {};
    str = str.replace(/((?:\/\*(?:[^*]|(?:\*+[^*\/]))*\*+\/)|(?:\/\/.*))/g, '');
    str = str.replace(/\/\/.*\n/g, '');
    var m,
        re = /uniform\s*([\w_]+)\s*([\w_]+)/g;
    while (m = re.exec(str)) {
        uniforms[m[2]] = m[1];
    }return uniforms;
}

function createShaderProgram(gl, vertexSource, fragmentSource) {
    var vertexShader = compileShader(gl, vertexSource, gl.VERTEX_SHADER);
    var fragmentShader = compileShader(gl, fragmentSource, gl.FRAGMENT_SHADER);

    // var debug = gl.getExtension('WEBGL_debug_shaders')
    // console.log(debug.getTranslatedShaderSource(vertexShader));
    // console.log(debug.getTranslatedShaderSource(fragmentShader));

    var program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    // interestingly enough it seems like Safari never emits
    // a shader program link error. 
    (0, _check.checkLinkError)(gl, program, fragmentSource, vertexSource);

    return program;
}

function compileShader(gl, shaderSource, shaderType) {
    var shader = gl.createShader(shaderType);
    gl.shaderSource(shader, shaderSource);
    gl.compileShader(shader);
    var success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
    (0, _check.checkShaderError)(gl, shader, shaderSource, shaderType);
    return shader;
}

},{"./check.js":31}],35:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
	value: true
});
exports.now = now;
exports.beginTimer = beginTimer;
exports.endTimer = endTimer;
function now() {
	if (typeof performance === 'undefined') {
		return Date.now();
	} else {
		return performance.now();
	}
}

function getTimer(gl) {
	if (gl.NO_PROFILE) return;
	if (typeof gl.TIMER_POOL === 'undefined') {
		var extTimer = gl.getExtension('ext_disjoint_timer_query');
		if (!extTimer || !extTimer.createQueryEXT) {
			gl.NO_PROFILE = true;
			return;
		}
		gl.TIMER_POOL = createTimer(gl);
	}
	return gl.TIMER_POOL;
}

function beginTimer(gl) {
	var info = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};

	var timer = getTimer(gl);
	if (timer) {
		timer.begin(info);
	}
}

function endTimer(gl, callback) {
	var timer = getTimer(gl);
	if (timer) {
		timer.end(callback);
	} else if (callback) {
		console.warn("Can not trigger callback: implementation does not support ext_disjoint_timer_query");
	}
}

function createTimer(gl) {
	var extTimer = gl.getExtension('ext_disjoint_timer_query');

	var queryPool = [];
	function allocQuery() {
		return queryPool.pop() || extTimer.createQueryEXT();
	}
	function freeQuery(query) {
		queryPool.push(query);
	}

	var pendingQueries = [];
	function beginQuery(info) {
		var query = allocQuery();
		extTimer.beginQueryEXT(extTimer.TIME_ELAPSED_EXT, query);
		pendingQueries.push([query, info]);
	}

	function endQuery() {
		extTimer.endQueryEXT(extTimer.TIME_ELAPSED_EXT);
	}

	function callback(info, time) {
		var fn = info.callback;
		info.gpuTime = time;
		delete info.callback;
		if (fn) fn(info);
	}

	function monitorPending() {
		for (var i = 0; i < pendingQueries.length; ++i) {
			var query = pendingQueries[i][0];
			if (extTimer.getQueryObjectEXT(query, extTimer.QUERY_RESULT_AVAILABLE_EXT)) {
				var queryTime = extTimer.getQueryObjectEXT(query, extTimer.QUERY_RESULT_EXT);
				callback(pendingQueries[i][1], queryTime / 1e6);
				freeQuery(query);
				pendingQueries.splice(i, 1);
				i--;
			}
		}
	}

	var isPolling = false;
	function loop() {
		if (pendingQueries.length > 0) {
			monitorPending();
			requestAnimationFrame(loop);
		} else {
			isPolling = false;
		}
	}

	var currentInfo = null;
	return {
		begin: function begin() {
			var info = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};

			if (currentInfo) throw new Error('beginTimer was called before previous endTimer');
			currentInfo = info;
			info.cpuStartTime = now();
			beginQuery(currentInfo);
		},
		end: function end(fn) {
			currentInfo.cpuTime = now() - currentInfo.cpuStartTime;
			delete currentInfo.cpuStartTime;
			currentInfo.callback = fn;
			currentInfo = null;
			endQuery();

			if (isPolling === false) {
				isPolling = true;
				requestAnimationFrame(loop);
			}
		}
	};
}

},{}],36:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.default = TNSL;
// TNSL (pronounced tinsel)
// is a domain specific language based on GLSL
// for helping with the writing code that
// computes with tensors. 

// A limitation of GLSL is that the condition
// of any loop has to be statically known 
// (e.g. counters up to a fixed constant
// value) which is problematic if we want
// to write general code that depends on
// the size of the input tensors

// TNSL adds the following syntax:
//      #(image.shape)
// which will be replaced with an ivec4
// containing the shape of the input tensor "image"
// automatically

function TNSL(str) {
    if (typeof str != 'string') throw new Error('TNSL shader preprocessor only accepts strings');

    return function (uniforms, output) {
        return str
        // comment out the tensor struct definitions
        .replace(/uniform\s*Tensor\s*([\w_]+)\s*;/g, '/* (Tensor $1) */')

        // this is the macro syntax
        .replace(/\#\(([\w\.\s]+)\)/g, function (all, body) {
            var obj = uniforms;
            var _iteratorNormalCompletion = true;
            var _didIteratorError = false;
            var _iteratorError = undefined;

            try {
                for (var _iterator = body.split('.')[Symbol.iterator](), _step; !(_iteratorNormalCompletion = (_step = _iterator.next()).done); _iteratorNormalCompletion = true) {
                    var part = _step.value;

                    obj = obj[part.trim()];
                }
            } catch (err) {
                _didIteratorError = true;
                _iteratorError = err;
            } finally {
                try {
                    if (!_iteratorNormalCompletion && _iterator.return) {
                        _iterator.return();
                    }
                } finally {
                    if (_didIteratorError) {
                        throw _iteratorError;
                    }
                }
            }

            if (typeof obj == 'number') {
                return obj.toString();
            } else if (Array.isArray(obj) && obj.length <= 4 && obj.length > 1) {
                return (obj.every(Number.isInteger) ? 'i' : '') + 'vec' + obj.length + '(' + obj.join(',') + ')';
            }
            throw new Error('Can not inline expression ' + body);
        })
        // tensor.read4(x, 0) => tensor.read4(ivec4(x, 0, 0, 0))
        // this transformation takes place when there are 2 or more arguments
        // as otherwise it's not possible to statically determine whether x is
        // of type ivec4 or a number
        .replace(/\b(\w+)\s*\.\s*(read4?)\b\s*\(([^\(\)]+)\)/g, function (all, name, prop, arg) {
            if (name in uniforms && uniforms[name].shape) {
                var parts = arg.split(','),
                    padded = parts.concat(['0', '0', '0', '0'].slice(0, 4 - parts.length));
                if (parts.length < 2 || parts.length > 4) return all;
                var vec = 'ivec4(' + padded.join(',') + ')';
                return name + '_' + prop + '(' + vec + ')';
            }
            return all;
        })

        // tensor.shape => tensor_shape
        .replace(/\b(\w+)\s*\.\s*(\w+)\b/g, function (all, name, prop) {
            if (name in uniforms && uniforms[name].shape) {
                return name + '_' + prop;
            }
            return all;
        });
        // .replace(/\#\s*(\w+)\s*\[(.*?)\]/g, function(all, tensor, body){
        //     return tensor + '_read(ivec4(' + body + '))'
        // })
    };
}

},{}],37:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
	value: true
});

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _helpers = require('./helpers.js');

var _index = require('../format/index.js');

var _index2 = _interopRequireDefault(_index);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

// The tensor format is a JSON object that specifies how 
// the tensor is represented as a texture
// it consists of several keys:

//     type: uint8 | float32
//     density: 4:4 | 1:4
//     pack: stride | tile
//     codec: 
//			softfloat | fixnum (1:4)
//          raw | linquant (4:4)

var BaseTensor = function () {
	function BaseTensor() {
		_classCallCheck(this, BaseTensor);
	}

	_createClass(BaseTensor, [{
		key: '_init',

		// we arent using a constructor because we want to be able to run
		// this instanceof OutputTensor from within the Tensor constructor

		value: function _init(gl, format, shape, data) {
			// validate glcontext
			if (!gl.createTexture) throw new Error('Invalid WebGLRenderingContext');
			this.gl = gl;

			// validate shape
			if (!Array.isArray(shape)) throw new Error("shape must be Array");
			if (shape.length > 4) throw new Error("Tensor must have dimension <= 4");
			if (shape.some(function (k) {
				return !isFinite(k) || k < 1 || !Number.isInteger(k);
			})) throw new Error('Invalid shape: ' + shape);
			shape = shape.concat([1, 1, 1, 1]).slice(0, 4);
			this.shape = shape;

			// validate format
			if (!['float32', 'uint8'].includes(format.type)) throw new Error('format.type must be uint8 or float32');
			if (format.density in _index2.default) {
				var fd = _index2.default[format.density];
				if (!(format.pack in fd.pack)) throw new Error('format.pack must be ' + Object.keys(fd.pack).join(' or '));
				if (!(format.codec in fd.codec)) throw new Error('format.codec must be ' + Object.keys(fd.codec).join(' or '));
			} else throw new Error('format.density must be ' + Object.keys(_index2.default).join(' or '));

			this.format = format;

			// calculate texture size
			this.info = Object.assign({}, this._format.pack.init(shape, format), this._format.codec.init(shape, format));
			if (!this.info.texSize) throw new Error('Format did not yield texSize');

			// initialize texture
			this.tex = (0, _helpers.makeTexture)(gl);
			this.update(data);
		}
	}, {
		key: '_update',
		value: function _update(data) {
			if (data !== null) {
				if (this.format.type === 'uint8') {
					if (Array.isArray(data) || data instanceof Uint8ClampedArray) data = new Uint8Array(data);
					if (!(data instanceof Uint8Array)) throw new Error('data must be Uint8Array');
				} else if (this.format.type === 'float32') {
					if (Array.isArray(data) || data instanceof Float64Array) data = new Float32Array(data);
					if (!(data instanceof Float32Array)) throw new Error('data must be Float32Array');
				} else throw new Error('Type must be uint8 or float32');
				if (data.length !== this.info.texSize[0] * this.info.texSize[1] * 4) throw new Error('data is the wrong length');
			}
			// if(data) console.log('_update', data);
			var gl = this.gl;
			gl.bindTexture(gl.TEXTURE_2D, this.tex);
			gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.info.texSize[0], this.info.texSize[1], 0, gl.RGBA, this.format.type == 'uint8' ? gl.UNSIGNED_BYTE : gl.FLOAT, data);
		}
	}, {
		key: 'update',
		value: function update(data) {
			if (!data) return this._update(null);
			if (data.shape) return this._update(this._format.pack.pack(this.info, data, this._format.codec.encode, this.format));
			if (this.type != 'uint8') console.warn('Calling update with raw TypedArray may not work across all browsers.');
			return this._update(data);
		}
	}, {
		key: 'destroy',
		value: function destroy() {
			this.gl.deleteTexture(this.tex);
		}
	}, {
		key: '_format',
		get: function get() {
			return {
				pack: _index2.default[this.format.density].pack[this.format.pack],
				codec: _index2.default[this.format.density].codec[this.format.codec],
				activations: _index2.default[this.format.density].activations,
				read_shim: _index2.default[this.format.density].read_shim,
				write_shim: _index2.default[this.format.density].write_shim
			};
		}
	}]);

	return BaseTensor;
}();

exports.default = BaseTensor;

},{"../format/index.js":29,"./helpers.js":39}],38:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.default = runFeatureTests;
exports.testRenderFloat = testRenderFloat;

var _program = require('../runtime/program.js');

var _helpers = require('./helpers.js');

function runFeatureTests(gl) {

    if (!gl.FLOAT_TEXTURES_TESTED && !gl.NO_FLOAT_TEXTURES) {
        if (!gl.getExtension('OES_texture_float')) {
            console.info("This browser does not seem to support OES_texture_float. " + "Using float codec workaround from now on.");
            gl.NO_FLOAT_TEXTURES = true;
        }
        gl.FLOAT_TEXTURES_TESTED = true;
    }

    if (!gl.NO_FLOAT_TEXTURES) {
        if (!gl.RENDER_FLOAT_TESTED && !gl.NO_RENDER_FLOAT) {
            if (!testRenderFloat(gl)) {
                console.info("This browser supports OES_texture_float, " + "but can not render to floating textures. " + "Using float codec workaround for output tensors from now on.");
                gl.NO_RENDER_FLOAT = true;
            }
            gl.RENDER_FLOAT_TESTED = true;
        }

        if (!gl.READ_FLOAT_TESTED && !gl.NO_READ_FLOAT && !gl.NO_READ_FLOAT) {
            if (!testReadFloat(gl)) {
                console.info("This browser supports OES_texture_float, " + "can render to floating point textures, but can not " + "read into a Float32Array buffer. Using float codec " + "workaround for reading from output tensors from now on.");
                gl.NO_READ_FLOAT = true;
            }
            gl.READ_FLOAT_TESTED = true;
        }
    }
}

var CHECK_FLOAT_VERTEX = '\n    attribute vec2 a_position;\n    void main() {\n        gl_Position = vec4(a_position, 0, 1);\n    }\n';
var CHECK_FLOAT_FRAGMENT = '\n    void main() {\n        gl_FragColor = vec4(3.14159, -2.71828, 1.61828, 42);\n    }\n';

// some browsers (e.g. mobile safari) are capable of initializing floating 
// point textures but unable to write to them. The only way of finding this
// out is by trying to render to a floating point texture and noticing
// the invalid framebuffer status.

function testRenderFloat(gl) {
    var tex = (0, _helpers.makeTexture)(gl);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 10, 10, 0, gl.RGBA, gl.FLOAT, null);
    var fbo = (0, _helpers.makeFrameBuffer)(gl, tex);

    var program = (0, _program.createShaderProgram)(gl, CHECK_FLOAT_VERTEX, CHECK_FLOAT_FRAGMENT);
    gl.useProgram(program);
    (0, _program.bindAttributeBuffer)(gl, program);

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, 10, 10);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.deleteTexture(tex);
    gl.deleteFramebuffer(fbo);
    gl.deleteProgram(program);

    return status == gl.FRAMEBUFFER_COMPLETE;
}

function testReadFloat(gl) {
    var tex = (0, _helpers.makeTexture)(gl);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 10, 10, 0, gl.RGBA, gl.FLOAT, null);
    var fbo = (0, _helpers.makeFrameBuffer)(gl, tex);

    var program = (0, _program.createShaderProgram)(gl, CHECK_FLOAT_VERTEX, CHECK_FLOAT_FRAGMENT);
    gl.useProgram(program);
    (0, _program.bindAttributeBuffer)(gl, program);

    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, 10, 10);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    var size = [3, 3];
    var pixels = pixels = new Float32Array(size[0] * size[1] * 4);
    gl.readPixels(0, 0, size[0], size[1], gl.RGBA, gl.FLOAT, pixels);

    gl.deleteTexture(tex);
    gl.deleteFramebuffer(fbo);
    gl.deleteProgram(program);

    var total_error = Math.abs(pixels[0] - 3.14159) + Math.abs(pixels[1] + 2.71828) + Math.abs(pixels[2] - 1.61828) + Math.abs(pixels[3] - 42);

    return total_error < 0.01;
}

},{"../runtime/program.js":34,"./helpers.js":39}],39:[function(require,module,exports){
"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.makeFrameBuffer = makeFrameBuffer;
exports.makeTexture = makeTexture;
function makeFrameBuffer(gl, texture) {
    var framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    return framebuffer;
}

function makeTexture(gl) {
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    return texture;
}

},{}],40:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.InPlaceTensor = exports.OutputTensor = exports.Tensor = undefined;

var _get = function get(object, property, receiver) { if (object === null) object = Function.prototype; var desc = Object.getOwnPropertyDescriptor(object, property); if (desc === undefined) { var parent = Object.getPrototypeOf(object); if (parent === null) { return undefined; } else { return get(parent, property, receiver); } } else if ("value" in desc) { return desc.value; } else { var getter = desc.get; if (getter === undefined) { return undefined; } return getter.call(receiver); } };

var _typeof = typeof Symbol === "function" && typeof Symbol.iterator === "symbol" ? function (obj) { return typeof obj; } : function (obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; };

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _base = require('./base.js');

var _base2 = _interopRequireDefault(_base);

var _show2 = require('./show.js');

var _show3 = _interopRequireDefault(_show2);

var _feature = require('./feature.js');

var _feature2 = _interopRequireDefault(_feature);

var _helpers = require('./helpers.js');

var _index = require('../runtime/index.js');

var _ndarrayShow = require('ndarray-show');

var _ndarrayShow2 = _interopRequireDefault(_ndarrayShow);

var _ndarray = require('ndarray');

var _ndarray2 = _interopRequireDefault(_ndarray);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var Tensor = exports.Tensor = function (_BaseTensor) {
    _inherits(Tensor, _BaseTensor);

    // new Tensor(gl)
    // new Tensor(gl, [1, 1])
    // new Tensor(gl, [1, 1], null)
    // new Tensor(gl, [1, 1], data)
    // new Tensor(gl, [1, 1], data, { type, pack, codec, density })
    // new Tensor(gl, [1, 1], { type, pack, codec, density })
    // new Tensor(gl, [1, 1], 'softfloat')
    // new Tensor(gl, [1, 1], 'float32')
    // new Tensor(gl, [1, 1], 'uint8')
    // new Tensor(gl, { shape, data })
    // new Tensor(gl, { width, height, data })
    // pix = new Tensor(gl, [1, 1, 4], [1, 0.4, 3, 4], 'uint8')

    function Tensor(gl) {
        var shape = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : [];
        var data = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : null;
        var format = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : null;

        _classCallCheck(this, Tensor);

        var _this = _possibleConstructorReturn(this, (Tensor.__proto__ || Object.getPrototypeOf(Tensor)).call(this));

        (0, _feature2.default)(gl);

        var xdata = data;
        if (shape.shape) {
            // ndarrays
            format = data;
            xdata = shape.data;
            data = shape;
            shape = shape.shape;
        }

        if (shape.width && shape.height && shape.data) {
            // imagedata
            data = shape.data;
            shape = [shape.width, shape.height];
        }

        if (typeof data === 'string') {
            // data = uint8 | float32
            if (format !== null) throw new Error('Format must not be specified if data is a string.');
            format = data;
            data = null;
        } else if (data && (typeof data === 'undefined' ? 'undefined' : _typeof(data)) === 'object' && data.type && data.codec && data.pack && data.density) {
            if (format !== null) throw new Error('Format must not be specified if data is an object.');
            format = data;
            data = null;
        }

        if (format === null) {
            // auto-infer format based on data
            if (data === null) {
                format = 'float32';
            } else if (xdata instanceof Uint8Array || xdata instanceof Uint8ClampedArray) {
                format = 'uint8';
            } else if (xdata instanceof Float32Array || xdata instanceof Float64Array || Array.isArray(xdata)) {
                format = 'float32';
            } else throw new Error("Invalid format for data: must be Uint8Array or Float32Array or ndarray");
        }

        var type = null;
        if (format === 'float32' && (gl.NO_FLOAT_TEXTURES || gl.NO_RENDER_FLOAT && _this instanceof OutputTensor) || format === 'softfloat') {
            format = { type: 'uint8', pack: 'stride', density: '1:4', codec: 'softfloat' };
            type = 'float32';
        } else if (format === 'uint8' || format === 'float32') {
            format = { type: format, pack: 'stride', density: '4:4', codec: 'raw' };
        }

        _this.type = type || format.type;
        _this._init(gl, format, shape, data);
        return _this;
    }

    _createClass(Tensor, [{
        key: 'copy',
        value: function copy() {
            var format = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : this.type;
            var T = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : OutputTensor;

            var TENSOR_IDENTITY = '\n            uniform Tensor image;\n            vec4 process4(ivec4 pos) { return image.read4(pos); }\n        ';
            var out = new T(this.gl, this.shape, format);
            out.run(TENSOR_IDENTITY, { image: this });
            return out;
        }
    }, {
        key: 'withCopy',
        value: function withCopy(fn) {
            for (var _len = arguments.length, args = Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
                args[_key - 1] = arguments[_key];
            }

            var copy = this.copy.apply(this, args);
            var result = fn(copy);
            copy.destroy();
            return result;
        }
    }, {
        key: '_show',
        value: function _show() {
            var opt = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
            (0, _show3.default)(this.gl, this.tex, opt);
        }
    }, {
        key: 'show',
        value: function show() {
            var opt = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};

            var gl = this.gl;
            if (this.format.pack == 'tile' && this.format.density == '4:4' && this.format.codec == 'raw') {
                this._show(opt);
            } else {
                // C.info.main_input.output.copy({ type: 'uint8', pack: 'tile', density: '4:4', codec: 'linquant', min: 0, max: 255 })._show({ })
                this.withCopy(function (x) {
                    return x.show(opt);
                }, { type: gl.NO_FLOAT_TEXTURES || gl.NO_RENDER_FLOAT ? 'uint8' : 'float32',
                    pack: 'tile', density: '4:4', codec: 'raw' });
            };
        }
    }, {
        key: 'run',
        value: function run(shader, params) {
            throw new Error('Only OutputTensor can run shaders.');
        }
    }, {
        key: 'compile',
        value: function compile(shader, params) {
            throw new Error('Only OutputTensor can compile shaders.');
        }
    }, {
        key: 'read',
        value: function read() {
            console.warn("Copying before read...");
            return this.withCopy(function (x) {
                return x.read();
            });
        }
    }, {
        key: 'print',
        value: function print() {
            return (0, _ndarrayShow2.default)(this.read());
        }
    }, {
        key: 'swap',
        value: function swap() {
            throw new Error("Only InPlaceTensor can be both a parameter and destination.");
        }
    }]);

    return Tensor;
}(_base2.default);

var OutputTensor = exports.OutputTensor = function (_Tensor) {
    _inherits(OutputTensor, _Tensor);

    function OutputTensor() {
        var _ref;

        _classCallCheck(this, OutputTensor);

        for (var _len2 = arguments.length, args = Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {
            args[_key2] = arguments[_key2];
        }

        var _this2 = _possibleConstructorReturn(this, (_ref = OutputTensor.__proto__ || Object.getPrototypeOf(OutputTensor)).call.apply(_ref, [this].concat(args)));

        _this2.fbo = (0, _helpers.makeFrameBuffer)(_this2.gl, _this2.tex);
        return _this2;
    }

    _createClass(OutputTensor, [{
        key: 'destroy',
        value: function destroy() {
            _get(OutputTensor.prototype.__proto__ || Object.getPrototypeOf(OutputTensor.prototype), 'destroy', this).call(this);
            this.gl.deleteFramebuffer(this.fbo);
        }
    }, {
        key: '_read',
        value: function _read() {
            var gl = this.gl,
                size = this.info.texSize;

            if (this.format.type == 'uint8') {
                var glType = gl.UNSIGNED_BYTE,
                    pixels = new Uint8Array(size[0] * size[1] * 4);
            } else if (this.format.type === 'float32') {
                var glType = gl.FLOAT,
                    pixels = new Float32Array(size[0] * size[1] * 4);
            }

            gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
            gl.readPixels(0, 0, size[0], size[1], gl.RGBA, glType, pixels);

            // console.log('___read', pixels)
            return pixels;
        }
    }, {
        key: 'run',
        value: function run(shader, params, callback) {
            return (0, _index.Run)(shader, this, params, callback);
        }
    }, {
        key: 'compile',
        value: function compile(shader, params) {
            return (0, _index.Compile)(shader, this, params);
        }
    }, {
        key: 'read',
        value: function read() {
            if (this.format.type === 'float32' && this.gl.NO_READ_FLOAT) {
                return this.withCopy(function (x) {
                    return x.read();
                }, 'softfloat');
            }

            var array = this._format.pack.unpack(this.info, this._read(), this._format.codec.decode, this.type);

            // strip trailing singleton dimensions
            var shape = array.shape.slice(0),
                stride = array.stride.slice(0);
            while (shape[shape.length - 1] == 1 && shape.length > 1) {
                shape.pop();
                stride.pop();
            }
            return (0, _ndarray2.default)(array.data, shape, stride, array.offset);
        }
    }]);

    return OutputTensor;
}(Tensor);

var InPlaceTensor = exports.InPlaceTensor = function (_OutputTensor) {
    _inherits(InPlaceTensor, _OutputTensor);

    function InPlaceTensor() {
        var _ref2;

        _classCallCheck(this, InPlaceTensor);

        for (var _len3 = arguments.length, args = Array(_len3), _key3 = 0; _key3 < _len3; _key3++) {
            args[_key3] = arguments[_key3];
        }

        var _this3 = _possibleConstructorReturn(this, (_ref2 = InPlaceTensor.__proto__ || Object.getPrototypeOf(InPlaceTensor)).call.apply(_ref2, [this].concat(args)));

        _this3.tex2 = _this3.tex;
        _this3.tex = (0, _helpers.makeTexture)(_this3.gl);
        _this3.update(null);
        _this3.swap();
        return _this3;
    }

    _createClass(InPlaceTensor, [{
        key: 'destroy',
        value: function destroy() {
            _get(InPlaceTensor.prototype.__proto__ || Object.getPrototypeOf(InPlaceTensor.prototype), 'destroy', this).call(this);
            this.gl.deleteTexture(this.tex2);
        }
    }, {
        key: 'swap',
        value: function swap() {
            var tmp = this.tex;
            this.tex = this.tex2;
            this.tex2 = tmp;

            // TODO: investigate performance of using multiple FBOs instead
            // of rebinding the framebuffer
            var gl = this.gl;
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, 0);
        }
    }]);

    return InPlaceTensor;
}(OutputTensor);

},{"../runtime/index.js":33,"./base.js":37,"./feature.js":38,"./helpers.js":39,"./show.js":41,"ndarray":15,"ndarray-show":14}],41:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.default = showTexture;

var _program = require('../runtime/program.js');

var SHOW_TEXTURE_VERTEX = '\n    attribute vec2 a_position;\n    varying mediump vec2 pos;\n    void main() {\n        pos = (a_position + vec2(1, 1)) / 2.0;\n        gl_Position = vec4(a_position, 0, 1);\n    }\n';

var SHOW_TEXTURE_FRAGMENT = '\n    precision mediump float;\n\n    uniform sampler2D tex;\n    uniform float scale;\n    uniform float offset;\n    uniform bool transpose;\n    uniform bool flipX;\n    uniform bool flipY;\n    uniform int channels;\n\n    varying vec2 pos;\n\n    vec4 colormap(float x) {\n        float r = clamp(8.0 / 3.0 * x, 0.0, 1.0);\n        float g = clamp(8.0 / 3.0 * x - 1.0, 0.0, 1.0);\n        float b = clamp(4.0 * x - 3.0, 0.0, 1.0);\n        return vec4(r, g, b, 1.0);\n    }\n\n    void main() {\n        vec2 p = pos;\n        if(flipX) p.x = 1.0 - p.x;\n        if(flipY) p.y = 1.0 - p.y;\n        if(transpose) p = p.yx;\n        if(channels == 4){\n            gl_FragColor = vec4(vec4(offset, offset, offset, offset) \n                + scale * texture2D(tex, p));\n        }else if(channels == 3){\n            gl_FragColor = vec4(vec3(offset, offset, offset) \n                + scale * texture2D(tex, p).rgb, 1);\n        }else if(channels == 2){\n            gl_FragColor = vec4(vec2(offset, offset) \n                + scale * texture2D(tex, p).rg, 0, 1);\n        }else if(channels == 1){\n            gl_FragColor = colormap(offset + scale * texture2D(tex, p).r);\n        }\n    }\n';

function showTexture(gl, tex) {
    var opt = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};

    if (!gl._showProgram) {
        gl._showProgram = (0, _program.createShaderProgram)(gl, SHOW_TEXTURE_VERTEX, SHOW_TEXTURE_FRAGMENT);
        gl.useProgram(gl._showProgram);
        (0, _program.bindAttributeBuffer)(gl, gl._showProgram);
        gl.uniform1i(gl.getUniformLocation(gl._showProgram, 'tex'), 0);
    }

    if (gl.canvas && gl.canvas._tfAuto) {
        gl.canvas.style.display = 'block';
        gl.canvas.style.position = 'absolute';
        gl.canvas.style.top = 0;
        gl.canvas.style.left = 0;
        gl.canvas.style.width = Math.min(innerHeight, innerWidth) + 'px';
        gl.canvas.style.height = Math.min(innerHeight, innerWidth) + 'px';
    }

    gl.useProgram(gl._showProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1f(gl.getUniformLocation(gl._showProgram, 'scale'), opt.scale || 1);
    gl.uniform1f(gl.getUniformLocation(gl._showProgram, 'offset'), opt.offset || 0);
    gl.uniform1i(gl.getUniformLocation(gl._showProgram, 'transpose'), opt.transpose ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(gl._showProgram, 'flipX'), opt.flipX ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(gl._showProgram, 'flipY'), opt.flipY ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(gl._showProgram, 'channels'), opt.channels || 3);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

},{"../runtime/program.js":34}],42:[function(require,module,exports){
'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.createGL = createGL;
function createGL(canvas) {
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        canvas.style.display = 'none';
        canvas._tfAuto = true;
        document.body.appendChild(canvas);
    }
    var gl = canvas.getContext("webgl", { antialias: false }) || canvas.getContext("experimental-webgl", { antialias: false });
    if (!gl) alert('Could not initialize WebGL, try another browser');
    return gl;
}

},{}]},{},[1])
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJjbGllbnQvY2xpZW50LmpzIiwibGliL0xheWVycy5qcyIsImxpYi9Nb2RlbC5qcyIsImxpYi9sYXllcnMvQWN0aXZhdGlvbnMuanMiLCJsaWIvbGF5ZXJzL0Nyb3NzRW50cm9weS5qcyIsImxpYi9sYXllcnMvRGVuc2UuanMiLCJsaWIvbGF5ZXJzL01TRS5qcyIsImxpYi9sYXllcnMvT3V0cHV0LmpzIiwibGliL2xheWVycy9Tb2Z0bWF4LmpzIiwibGliL3V0aWwvZ2VuZXJhdGVXZWlnaHRzLmpzIiwibm9kZV9tb2R1bGVzL2ZpeGVkLXdpZHRoLWZsb2F0L2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL2lvdGEtYXJyYXkvaW90YS5qcyIsIm5vZGVfbW9kdWxlcy9pcy1idWZmZXIvaW5kZXguanMiLCJub2RlX21vZHVsZXMvbmRhcnJheS1zaG93L2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL25kYXJyYXkvbmRhcnJheS5qcyIsIm5vZGVfbW9kdWxlcy9zcHJpbnRmL2xpYi9zcHJpbnRmLmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC8xLTQvYWN0aXZhdGlvbi9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9mb3JtYXQvMS00L2NvZGVjL2ZpeG51bS9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9mb3JtYXQvMS00L2NvZGVjL3NvZnRmbG9hdC9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9mb3JtYXQvMS00L2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC8xLTQvcGFjay9zdHJpZGUvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvZm9ybWF0LzEtNC9wYWNrL3RpbGUvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvZm9ybWF0LzQtNC9hY3RpdmF0aW9uL2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC80LTQvY29kZWMvbGlucXVhbnQvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvZm9ybWF0LzQtNC9jb2RlYy9yYXcvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvZm9ybWF0LzQtNC9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9mb3JtYXQvNC00L3BhY2svc3RyaWRlL2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC80LTQvcGFjay90aWxlL2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9ydW50aW1lL2NoZWNrLmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL3J1bnRpbWUvZnJhZy5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9ydW50aW1lL2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL3J1bnRpbWUvcHJvZ3JhbS5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9ydW50aW1lL3RpbWVyLmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL3J1bnRpbWUvdG5zbC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy90ZW5zb3IvYmFzZS5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy90ZW5zb3IvZmVhdHVyZS5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy90ZW5zb3IvaGVscGVycy5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy90ZW5zb3IvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvdGVuc29yL3Nob3cuanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvdXRpbC5qcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7O0FDQUEsSUFBSSxRQUFRLFFBQVosQUFBWSxBQUFRO0lBQ25CLEtBQUssUUFETixBQUNNLEFBQVE7SUFDYixLQUFLLEdBRk4sQUFFTSxBQUFHOztBQUVULFNBQUEsQUFBUyxJQUFULEFBQWEsTUFBYixBQUFtQixjQUFuQixBQUFpQyxVQUFVLEFBQzFDO0tBQUksSUFBSSxJQUFSLEFBQVEsQUFBSSxBQUNaO0dBQUEsQUFBRSxxQkFBcUIsWUFBWSxBQUNsQztNQUFJLEVBQUEsQUFBRSxlQUFlLGVBQWpCLEFBQWdDLFFBQVEsRUFBQSxBQUFFLFdBQTlDLEFBQXlELEtBQUssQUFDN0Q7WUFBUyxFQUFULEFBQVcsQUFDWDtBQUNEO0FBSkQsQUFLQTtHQUFBLEFBQUUsS0FBRixBQUFPLE9BQVAsQUFBYyxBQUNkO0dBQUEsQUFBRSxlQUFGLEFBQWlCLEFBQ2pCO0dBQUEsQUFBRSxBQUNGOzs7QUFFRCxTQUFBLEFBQVMsSUFBVCxBQUFhLE1BQWIsQUFBbUIsYUFBbkIsQUFBZ0MsTUFBaEMsQUFBc0MsVUFBVSxBQUMvQztLQUFJLElBQUksSUFBUixBQUFRLEFBQUksQUFDWjtHQUFBLEFBQUUscUJBQXFCLFlBQVksQUFDbEM7TUFBSSxFQUFBLEFBQUUsZUFBZSxlQUFqQixBQUFnQyxRQUFRLEVBQUEsQUFBRSxXQUE5QyxBQUF5RCxLQUFLLEFBQzdEO09BQUEsQUFBSSxVQUFVLFNBQVMsRUFBVCxBQUFXLEFBQ3pCO0FBQ0Q7QUFKRCxBQUtBO0dBQUEsQUFBRSxLQUFGLEFBQU8sT0FBUCxBQUFjLEFBQ2Q7S0FBQSxBQUFJLFVBQVUsRUFBQSxBQUFFLGVBQUYsQUFBaUIsQUFDL0I7R0FBQSxBQUFFLGlCQUFGLEFBQW1CLGdCQUFuQixBQUFtQyxBQUNuQztHQUFBLEFBQUUsS0FBRixBQUFPLEFBQ1A7OztBQUVELFNBQUEsQUFBUyxLQUFULEFBQWMs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QUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FDckZBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FDVkE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FDckJBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQ3REQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQ3ZWQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FB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RUFBMkIsRUFBRSxXQUFXLEtBQWIsRUFBM0IsS0FDQSxPQUFPLFVBQVAsQ0FBa0Isb0JBQWxCLEVBQXdDLEVBQUUsV0FBVyxLQUFiLEVBQXhDLENBRFQ7QUFFQSxRQUFJLENBQUMsRUFBTCxFQUFTLE1BQU0saURBQU47QUFDVCxXQUFPLEVBQVA7QUFDSCIsImZpbGUiOiJnZW5lcmF0ZWQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlc0NvbnRlbnQiOlsiKGZ1bmN0aW9uIGUodCxuLHIpe2Z1bmN0aW9uIHMobyx1KXtpZighbltvXSl7aWYoIXRbb10pe3ZhciBhPXR5cGVvZiByZXF1aXJlPT1cImZ1bmN0aW9uXCImJnJlcXVpcmU7aWYoIXUmJmEpcmV0dXJuIGEobywhMCk7aWYoaSlyZXR1cm4gaShvLCEwKTt2YXIgZj1uZXcgRXJyb3IoXCJDYW5ub3QgZmluZCBtb2R1bGUgJ1wiK28rXCInXCIpO3Rocm93IGYuY29kZT1cIk1PRFVMRV9OT1RfRk9VTkRcIixmfXZhciBsPW5bb109e2V4cG9ydHM6e319O3Rbb11bMF0uY2FsbChsLmV4cG9ydHMsZnVuY3Rpb24oZSl7dmFyIG49dFtvXVsxXVtlXTtyZXR1cm4gcyhuP246ZSl9LGwsbC5leHBvcnRzLGUsdCxuLHIpfXJldHVybiBuW29dLmV4cG9ydHN9dmFyIGk9dHlwZW9mIHJlcXVpcmU9PVwiZnVuY3Rpb25cIiYmcmVxdWlyZTtmb3IodmFyIG89MDtvPHIubGVuZ3RoO28rKylzKHJbb10pO3JldHVybiBzfSkiLCJ2YXIgTW9kZWwgPSByZXF1aXJlKFwiLi4vbGliL01vZGVsXCIpLFxuXHRURiA9IHJlcXVpcmUoXCIuLi9ub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvaW5kZXhcIiksXG5cdEdMID0gVEYuY3JlYXRlR0woKTtcblxuZnVuY3Rpb24gR0VUKHBhdGgsIHJlc3BvbnNlVHlwZSwgY2FsbGJhY2spIHtcblx0dmFyIHIgPSBuZXcgWE1MSHR0cFJlcXVlc3QoKTtcblx0ci5vbnJlYWR5c3RhdGVjaGFuZ2UgPSBmdW5jdGlvbiAoKSB7XG5cdFx0aWYgKHIucmVhZHlTdGF0ZSA9PT0gWE1MSHR0cFJlcXVlc3QuRE9ORSAmJiByLnN0YXR1cyA9PT0gMjAwKSB7XG5cdFx0XHRjYWxsYmFjayhyLnJlc3BvbnNlKTtcblx0XHR9XG5cdH07XG5cdHIub3BlbihcIkdFVFwiLCBwYXRoKTtcblx0ci5yZXNwb25zZVR5cGUgPSByZXNwb25zZVR5cGU7XG5cdHIuc2VuZCgpO1xufVxuXG5mdW5jdGlvbiBQVVQocGF0aCwgY29udGVudFR5cGUsIGJvZHksIGNhbGxiYWNrKSB7XG5cdHZhciByID0gbmV3IFhNTEh0dHBSZXF1ZXN0KCk7XG5cdHIub25yZWFkeXN0YXRlY2hhbmdlID0gZnVuY3Rpb24gKCkge1xuXHRcdGlmIChyLnJlYWR5U3RhdGUgPT09IFhNTEh0dHBSZXF1ZXN0LkRPTkUgJiYgci5zdGF0dXMgPT09IDIwMCkge1xuXHRcdFx0aWYgKGNhbGxiYWNrKSBjYWxsYmFjayhyLnJlc3BvbnNlKTtcblx0XHR9XG5cdH1cblx0ci5vcGVuKFwiUFVUXCIsIHBhdGgpO1xuXHRpZiAoY2FsbGJhY2spIHIucmVzcG9uc2VUeXBlID0gY29udGVudFR5cGU7XG5cdHIuc2V0UmVxdWVzdEhlYWRlcihcIkNvbnRlbnQtVHlwZVwiLCBjb250ZW50VHlwZSk7XG5cdHIuc2VuZChib2R5KTtcbn1cblxuZnVuY3Rpb24gUE9TVChwYXRoLCBjb250ZW50VHlwZSwgYm9keSkge1xuXHR2YXIgciA9IG5ldyBYTUxIdHRwUmVxdWVzdCgpO1xuXHRyLm9ucmVhZHlzdGF0ZWNoYW5nZSA9IGZ1bmN0aW9uICgpIHtcblx0XHRpZiAoci5yZWFkeVN0YXRlID09PSBYTUxIdHRwUmVxdWVzdC5ET05FICYmIHIuc3RhdHVzICE9PSAyMDApIHtcblx0XHRcdC8vIFRPRE8gLSByZXNlbmQgb3Igc2F2ZSB0byBsb2NhbD9cblx0XHR9XG5cdH1cblx0ci5vcGVuKFwiUE9TVFwiLCBwYXRoKTtcblx0aWYgKGNvbnRlbnRUeXBlICE9PSB1bmRlZmluZWQpXG5cdFx0ci5zZXRSZXF1ZXN0SGVhZGVyKFwiQ29udGVudC1UeXBlXCIsIGNvbnRlbnRUeXBlKTtcblx0aWYgKGJvZHkgIT09IHVuZGVmaW5lZClcblx0XHRyLnNlbmQoYm9keSk7XG5cdGVsc2Vcblx0XHRyLnNlbmQoKTtcbn1cblxuLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy9cbi8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vXG4vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vL1xuLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy9cblxuXG4vKlxuXG5cdDEuIEdldCBtb2RlbCBmcm9tIHNlcnZlclxuXHQyLiBHZXQgd2VpZ2h0cyBmcm9tIHNlcnZlclxuXHQzLiBHZXQgZGF0YSBmcm9tIHNlcnZlclxuXHQ0LiBUcmFpbiBhbmQgcmV0dXJuIHVwZGF0ZXNcblxuXG4qL1xuXG4oZnVuY3Rpb24gbWFpbigpIHtcblx0dmFyIHJ1biA9IHRydWUsXG5cdFx0bmV0LFxuXHRcdG1vZGVsLFxuXHRcdGl0ZXJhdGlvbnMsXG5cdFx0dGltZXMgPSB7XG5cdFx0XHRyZXF1ZXN0ZWQ6IG51bGwsXHQvLyBtb2RlbCB3ZWlnaHRzIGFuZCBkYXRhIHJlcXVlc3Qgc2VudFxuXHRcdFx0cmVjaWV2ZWQ6IG51bGwsXHRcdC8vIG1vZGVsIGRhdGEgJiB3ZWlnaHRzIHJlY2lldmVkIGZyb20gc2VydmVyXG5cdFx0XHRsb2FkZWQ6IG51bGwsIFx0XHQvLyBtb2RlbCB3ZWlnaHRzIGxvYWRlZFxuXHRcdFx0dHJhaW5lZDogbnVsbCwgXHRcdC8vIG1vZGVsIGZpbmlzaGVkIHRyYWluaW5nXG5cdFx0XHR1cGRhdGVkOiBudWxsIFx0XHQvLyBtb2RlbCB1cGRhdGVzIHNlbnRcblx0XHR9O1xuXG5cdE1vZGVsID0gTW9kZWwoVEYsIEdMKTtcblxuXHRmdW5jdGlvbiB1cGRhdGUoYXJyYXlidWZmZXIpIHtcblxuXHRcdHZhciBpdGVyYXRpb24gPSBuZXcgRmxvYXQzMkFycmF5KGFycmF5YnVmZmVyLCAwLCAxKSxcblx0XHRcdHZpZXcsXG5cdFx0XHR3ZWlnaHRzLFxuXHRcdFx0ZGF0YSxcblx0XHRcdGxlbixcblx0XHRcdGksXG5cdFx0XHRiYXRjaDtcblxuXHRcdHRpbWVzLnJlY2lldmVkID0gd2luZG93LnBlcmZvcm1hbmNlLm5vdygpO1xuXG5cdFx0dmlldyA9IG5ldyBGbG9hdDMyQXJyYXkoYXJyYXlidWZmZXIsIDQpO1xuXG5cblx0XHRpZiAoaXRlcmF0aW9uWzBdID49IDApIHsgLy8gaW5jbHVkZXMgbmV3IHdlaWdodHMgYW5kIGRhdGFcblx0XHRcdGl0ZXJhdGlvbnMgPSBpdGVyYXRpb25bMF07XG5cdFx0XHRpID0gbW9kZWwuc2l6ZTtcblx0XHRcdHdlaWdodHMgPSB2aWV3LnN1YmFycmF5KDAsIGkpO1xuXHRcdFx0bGVuID0gdmlld1tpXSAqIG5ldC5sYXllcnNbMF0uc2hhcGVbMV07IC8vIGZpcnN0IGZsb2F0IGlzIG51bWJlciBvZiBzYW1wbGVzIGluIHRoaXMgYmF0Y2hcblx0XHRcdGxlbiArPSBpICsgMTtcblx0XHRcdGJhdGNoID0ge1xuXHRcdFx0XHR4OiB2aWV3LnN1YmFycmF5KGksIGxlbiksXG5cdFx0XHRcdHk6IHZpZXcuc3ViYXJyYXkobGVuKVxuXHRcdFx0fTtcblxuXHRcdFx0bW9kZWwubG9hZCh3ZWlnaHRzLmJ1ZmZlcik7XG5cblx0XHR9IGVsc2UgeyAvLyB3ZWlnaHRzIGFyZSBmcmVzaCwgc28gZGF0YSBvbmx5XG5cdFx0XHRpdGVyYXRpb25zKys7XG5cdFx0XHRsZW4gPSB2aWV3WzBdICogbmV0LmxheWVyc1swXS5zaGFwZVsxXTsgLy8gZmlyc3QgZmxvYXQgaXMgbnVtYmVyIG9mIHNhbXBsZXMgaW4gdGhpcyBiYXRjaFxuXHRcdFx0YmF0Y2ggPSB7XG5cdFx0XHRcdHg6IHZpZXcuc3ViYXJyYXkoMSwgKytsZW4pLFxuXHRcdFx0XHR5OiB2aWV3LnN1YmFycmF5KGxlbilcblx0XHRcdH07XG5cdFx0fVxuXG5cdFx0Ly8gVFJBSU5cblx0XHR0aW1lcy5sb2FkZWQgPSB3aW5kb3cucGVyZm9ybWFuY2Uubm93KCk7XG5cdFx0bW9kZWwudHJhaW4obmV0LmxlYXJuaW5nX3JhdGUsIG5ldC5pdGVyYXRpb25zLCBiYXRjaC54LCBiYXRjaC55LCBmdW5jdGlvbihtb2RlbCkge1xuXHRcdFx0dmFyIHIgPSAwLCBsb2cgPSBcIlwiO1xuXHRcdFx0dGltZXMudHJhaW5lZCA9IHdpbmRvdy5wZXJmb3JtYW5jZS5ub3coKTtcblx0XHRcdC8vY29uc29sZS5sb2coXCJUaW1lIHRvIHRyYWluOiBcIiArIChkZWx0YSAvIDEwMDApICsgXCIgc2Vjb25kc1wiKTtcblx0XHRcdC8vIHBvc3QgcmVzdWx0cyB0byBzZXJ2ZXJcblx0XHRcdFxuXHRcdFx0UFVUKFwiLi93ZWlnaHRzL1wiICsgbmV0LmlkLCBcImFycmF5YnVmZmVyXCIsIG1vZGVsLnNhdmUoKSwgdXBkYXRlKTtcblx0XHRcdHIgPSB0aW1lcy51cGRhdGVkID0gd2luZG93LnBlcmZvcm1hbmNlLm5vdygpO1xuXG5cdFx0XHRsb2cgKz0gbmV0LndlaWdodHNfdmVyc2lvbiArIFwiLFwiO1xuXHRcdFx0bG9nICs9IG1vZGVsLmxvc3MgKyBcIixcIjtcblx0XHRcdGxvZyArPSB0aW1lcy5yZXF1ZXN0ZWQgKyBcIixcIjtcblx0XHRcdGxvZyArPSB0aW1lcy5yZWNpZXZlZCArIFwiLFwiO1xuXHRcdFx0bG9nICs9IHRpbWVzLmxvYWRlZCArIFwiLFwiO1xuXHRcdFx0bG9nICs9IHRpbWVzLnRyYWluZWQgKyBcIixcIjtcblx0XHRcdGxvZyArPSB0aW1lcy51cGRhdGVkICsgXCJcXG5cIjtcblx0XHRcdC8vIHNlbmQgdGltZSBhbmQgdHJhaW5pbmcgbG9nIHRvIHNlcnZlclxuXHRcdFx0UFVUKFwiLi9sb2cvXCIgKyBuZXQuaWQsIFwidGV4dFwiLCBsb2cpO1xuXHRcdFx0dGltZXMucmVxdWVzdGVkID0gcjtcblx0XHRcdG5ldC53ZWlnaHRzX3ZlcnNpb24rKztcblx0XHR9KTtcblx0fVxuXG5cdC8vdmFyIHNlcnZlciA9IGlvKCk7XG5cblx0Ly8gcmVxdWVzdCBtb2RlbCB0byB0cmFpblxuXHRHRVQoXCIuL21vZGVsXCIsIFwiYXBwbGljYXRpb24vanNvblwiLCBmdW5jdGlvbihqc29uTW9kZWwpIHtcblx0XHRuZXQgPSBKU09OLnBhcnNlKGpzb25Nb2RlbCk7XG5cblx0XHRtb2RlbCA9IG5ldyBNb2RlbChuZXQsIG51bGwpO1xuXHRcdHdpbmRvdy5vbmJlZm9yZXVubG9hZCA9IGZ1bmN0aW9uKCkge1xuXHRcdFx0UE9TVChcIi4vY2xvc2UvXCIgKyBuZXQuaWQsIFwic3RyaW5nXCIpXG5cdFx0fTtcblx0XHR0aW1lcy5yZXF1ZXN0ZWQgPSB3aW5kb3cucGVyZm9ybWFuY2Uubm93KCk7XG5cdFx0R0VUKFwiLi93ZWlnaHRzL1wiICsgbmV0LmlkLCBcImFycmF5YnVmZmVyXCIsIHVwZGF0ZSk7XG5cdH0pO1xufSkoKTsiLCJ2YXIgT3V0cHV0ID0gcmVxdWlyZSgnLi9sYXllcnMvT3V0cHV0JyksXG5cdERlbnNlID0gcmVxdWlyZSgnLi9sYXllcnMvRGVuc2UnKTtcblxubW9kdWxlLmV4cG9ydHMgPSBmdW5jdGlvbih0ZW5zb3JmaXJlLCBnbENvbnRleHQpIHtcblx0cmV0dXJuIHtcblx0XHRcImRlbnNlXCI6IERlbnNlKHRlbnNvcmZpcmUsIGdsQ29udGV4dCksXG5cdFx0XCJvdXRwdXRcIjogT3V0cHV0KHRlbnNvcmZpcmUsIGdsQ29udGV4dClcblx0fTtcbn07IiwidmFyIExheWVycyA9IHJlcXVpcmUoXCIuL0xheWVyc1wiKTtcblxudmFyIE1vZGVsID0gZnVuY3Rpb24obW9kZWwsIGxheWVycykge1xuXHR0aGlzLmxheWVycyA9IG5ldyBBcnJheShtb2RlbC5sYXllcnMubGVuZ3RoKTtcblx0dGhpcy5sb3NzID0gMC4wO1xuXHR0aGlzLnNpemUgPSAwLjA7XG5cdHRoaXMubW9kZWwgPSBtb2RlbDtcblx0dGhpcy5sb2FkKGxheWVycyk7XG5cblx0Ly9jb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeSh0aGlzLmxheWVyc1swXS5zYXZlKCkpKTtcbn07XG5Nb2RlbC5wcm90b3R5cGUucnVuID0gZnVuY3Rpb24oaW5wdXQpIHtcblx0dmFyIG91dHB1dCA9IGlucHV0LFxuXHRcdGwgPSAtMTtcblx0d2hpbGUgKCsrbCA8IHRoaXMubGF5ZXJzLmxlbmd0aClcblx0XHRvdXRwdXQgPSB0aGlzLmxheWVyc1tsXS5ydW4ob3V0cHV0KTtcbn07XG5Nb2RlbC5wcm90b3R5cGUuZm9yd2FyZCA9IGZ1bmN0aW9uKG91dHB1dCkge1xuXHQvL2NvbnNvbGUud2FybihcIkNhbGN1bG9uLSBGb3J3YXJkIHBhc3NcXG5cIik7XG5cdC8vIGZvcndhcmQgcHJvcG9nYXRpb25cblx0dmFyIGwgPSAtMTtcblx0d2hpbGUgKCsrbCA8IHRoaXMubGF5ZXJzLmxlbmd0aCkge1xuXHRcdG91dHB1dCA9IHRoaXMubGF5ZXJzW2xdLnJ1bihvdXRwdXQpO1xuXHRcdC8vY29uc29sZS5sb2coXCJDYWxjdWxvbi0gb3V0cHV0IFwiICsgbCArIFwiOiBcIiArIG91dHB1dC5yZWFkKCkuZGF0YSk7XG5cdH1cblx0cmV0dXJuIG91dHB1dDtcbn07XG5Nb2RlbC5wcm90b3R5cGUuYmFja3dhcmQgPSBmdW5jdGlvbihvdXRwdXQsIGxlYXJuKSB7XG5cdC8vY29uc29sZS53YXJuKFwiQ2FsY3Vsb24tIEJhY2t3YXJkIHBhc3NcIik7XG5cdC8vIGJhY2t3YXJkIHByb3BvZ2F0aW9uXG5cdHZhciBsID0gdGhpcy5sYXllcnMubGVuZ3RoIC0gMTtcblx0d2hpbGUgKGwtLSA+IDApIHtcblx0XHRvdXRwdXQgPSB0aGlzLmxheWVyc1tsXS50cmFpbihvdXRwdXQsIGxlYXJuKTtcblx0fVxufTtcblxuTW9kZWwucHJvdG90eXBlLnZhbGlkYXRlID0gZnVuY3Rpb24oaW5wdXQsIGV4cGVjdCwgY2FsbGJhY2spIHtcblx0dmFyIG91dHB1dCA9IGlucHV0LFxuXHRcdGxvc3NMYXllciA9IHRoaXMubGF5ZXJzW3RoaXMubGF5ZXJzLmxlbmd0aCAtIDFdO1xuXHRvdXRwdXQgPSB0aGlzLmZvcndhcmQob3V0cHV0KTtcblxuXHQvLyBjYWxjdWxhdGUgbG9zc1xuXHRvdXRwdXQgPSBsb3NzTGF5ZXIudHJhaW4oZXhwZWN0KTtcblx0aWYgKHR5cGVvZiBjYWxsYmFjayA9PT0gXCJmdW5jdGlvblwiKSBjYWxsYmFjayhsb3NzTGF5ZXIuYWNjdXJhY3kpXG5cbn1cblxuTW9kZWwucHJvdG90eXBlLnRyYWluID0gZnVuY3Rpb24obGVhcm4sIGl0ZXJhdGlvbnMsIGlucHV0LCBleHBlY3QsIGNhbGxiYWNrKSB7XG5cdHZhciBvdXRwdXQsXG5cdFx0ZSA9IDAsXG5cdFx0bG9zc0xheWVyID0gdGhpcy5sYXllcnNbdGhpcy5sYXllcnMubGVuZ3RoIC0gMV07XG5cdHdoaWxlIChlKysgPCBpdGVyYXRpb25zKSB7XG5cdFx0b3V0cHV0ID0gaW5wdXQ7XG5cdFx0b3V0cHV0ID0gdGhpcy5mb3J3YXJkKG91dHB1dCk7XG5cblx0XHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIG91dHB1dDogXCIgKyBvdXRwdXQucmVhZCgpLmRhdGEpO1xuXHRcdC8vIGNhbGN1bGF0ZSBsb3NzXG5cdFx0b3V0cHV0ID0gbG9zc0xheWVyLnRyYWluKGV4cGVjdCk7XG5cdFx0dGhpcy5sb3NzID0gbG9zc0xheWVyLmFjY3VyYWN5O1xuXHRcdGNvbnNvbGUubG9nKFwiQWNjdXJhY3k6IFwiICsgbG9zc0xheWVyLmFjY3VyYWN5KTtcblxuXHRcdHRoaXMuYmFja3dhcmQob3V0cHV0LCBsZWFybik7XG5cblx0XHQvLyBjaGFuY2UgdG8gc2VuZCBvdXQgZGF0YSBmcm9tIG1vZGVsIChtZXRhZGF0YSBhbmQgbG9nIGRhdGEpXG5cdFx0aWYgKHR5cGVvZiB0aGlzLmFmdGVySXRlcmF0aW9uID09PSBcImZ1bmN0aW9uXCIpIHRoaXMuYWZ0ZXJJdGVyYXRpb24odGhpcywgZSk7XG5cblx0XHQvL2NvbnNvbGUud2FybihcIkNhbGN1bG9uLSBJdGVyYXRpb246IFwiICsgZSArIFwiLCBMb3NzOiBcIiArIHRoaXMubG9zcyk7XG5cdH1cblx0aWYgKHR5cGVvZiBjYWxsYmFjayA9PT0gXCJmdW5jdGlvblwiKSBjYWxsYmFjayh0aGlzKTtcbn1cbk1vZGVsLnByb3RvdHlwZS5zYXZlID0gZnVuY3Rpb24oKSB7XG5cdC8vIFR5cGVkQXJyYXkgdG8gaG9sZCB3ZWlnaHRzLCBiaWFzLCBldGMuIGZyb20gZXZlcnkgbGF5ZXIgb2YgbW9kZWxcblx0dmFyIHdlaWdodHMgPSBuZXcgRmxvYXQzMkFycmF5KHRoaXMuc2l6ZSk7XG5cdFxuXHR2YXIgbCA9IC0xLFxuXHRcdG8gPSAwO1xuXHQvLyBwdWxsIG91dCB0cmFpbmVkIHdlaWdodHMgZm9yIGVhY2ggbGF5ZXJcblx0d2hpbGUgKCsrbCA8ICh0aGlzLmxheWVycy5sZW5ndGggLSAxKSkge1xuXHRcdHdlaWdodHMuc2V0KCB0aGlzLmxheWVyc1tsXS5zYXZlKCksIG8pO1xuXHRcdG8gKz0gdGhpcy5sYXllcnNbbF0uc2l6ZTtcblx0fVxuXHQvL2NvbnNvbGUubG9nKFwid2VpZ2h0czogXCIgKyB3ZWlnaHRzKTtcblx0cmV0dXJuIHdlaWdodHMuYnVmZmVyO1xufTtcbk1vZGVsLnByb3RvdHlwZS5sb2FkID0gZnVuY3Rpb24obGF5ZXJzKSB7XG5cdC8vIGNvbnN0cnVjdCBsYXllcnNcblx0dmFyIG9mZnNldCA9IDAsXG5cdFx0bGF5ZXIsXG5cdFx0bCA9IC0xO1xuXG5cblx0dGhpcy5zaXplID0gMDtcblx0aWYgKGxheWVycyAhPSBudWxsKSB7XG5cdFx0bGF5ZXJzID0gbmV3IEZsb2F0MzJBcnJheShsYXllcnMpO1xuXHR9XG5cdHdoaWxlICgrK2wgPCAodGhpcy5sYXllcnMubGVuZ3RoIC0gMSkpIHtcblx0XHRsYXllciA9IHRoaXMubW9kZWwubGF5ZXJzW2xdO1xuXHRcdGxheWVyID0gbmV3IExheWVyc1tsYXllci50eXBlXShsYXllciwgbCk7XG5cdFx0dGhpcy5zaXplICs9IGxheWVyLnNpemU7XG5cdFx0aWYgKGxheWVycyAhPSBudWxsKVxuXHRcdFx0b2Zmc2V0ID0gbGF5ZXIubG9hZChsYXllcnMsIG9mZnNldCk7XG5cdFx0ZWxzZSBsYXllci5yYW5kb21XZWlnaHRzKCk7XG5cdFx0dGhpcy5sYXllcnNbbF0gPSBsYXllcjtcblx0fVxuXHQvLyBpbml0aWFsaXplIG91dHB1dCBsYXllclxuXHRsYXllciA9IHRoaXMubW9kZWwubGF5ZXJzW2xdO1xuXHRsYXllciA9IG5ldyBMYXllcnNbbGF5ZXIudHlwZV0obGF5ZXIsIGwpO1xuXHR0aGlzLmxheWVyc1tsXSA9IGxheWVyO1xuXG59O1xuXG5tb2R1bGUuZXhwb3J0cyA9IGZ1bmN0aW9uKHRlbnNvcmZpcmUsIGdsQ29udGV4dCkge1xuXHRMYXllcnMgPSBMYXllcnModGVuc29yZmlyZSwgZ2xDb250ZXh0KTtcblx0cmV0dXJuIE1vZGVsO1xufTsiLCJtb2R1bGUuZXhwb3J0cyA9IHtcblx0QWN0aXZhdGlvbjoge1xuXHRcdFwibGluZWFyXCI6IFwibyA9IG47IFxcblwiLFxuXHRcdC8vXCJiaW5hcnlcIjogXCJpZiAobiA+IDAuMCkgeyBvID0gMC4wOyB9IGVsc2UgeyBvID0gMS4wOyB9IFxcblwiLFxuXHRcdFwicmVsdVwiOiBcIm8gPSBtYXgoMC4wLCBuKTsgXFxuXCIsXG5cdFx0XCJscmVsdVwiOiBcImlmIChuID4gMC4wKSB7IG8gPSBuOyB9IGVsc2UgeyBvID0gMC4wMSAqIG47IH0gXFxuXCIsXG5cdFx0XCJzaWdtb2lkXCI6IFwibyA9IDEuMCAvICgxLjAgKyBleHAoMC4wIC0gbikpOyBcXG5cIixcblx0XHRcInRhbmhcIjogXCJvID0gKDIuMCAvICgxLjAgKyBleHAoLTIuMCAqIG4pKSkgLSAxLjA7IFxcblwiLFxuXHRcdFwic29mdHBsdXNcIjogXCJvID0gbG9nKDEuMCArIGV4cChuKSk7IFxcblwiLFxuXHRcdFwic29mdG1heFwiOiBcImZsb2F0IGsgPSAwLjA7IFxcbmZvcihpbnQgaSA9IDA7IGkgPCAjKE8uc2hhcGUpLng7IGkrKyl7XFxuayArPSBleHAoTy5yZWFkKGksIHBvcy55KSk7XFxufVxcbm8gPSBleHAobikgLyBrOyBcXG5cIixcblx0XHQvL1wic29mdHNpZ25cIjogXCJvID0gbiAvICgxLjAgKyBhYnMobikpOyBcXG5cIlxuXHR9LFxuXHREZXJpdmF0aXZlOiB7XG5cdFx0XCJsaW5lYXJcIjogXCJkID0gMS4wOyBcXG5cIixcblx0XHQvL1wiYmluYXJ5XCI6IFwiaWYgKG8gPT0gMC4wKSB7IGQgPSAwLjA7IH0gZWxzZSB7IGQgPSAwLjA7IH0gXFxuXCIsXG5cdFx0XCJyZWx1XCI6IFwiaWYgKG8gPj0gMC4wKSB7IGQgPSAxLjA7IH0gZWxzZSB7IGQgPSAwLjA7IH0gXFxuXCIsXG5cdFx0XCJscmVsdVwiOiBcImlmIChvID49IDAuMCkgeyBkID0gMS4wOyB9IGVsc2UgeyBkID0gMC4wMTsgfSBcXG5cIixcblx0XHRcInNpZ21vaWRcIjogXCJkID0gbyAqICggMS4wIC0gbyApOyBcXG5cIixcblx0XHRcInRhbmhcIjogXCJkID0gKCAxLjAgLSBwb3cobywgMi4wKSApOyBcXG5cIixcblx0XHRcInNvZnRwbHVzXCI6IFwiZCA9IDEuMCAtICggMS4wIC8gZXhwKG8pICk7IFxcblwiLFxuXHRcdFwic29mdG1heFwiOiBcImQgPSBvICogKCAxLjAgLSBvICk7IFxcblwiIC8vIHNhbWUgYXMgc2lnbW9pZD9cblx0XHQvL1wic29mdHNpZ25cIjogXCJ2YXIgPSBcIlxuXHR9XG59OyIsInZhciBuZGFycmF5ID0gcmVxdWlyZShcIm5kYXJyYXlcIiksXG5cdFRGLFxuXHRHTCxcblxuXHRHcmFkIFx0PSBcInVuaWZvcm0gVGVuc29yIE87IFxcblwiXG5cdFx0XHQrIFwidW5pZm9ybSBUZW5zb3IgRTsgXFxuXCJcblx0XHRcdCsgXCJmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykgeyBcXG5cIlxuXHRcdFx0XHQrIFwicmV0dXJuIDAuMCAtIEUucmVhZChwb3MpIC8gTy5yZWFkKHBvcyk7IFxcblwiXG5cdFx0XHQrIFwifSBcXG5cIlxuXHRcdFx0LFxuXG5cdExvc3MgXHQ9IFwidW5pZm9ybSBUZW5zb3IgTzsgXFxuXCJcblx0XHRcdCsgXCJ1bmlmb3JtIFRlbnNvciBFOyBcXG5cIlxuXHRcdFx0KyBcImZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7IFxcblwiXG5cdFx0XHRcdCsgXCJmbG9hdCBsb3NzID0gMC4wOyBcXG5cIlxuXHRcdFx0XHQrIFwiZm9yKGludCBpID0gMDsgaSA8ICMoTy5zaGFwZSkueTsgaSsrKXsgXFxuXCIgLyogaXRlcmF0ZSBvdmVyIGVhY2ggc2FtcGxlICovXG5cdFx0XHRcdFx0KyBcImZsb2F0IGwgPSAwLjA7IFxcblwiXG5cdFx0XHRcdFx0KyBcImZvcihpbnQgaiA9IDA7IGogPCAjKE8uc2hhcGUpLng7IGorKyl7IFxcblwiIC8qIGl0ZXJhdGUgb3ZlciBldmVyeSBvdXRwdXQgYW5kIGNhbGN1bGF0ZSBhdmVyYWdlICovXG5cdFx0XHRcdFx0XHQrIFwibCAtPSBFLnJlYWQoaiwgaSkgKiBsb2coTy5yZWFkKGosIGkpKTsgXFxuXCJcblx0XHRcdFx0XHQrIFwifSBcXG5cIlxuXHRcdFx0XHRcdCsgXCJsb3NzID0gbCAvIGZsb2F0KCMoTy5zaGFwZSkueSk7IFxcblwiXG5cdFx0XHRcdCsgXCJ9IFxcblwiXG5cdFx0XHRcdCsgXCJyZXR1cm4gbG9zczsgXFxuXCJcblx0XHRcdCsgXCJ9IFxcblwiXG5cdFx0XHQ7XG5cbmZ1bmN0aW9uIENyb3NzRW50cm9weSgpIHtcblx0Ly8gY2FsY3VsYXRlIGxvc3MgZ3JhZGllbnRzXG5cdHRoaXMuZ3JhZCA9IEdyYWQ7XG5cblx0Ly8gY2FsY3VsYXRlIGJhdGNoIGF2ZXJhZ2UgbG9zc1xuXHR0aGlzLmxvc3NGID0gTG9zcztcblxuXHR0aGlzLmxvc3MgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBbMV0pO1xuXHR0aGlzLm91dHB1dCA9IG51bGw7XG59XG5Dcm9zc0VudHJvcHkucHJvdG90eXBlLmRlbHRhcyA9IGZ1bmN0aW9uKG91dHB1dCwgZXhwZWN0KSB7XG5cdGlmIChleHBlY3QgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkpXG5cdFx0ZXhwZWN0ID0gbmV3IFRGLlRlbnNvcihHTCwgbmRhcnJheSggZXhwZWN0LCBvdXRwdXQuc2hhcGUpKTtcblxuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIGV4cGVjdGVkOiBcIiArIGV4cGVjdC5yZWFkKCkuZGF0YSk7XG5cblx0dGhpcy5vdXRwdXQgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBvdXRwdXQuc2hhcGUpO1xuXHR0aGlzLm91dHB1dC5ydW4odGhpcy5ncmFkLCB7IE86IG91dHB1dCwgRTogZXhwZWN0IH0pO1xuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIGdyYWRpZW50OiBcIiArIHRoaXMub3V0cHV0LnJlYWQoKS5kYXRhKTtcblxuXHR0aGlzLmxvc3MucnVuKHRoaXMubG9zc0YsIHsgTzogb3V0cHV0LCBFOiBleHBlY3QgfSk7XG5cdC8vY29uc29sZS5sb2codGhpcy5sb3NzLnJlYWQoKSk7XG5cblx0cmV0dXJuIHRoaXMub3V0cHV0O1xufVxuXG5tb2R1bGUuZXhwb3J0cyA9IGZ1bmN0aW9uKHRlbnNvcmZpcmUsIGdsQ29udGV4dCkge1xuXHRURiA9IHRlbnNvcmZpcmU7XG5cdEdMID0gZ2xDb250ZXh0O1xuXHRyZXR1cm4gQ3Jvc3NFbnRyb3B5O1xufTsiLCJ2YXIgbmRhcnJheSA9IHJlcXVpcmUoXCJuZGFycmF5XCIpLFxuXHRURixcblx0R0wsXG5cblx0RnVuY3MgPSByZXF1aXJlKCcuL0FjdGl2YXRpb25zJyksXG5cdGdlbmVyYXRlV2VpZ2h0cyA9IHJlcXVpcmUoJy4uL3V0aWwvZ2VuZXJhdGVXZWlnaHRzJyksXG5cblx0Rm9yd2FyZCA9IFwidW5pZm9ybSBUZW5zb3IgVzsgXFxuXCIgLyogd2VpZ2h0cyAqL1xuXHRcdFx0KyBcInVuaWZvcm0gVGVuc29yIEk7IFxcblwiIC8qIGlucHV0ICovXG5cdFx0XHQrIFwiZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHsgXFxuXCJcblx0XHRcdFx0KyBcImZsb2F0IG4gPSAwLjA7IFxcblwiXG5cdFx0XHRcdCsgXCJmb3IoaW50IGkgPSAwOyBpIDwgIyhXLnNoYXBlKS55OyBpKyspeyBcXG5cIlxuXHRcdFx0XHRcdCsgXCJpZiAoaSA9PSAjKFcuc2hhcGUpLnkgLSAxKSB7IG4gKz0gVy5yZWFkKHBvcy54LCBpKTsgfSBcXG5cIlxuXHRcdFx0XHRcdCsgXCJlbHNlIHsgbiArPSBJLnJlYWQoaSwgcG9zLnkpICogVy5yZWFkKHBvcy54LCBpKTsgfSBcXG5cIlxuXHRcdFx0XHQrIFwifSBcXG5cIlxuXHRcdFx0XHQrIFwicmV0dXJuIG47XFxuXCJcblx0XHRcdCsgXCJ9IFxcblwiXG5cdFx0XHQsXG5cdEJhY2t3YXJkPSBcInVuaWZvcm0gVGVuc29yIEU7IFxcblwiIC8qIGxvY2FsIGVycm9yIChmcm9tIGFjdGl2YXRpb24pICovXG5cdFx0XHQrIFwidW5pZm9ybSBUZW5zb3IgVzsgXFxuXCIgLyogd2VpZ2h0cyAqL1xuXHRcdFx0KyBcImZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7IFxcblwiIC8vIHBvc2l0aW9uIGluIHBhcnRpYWwgVGVuc29yXG5cdFx0XHRcdCsgXCJmbG9hdCBlID0gMC4wOyBcXG5cIiAvKiBzdW0gb3V0cHV0IGVycm9yICovXG5cdFx0XHRcdCsgXCJmb3IoaW50IGkgPSAwOyBpIDwgIyhFLnNoYXBlKS54IDsgaSsrKXsgXFxuXCJcblx0XHRcdFx0XHQrIFwiZSArPSBXLnJlYWQocG9zLngsIGkpICogRS5yZWFkKGksIHBvcy55KTsgXFxuXCJcblx0XHRcdFx0KyBcIn0gXFxuXCJcblx0XHRcdFx0KyBcInJldHVybiBlOyBcXG5cIlxuXHRcdFx0KyBcIn0gXFxuXCJcblx0XHRcdCxcblx0V2VpZ2h0cyA9IFwidW5pZm9ybSBUZW5zb3IgRTsgXFxuXCIgLyogbG9jYWwgZXJyb3IgKGZyb20gYWN0aXZhdGlvbikgKi9cblx0XHRcdCsgXCJ1bmlmb3JtIFRlbnNvciBXOyBcXG5cIiAvKiB3ZWlnaHRzICovXG5cdFx0XHQrIFwidW5pZm9ybSBUZW5zb3IgSTsgXFxuXCIgLyogaW5wdXQgKi9cblx0XHRcdCsgXCJ1bmlmb3JtIGZsb2F0IGw7IFxcblwiIC8qIGxlYXJuaW5nIHJhdGUgKi9cblx0XHRcdCsgXCJmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykgeyBcXG5cIiAvLyBwb3MgaW4gd2VpZ2h0cyBUZW5zb3Jcblx0XHRcdFx0KyBcImZsb2F0IGUgPSAwLjA7IFxcblwiIC8qIGF2ZyBub2RlIGJhdGNoIGVycm9yICovXG5cdFx0XHRcdCsgXCJmb3IoaW50IGkgPSAwOyBpIDwgIyhFLnNoYXBlKS55OyBpKyspeyBcXG5cIlxuXHRcdFx0XHRcdCsgXCJpZiAocG9zLnkgPT0gIyhJLnNoYXBlKS54KSB7IFxcblwiIC8qIGhhbmRsZSBiaWFzIGxheWVyID8gKi9cblx0XHRcdFx0XHRcdCsgXCJlICs9IEUucmVhZChwb3MueCwgaSkgLyBmbG9hdCgjKEUuc2hhcGUpLnkpOyBcXG5cIlxuXHRcdFx0XHRcdCsgXCJ9IGVsc2UgeyBcXG5cIlxuXHRcdFx0XHRcdFx0KyBcImUgKz0gRS5yZWFkKHBvcy54LCBpKSAqIEkucmVhZChwb3MueSwgaSkgLyBmbG9hdCgjKEUuc2hhcGUpLnkpOyBcXG5cIlxuXHRcdFx0XHRcdCsgXCJ9IFxcblwiXG5cdFx0XHRcdCsgXCJ9IFxcblwiXG5cdFx0XHRcdCsgXCJyZXR1cm4gVy5yZWFkKHBvcykgLSAobCAqIGUpOyBcXG5cIlxuXHRcdFx0KyBcIn0gXFxuXCJcblx0XHRcdCxcblx0QWN0QSBcdD0gXCJ1bmlmb3JtIFRlbnNvciBPOyBcXG5cIiAvKiB3ZWlnaHRlZCBvdXRwdXQgKi9cblx0XHRcdCsgXCJmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykgeyBcXG5cIlxuXHRcdFx0XHQrIFwiZmxvYXQgbiA9IE8ucmVhZChwb3MpOyBcXG5cIlxuXHRcdFx0XHQrIFwiZmxvYXQgbzsgXFxuXCJcblx0XHRcdCxcblx0QWN0QiBcdD0gXHQgIFwicmV0dXJuIG87IFxcblwiXG5cdFx0XHQrIFwifSBcXG5cIlxuXHRcdFx0LFxuXHRHcmFkQVx0PSBcInVuaWZvcm0gVGVuc29yIEU7IFxcblwiXG5cdFx0XHQrIFwidW5pZm9ybSBUZW5zb3IgTzsgXFxuXCJcblx0XHRcdCsgXCJ1bmlmb3JtIFRlbnNvciBIOyBcXG5cIlxuXHRcdFx0KyBcImZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7IFxcblwiXG5cdFx0XHRcdCsgXCJmbG9hdCBkOyBcXG5cIlxuXHRcdFx0XHQrIFwiZmxvYXQgbyA9IE8ucmVhZChwb3MpOyBcXG5cIlxuXHRcdFx0LFxuXHRHcmFkQiBcdD0gXHQgIFwiZCAqPSBFLnJlYWQocG9zKTsgXFxuXCJcblx0XHRcdCsgXHQgIFwicmV0dXJuIGQ7IFxcblwiXG5cdFx0XHQrIFwifSBcXG5cIlxuXHRcdFx0O1xuXG5cblxuZnVuY3Rpb24gRGVuc2UobGF5ZXIsIGluZGV4KSB7XG5cdHRoaXMubCA9IGluZGV4O1xuXHQvLyBwcm9kdWNlIE91dHB1dCBUZW5zb3IgZ2l2ZW4gaW5wdXQsIHdlaWdodHMsIGFuZCBiaWFzIFRlbnNvcnNcblx0dGhpcy5mb3J3YXJkID0gRm9yd2FyZDtcblxuXHR0aGlzLmFjdGl2YXRpb24gPSBBY3RBICsgRnVuY3MuQWN0aXZhdGlvbltsYXllci5hY3RpdmF0aW9uXSArIEFjdEI7XG5cdFx0XHRcdFx0XHRcblx0Ly8gcHJvZHVjZSB1cHN0cmVhbSBlcnJvciBUZW5zb3IgZ2l2ZW4gZG93bnN0cmVhbSBlcnJvciwgaW5wdXQsIHdlaWdodHMsIGJpYXNcblx0dGhpcy5iYWNrd2FyZCBcdD0gQmFja3dhcmQ7XG5cdHRoaXMuZ3JhZGllbnQgXHQ9IEdyYWRBICsgRnVuY3MuRGVyaXZhdGl2ZVtsYXllci5hY3RpdmF0aW9uXSArIEdyYWRCO1xuXHQvLyBhZGp1c3Qgd2VpZ2h0cyBUZW5zb3IgZ2l2ZW4gZXJyb3IgYW5kIGlucHV0IFRlbnNvclxuXHR0aGlzLnVwZGF0ZSA9IFdlaWdodHM7XG5cblx0dGhpcy5zaGFwZSA9IGxheWVyLnNoYXBlO1xuXHR0aGlzLmlucHV0ID0gbnVsbDtcblx0dGhpcy5vdXRwdXQgPSBudWxsO1xuXHR0aGlzLndlaWdodGVkT3V0cHV0ID0gbnVsbDtcblx0dGhpcy53ZWlnaHRzID0gbnVsbDtcblx0dGhpcy5iaWFzID0gbGF5ZXIuYmlhcztcblx0dGhpcy5zaXplID0gdGhpcy5zaGFwZVswXSAqIHRoaXMuc2hhcGVbMV0gKyAodGhpcy5iaWFzID8gdGhpcy5zaGFwZVswXSA6IDApO1xuXG59XG5EZW5zZS5wcm90b3R5cGUubG9hZCA9IGZ1bmN0aW9uKGFycmF5LCBvZmZzZXQpIHtcblx0dmFyIGxlbmd0aCA9IHRoaXMuc2l6ZTtcblx0Ly8gcmVhZCBpbiB3ZWlnaHRzIChhbmQgYmlhcylcblx0dGhpcy53ZWlnaHRzID0gbmV3IFRGLkluUGxhY2VUZW5zb3IoR0wsIG5kYXJyYXkoIGFycmF5LnN1YmFycmF5KG9mZnNldCwgb2Zmc2V0ICsgbGVuZ3RoKSwgW3RoaXMuc2hhcGVbMF0sIHRoaXMuc2hhcGVbMV0gKyAodGhpcy5iaWFzID8gMSA6IDApXSApICk7XG5cdG9mZnNldCArPSBsZW5ndGg7XG5cdHJldHVybiBvZmZzZXQ7XG59XG5EZW5zZS5wcm90b3R5cGUucmFuZG9tV2VpZ2h0cyA9IGZ1bmN0aW9uKCkge1xuXHR0aGlzLndlaWdodHMgPSBuZXcgVEYuSW5QbGFjZVRlbnNvcihHTCwgbmRhcnJheSggZ2VuZXJhdGVXZWlnaHRzKHRoaXMuc2hhcGUsICh0aGlzLmJpYXMgPyB0aGlzLnNoYXBlWzBdIDogMCkpLCBbdGhpcy5zaGFwZVswXSwgdGhpcy5zaGFwZVsxXSArICh0aGlzLmJpYXMgPyAxIDogMCldICkgKTtcbn1cbkRlbnNlLnByb3RvdHlwZS5zYXZlID0gZnVuY3Rpb24oKSB7XG5cdHJldHVybiB0aGlzLndlaWdodHMucmVhZCgpLmRhdGE7XG59XG5EZW5zZS5wcm90b3R5cGUucnVuID0gZnVuY3Rpb24oaW5wdXQpIHtcblx0aWYgKGlucHV0IGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KSB7XG5cdFx0dGhpcy5pbnB1dCA9IG5ldyBURi5UZW5zb3IoR0wsIG5kYXJyYXkoIGlucHV0LCBbIHRoaXMuc2hhcGVbMV0sIChpbnB1dC5sZW5ndGggLyB0aGlzLnNoYXBlWzFdKSA+PiAwIF0pKTtcblx0fSBlbHNlIHRoaXMuaW5wdXQgPSBpbnB1dDtcblx0Ly9jb25zb2xlLmxvZyh0aGlzLmlucHV0LnNoYXBlKTtcblx0Ly9jb25zb2xlLmxvZyhcIkNhbGN1bG9uLSBpbnB1dCBcIiArIHRoaXMubCArIFwiOiBcIiArIHRoaXMuaW5wdXQucmVhZCgpLmRhdGEpO1xuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIHdlaWdodHMgXCIgKyB0aGlzLmwgKyBcIjogXCIgKyB0aGlzLndlaWdodHMucmVhZCgpLmRhdGEpO1xuXG5cdHRoaXMud2VpZ2h0ZWRPdXRwdXQgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBbIHRoaXMuc2hhcGVbMF0sIHRoaXMuaW5wdXQuc2hhcGVbMV0gXSk7XG5cdHRoaXMud2VpZ2h0ZWRPdXRwdXQucnVuKHRoaXMuZm9yd2FyZCwge1c6IHRoaXMud2VpZ2h0cywgSTogdGhpcy5pbnB1dH0pO1xuXG5cdC8vY29uc29sZS5sb2coXCJDYWxjdWxvbi0gd2VpZ2h0ZWRPdXRwdXQgXCIgKyB0aGlzLmwgKyBcIjogXCIgKyB0aGlzLndlaWdodGVkT3V0cHV0LnJlYWQoKS5kYXRhKTtcblxuXHR0aGlzLm91dHB1dCA9IG5ldyBURi5PdXRwdXRUZW5zb3IoR0wsIFsgdGhpcy5zaGFwZVswXSwgdGhpcy5pbnB1dC5zaGFwZVsxXSBdKTtcblx0dGhpcy5vdXRwdXQucnVuKHRoaXMuYWN0aXZhdGlvbiwge086IHRoaXMud2VpZ2h0ZWRPdXRwdXR9KTtcblxuXHQvL2NvbnNvbGUubG9nKFwib3V0cHV0IFwiICsgdGhpcy5sICsgXCI6IFwiICsgdGhpcy5vdXRwdXQucmVhZCgpLmRhdGEpO1xuXHRyZXR1cm4gdGhpcy5vdXRwdXQ7XG59O1xuRGVuc2UucHJvdG90eXBlLnRyYWluID0gZnVuY3Rpb24oZXJyb3IsIGxlYXJuaW5nX3JhdGUpIHtcblx0dmFyIHBhcnRpYWwgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCB0aGlzLmlucHV0LnNoYXBlKTtcblx0dmFyIGxvY2FsID0gbmV3IFRGLk91dHB1dFRlbnNvcihHTCwgdGhpcy5vdXRwdXQuc2hhcGUpO1xuXG5cdC8vY29uc29sZS5sb2coXCJDYWxjdWxvbi0gZXJyb3I6IFwiICsgZXJyb3IucmVhZCgpLmRhdGEpO1xuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIHdlaWdodHMgXCIgKyB0aGlzLmwgKyBcIjogXCIgKyB0aGlzLndlaWdodHMucmVhZCgpLmRhdGEpO1xuXG5cdGxvY2FsLnJ1bih0aGlzLmdyYWRpZW50LCB7RTogZXJyb3IsIE86IHRoaXMub3V0cHV0LCBIOiB0aGlzLndlaWdodGVkT3V0cHV0fSk7XG5cdC8vY29uc29sZS5sb2coXCJDYWxjdWxvbi0gbG9jYWxFOiBcIiArIGxvY2FsLnJlYWQoKS5kYXRhKTtcblxuXHQvLyB0cmFpbiB3ZWlnaHRzXG5cdHRoaXMud2VpZ2h0cy5ydW4odGhpcy51cGRhdGUsIHtXOiB0aGlzLndlaWdodHMsIEU6IGxvY2FsLCBJOiB0aGlzLmlucHV0LCBsOiBsZWFybmluZ19yYXRlfSk7XG5cblxuXG5cdC8vY29uc29sZS5sb2coXCJDYWxjdWxvbi0gdXBkYXRlZCBcIiArIHRoaXMubCArIFwiOiBcIiArIHRoaXMud2VpZ2h0cy5yZWFkKCkuZGF0YSk7XG5cblx0Ly8gY2FsY3VsYXRlIHVwc3RyZWFtIGVycm9yc1xuXHRwYXJ0aWFsLnJ1bih0aGlzLmJhY2t3YXJkLCB7RTogZXJyb3IsIEk6IHRoaXMuaW5wdXQsIFc6IHRoaXMud2VpZ2h0cywgTzogdGhpcy5vdXRwdXR9KTtcblxuXHRyZXR1cm4gcGFydGlhbDtcbn07XG5cbm1vZHVsZS5leHBvcnRzID0gZnVuY3Rpb24odGVuc29yZmlyZSwgZ2xDb250ZXh0KSB7XG5cdFRGID0gdGVuc29yZmlyZTtcblx0R0wgPSBnbENvbnRleHQ7XG5cdHJldHVybiBEZW5zZTtcbn07IiwidmFyIG5kYXJyYXkgPSByZXF1aXJlKFwibmRhcnJheVwiKSxcblx0VEYsXG5cdEdMLFxuXG5cdEdyYWQgXHQ9IFwidW5pZm9ybSBUZW5zb3IgTzsgXFxuXCJcblx0XHRcdCsgXCJ1bmlmb3JtIFRlbnNvciBFOyBcXG5cIlxuXHRcdFx0KyBcImZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7IFxcblwiXG5cdFx0XHRcdCsgXCJyZXR1cm4gTy5yZWFkKHBvcykgLSBFLnJlYWQocG9zKTsgXFxuXCJcblx0XHRcdCsgXCJ9IFxcblwiXG5cdFx0XHQsXG5cdExvc3MgXHQ9IFwidW5pZm9ybSBUZW5zb3IgRzsgXFxuXCJcblx0XHRcdCsgXCJmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykgeyBcXG5cIlxuXHRcdFx0XHQrIFwiZmxvYXQgbG9zcyA9IDAuMDsgXFxuXCJcblx0XHRcdFx0KyBcImZvcihpbnQgaSA9IDA7IGkgPCAjKEcuc2hhcGUpLnk7IGkrKyl7IFxcblwiIC8qIGl0ZXJhdGUgb3ZlciBlYWNoIHNhbXBsZSAqL1xuXHRcdFx0XHRcdCsgXCJmbG9hdCBsID0gMC4wOyBcXG5cIlxuXHRcdFx0XHRcdCsgXCJmb3IoaW50IGogPSAwOyBqIDwgIyhHLnNoYXBlKS54OyBqKyspeyBcXG5cIiAvKiBpdGVyYXRlIG92ZXIgZXZlcnkgb3V0cHV0IGFuZCBjYWxjdWxhdGUgYXZlcmFnZSAqL1xuXHRcdFx0XHRcdFx0KyBcImwgKz0gcG93KGZsb2F0KEcucmVhZChqLCBpKSksIDIuMCkgLyBmbG9hdCgjKEcuc2hhcGUpLngpOyBcXG5cIlxuXHRcdFx0XHRcdCsgXCJ9IFxcblwiXG5cdFx0XHRcdFx0KyBcImxvc3MgKz0gbCAvIGZsb2F0KCMoRy5zaGFwZSkueSk7IFxcblwiXG5cdFx0XHRcdCsgXCJ9IFxcblwiXG5cdFx0XHRcdCsgXCJyZXR1cm4gbG9zczsgXFxuXCJcblx0XHRcdCsgXCJ9IFxcblwiXG5cdFx0XHQ7XG5cbmZ1bmN0aW9uIE1TRSgpIHtcblx0Ly8gY2FsY3VsYXRlIGxvc3MgZ3JhZGllbnRzXG5cdHRoaXMuZ3JhZCA9IEdyYWQ7XG5cblx0Ly8gY2FsY3VsYXRlIGJhdGNoIGF2ZXJhZ2UgbG9zc1xuXHR0aGlzLmxvc3NGID0gTG9zcztcblxuXHR0aGlzLmxvc3MgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBbMV0pO1xuXHR0aGlzLm91dHB1dCA9IG51bGw7XG59XG5NU0UucHJvdG90eXBlLmRlbHRhcyA9IGZ1bmN0aW9uKG91dHB1dCwgZXhwZWN0KSB7XG5cdGlmIChleHBlY3QgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkpXG5cdFx0ZXhwZWN0ID0gbmV3IFRGLlRlbnNvcihHTCwgbmRhcnJheSggZXhwZWN0LCBvdXRwdXQuc2hhcGUpKTtcblxuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIGV4cGVjdGVkOiBcIiArIGV4cGVjdC5yZWFkKCkuZGF0YSk7XG5cblx0dGhpcy5vdXRwdXQgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBvdXRwdXQuc2hhcGUpO1xuXHR0aGlzLm91dHB1dC5ydW4odGhpcy5ncmFkLCB7IE86IG91dHB1dCwgRTogZXhwZWN0IH0pO1xuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIGdyYWRpZW50OiBcIiArIHRoaXMub3V0cHV0LnJlYWQoKS5kYXRhKTtcblxuXHR0aGlzLmxvc3MucnVuKHRoaXMubG9zc0YsIHsgRzogdGhpcy5vdXRwdXQgfSk7XG5cblx0cmV0dXJuIHRoaXMub3V0cHV0O1xufVxuXG5tb2R1bGUuZXhwb3J0cyA9IGZ1bmN0aW9uKHRlbnNvcmZpcmUsIGdsQ29udGV4dCkge1xuXHRURiA9IHRlbnNvcmZpcmU7XG5cdEdMID0gZ2xDb250ZXh0O1xuXHRyZXR1cm4gTVNFO1xufTsiLCJ2YXIgbmRhcnJheSA9IHJlcXVpcmUoXCJuZGFycmF5XCIpLFxuXHRURixcblx0R0wsXG5cdFNvZnRtYXggPSByZXF1aXJlKCcuL1NvZnRtYXgnKSxcblx0TVNFID0gcmVxdWlyZSgnLi9NU0UnKSxcblx0Q3Jvc3NFbnRyb3B5ID0gcmVxdWlyZSgnLi9Dcm9zc0VudHJvcHknKSxcblxuXHRBY2N1cmFjeT0gXCJ1bmlmb3JtIFRlbnNvciBPOyBcXG5cIlxuXHRcdFx0KyBcInVuaWZvcm0gVGVuc29yIEU7IFxcblwiXG5cdFx0XHQrIFwiZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHsgXFxuXCJcblx0XHRcdFx0KyBcImludCBsID0gIyhPLnNoYXBlKS54OyBcXG5cIlxuXHRcdFx0XHQrIFwiZmxvYXQgdiA9IDAuMDsgXFxuXCJcblx0XHRcdFx0KyBcImZvciAoaW50IGkgPSAwOyBpIDwgIyhPLnNoYXBlKS54OyBpKyspIHsgXFxuXCIgLyogaXRlcmF0ZSBvdmVyIGV2ZXJ5IG91dHB1dCBhbmQgY2FsY3VsYXRlIGF2ZXJhZ2UgKi9cblx0XHRcdFx0XHQrIFwiaWYgKE8ucmVhZChpLCBwb3MueCkgPj0gdikgeyBcXG5cIlxuXHRcdFx0XHRcdFx0KyBcInYgPSBPLnJlYWQoaSwgcG9zLnkpOyBcXG5cIiAvKiBnZXQgbGFyZ2VzdCBjYXRlZ29yeSBpbiBvdXRwdXQgKi9cblx0XHRcdFx0XHRcdCsgXCJsID0gaTsgXFxuXCIgLyogc2F2ZSBpbmRleCBvZiBsYXJnZXN0IHZhbHVlICovXG5cdFx0XHRcdFx0KyBcIn0gXFxuXCJcblx0XHRcdFx0KyBcIn0gXFxuXCJcblx0XHRcdFx0KyBcImlmIChFLnJlYWQobCwgcG9zLnkpID4gMC45KSB7IFxcblwiXG5cdFx0XHRcdFx0KyBcInJldHVybiAxLjA7IFxcblwiXG5cdFx0XHRcdCsgXCJ9IGVsc2UgeyBcXG5cIlxuXHRcdFx0XHRcdCsgXCJyZXR1cm4gMC4wOyBcXG5cIlxuXHRcdFx0XHQrIFwifSBcXG5cIlxuXHRcdFx0KyBcIn0gXFxuXCJcblx0XHRcdCxcblx0QWNjU3VtXHQ9IFwidW5pZm9ybSBUZW5zb3IgQTsgXFxuXCJcblx0XHRcdCsgXCJmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykgeyBcXG5cIlxuXHRcdFx0XHQrIFwiZmxvYXQgYWNjID0gMC4wOyBcXG5cIlxuXHRcdFx0XHQrIFwiZm9yKGludCBpID0gMDsgaSA8ICMoQS5zaGFwZSkueTsgaSsrKXsgXFxuXCIgLyogaXRlcmF0ZSBvdmVyIGVhY2ggc2FtcGxlICovXG5cdFx0XHRcdFx0KyBcImFjYyArPSBBLnJlYWQocG9zLngsIGkpOyBcXG5cIlxuXHRcdFx0XHQrIFwifSBcXG5cIlxuXHRcdFx0XHQrIFwiYWNjID0gYWNjIC8gZmxvYXQoIyhBLnNoYXBlKS55KTsgXFxuXCJcblx0XHRcdFx0KyBcInJldHVybiBhY2M7IFxcblwiXG5cdFx0XHQrIFwifSBcXG5cIlxuXHRcdFx0O1xuXG5mdW5jdGlvbiBPdXRwdXQobGF5ZXIsIGluZGV4KSB7XG5cdHRoaXMub3V0cHV0ID0gbnVsbDtcblx0aWYgKGxheWVyLmFjdGl2YXRpb24gPT09IFwic29mdG1heFwiICYmIGxheWVyLmxvc3MgPT09IFwieGVudHJvcHlcIikge1xuXHRcdHRoaXMub3V0cHV0ID0gbmV3IFNvZnRtYXgobGF5ZXIsIGluZGV4KTtcblx0XHR0aGlzLm91dHB1dC5kZWx0YXMgPSB0aGlzLm91dHB1dC50cmFpbjtcblx0XHR0aGlzLnJ1biA9IChpbnB1dCkgPT4ge1xuXHRcdFx0dGhpcy5fb3V0cHV0ID0gdGhpcy5vdXRwdXQucnVuKGlucHV0KTtcblx0XHRcdHJldHVybiB0aGlzLl9vdXRwdXQ7XG5cdFx0fTtcblx0fSBlbHNlIHtcblx0XHRzd2l0Y2ggKGxheWVyLmxvc3MpIHtcblx0XHRcdGNhc2UgXCJ4ZW50cm9weVwiOlxuXHRcdFx0XHR0aGlzLm91dHB1dCA9IG5ldyBDcm9zc0VudHJvcHkoKTtcblx0XHRcdFx0YnJlYWs7XG5cdFx0XHRjYXNlIFwibXNlXCI6XG5cdFx0XHRcdHRoaXMub3V0cHV0ID0gbmV3IE1TRSgpO1xuXHRcdFx0XHRicmVhaztcblx0XHR9XG5cdFx0dGhpcy5ydW4gPSB0aGlzLnJ1bi5iaW5kKHRoaXMpO1xuXHRcdHRoaXMudHJhaW4gPSB0aGlzLnRyYWluLmJpbmQodGhpcyk7XG5cdH1cblxuXHR0aGlzLl9vdXRwdXQgPSBudWxsO1xuXHR0aGlzLmFjY3VyYWN5ID0gMDtcbn1cbk91dHB1dC5wcm90b3R5cGUucnVuID0gZnVuY3Rpb24oaW5wdXQpIHtcblx0dGhpcy5fb3V0cHV0ID0gaW5wdXQ7XG5cdHJldHVybiBpbnB1dDtcbn07XG5PdXRwdXQucHJvdG90eXBlLnRyYWluID0gZnVuY3Rpb24oZXhwZWN0ZWQpIHtcblx0aWYgKGV4cGVjdGVkIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KVxuXHRcdGV4cGVjdGVkID0gbmV3IFRGLlRlbnNvcihHTCwgbmRhcnJheSggZXhwZWN0ZWQsIHRoaXMuX291dHB1dC5zaGFwZSkpO1xuXG5cdHRoaXMuYmF0Y2hBY2N1cmFjeSA9IG5ldyBURi5PdXRwdXRUZW5zb3IoR0wsIHRoaXMuX291dHB1dC5zaGFwZSk7XG5cdHRoaXMuX2FjY3VyYWN5ID0gbmV3IFRGLk91dHB1dFRlbnNvcihHTCwgWzFdKTtcblx0dGhpcy5iYXRjaEFjY3VyYWN5LnJ1bihBY2N1cmFjeSwgeyBPOiB0aGlzLl9vdXRwdXQsIEU6IGV4cGVjdGVkIH0pO1xuXHR0aGlzLl9hY2N1cmFjeS5ydW4oQWNjU3VtLCB7IEE6IHRoaXMuYmF0Y2hBY2N1cmFjeSB9KTtcblx0dGhpcy5hY2N1cmFjeSA9IHRoaXMuX2FjY3VyYWN5LnJlYWQoKS5kYXRhWzBdO1xuXG5cdHJldHVybiB0aGlzLm91dHB1dC5kZWx0YXModGhpcy5fb3V0cHV0LCBleHBlY3RlZCk7XG59O1xuXG5cbm1vZHVsZS5leHBvcnRzID0gZnVuY3Rpb24odGVuc29yZmlyZSwgZ2xDb250ZXh0KSB7XG5cblx0VEYgPSB0ZW5zb3JmaXJlO1xuXHRHTCA9IGdsQ29udGV4dDtcblxuXHRTb2Z0bWF4ID0gU29mdG1heCh0ZW5zb3JmaXJlLCBnbENvbnRleHQpO1xuXHRNU0UgPSBNU0UodGVuc29yZmlyZSwgZ2xDb250ZXh0KTtcblx0Q3Jvc3NFbnRyb3B5ID0gQ3Jvc3NFbnRyb3B5KHRlbnNvcmZpcmUsIGdsQ29udGV4dCk7XG5cblx0cmV0dXJuIE91dHB1dDtcbn07IiwidmFyIG5kYXJyYXkgPSByZXF1aXJlKFwibmRhcnJheVwiKSxcblx0VEYsXG5cdEdMLFxuXG5cdEFjdCBcdD0gXCJ1bmlmb3JtIFRlbnNvciBPOyBcXG5cIiAvKiBpbnB1dCAqL1xuXHRcdFx0KyBcImZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7IFxcblwiXG5cdFx0XHRcdCsgXCJmbG9hdCBuID0gTy5yZWFkKHBvcyk7IFxcblwiXG5cdFx0XHRcdCsgXCJmbG9hdCBvOyBcXG5cIlxuXHRcdFx0XHQrIFwiZmxvYXQgayA9IDAuMDsgXFxuXCJcblx0XHRcdFx0KyBcImZvcihpbnQgaSA9IDA7IGkgPCAjKE8uc2hhcGUpLng7IGkrKyl7IFxcblwiXG5cdFx0XHRcdFx0KyBcImsgKz0gZXhwKE8ucmVhZChpLCBwb3MueSkpOyBcXG5cIlxuXHRcdFx0XHQrIFwifSBcXG5cIlxuXHRcdFx0XHQrIFwicmV0dXJuIGV4cChuKSAvIGs7IFxcblwiXG5cdFx0XHQrIFwifSBcXG5cIlxuXHRcdFx0LFxuXHRHcmFkXHQ9IFwidW5pZm9ybSBUZW5zb3IgTzsgXFxuXCJcblx0XHRcdCsgXCJ1bmlmb3JtIFRlbnNvciBFOyBcXG5cIlxuXHRcdFx0KyBcImZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7IFxcblwiXG5cdFx0XHRcdCsgXCJyZXR1cm4gTy5yZWFkKHBvcykgLSBFLnJlYWQocG9zKTsgXFxuXCJcblx0XHRcdCsgXCJ9IFxcblwiXG5cdFx0XHQsXG5cdExvc3MgXHQ9IFwidW5pZm9ybSBUZW5zb3IgRzsgXFxuXCJcblx0XHRcdCsgXCJmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykgeyBcXG5cIlxuXHRcdFx0XHQrIFwiZmxvYXQgbG9zcyA9IDAuMDsgXFxuXCJcblx0XHRcdFx0KyBcImZvcihpbnQgaSA9IDA7IGkgPCAjKEcuc2hhcGUpLnk7IGkrKyl7IFxcblwiIC8qIGl0ZXJhdGUgb3ZlciBlYWNoIHNhbXBsZSAqL1xuXHRcdFx0XHRcdCsgXCJmbG9hdCBsID0gMC4wOyBcXG5cIlxuXHRcdFx0XHRcdCsgXCJmb3IoaW50IGogPSAwOyBqIDwgIyhHLnNoYXBlKS54OyBqKyspeyBcXG5cIiAvKiBpdGVyYXRlIG92ZXIgZXZlcnkgb3V0cHV0IGFuZCBjYWxjdWxhdGUgYXZlcmFnZSAqL1xuXHRcdFx0XHRcdFx0KyBcImwgKz0gcG93KGZsb2F0KEcucmVhZChqLCBpKSksIDIuMCkgLyBmbG9hdCgjKEcuc2hhcGUpLngpOyBcXG5cIlxuXHRcdFx0XHRcdCsgXCJ9IFxcblwiXG5cdFx0XHRcdFx0KyBcImxvc3MgKz0gbCAvIGZsb2F0KCMoRy5zaGFwZSkueSk7IFxcblwiXG5cdFx0XHRcdCsgXCJ9IFxcblwiXG5cdFx0XHRcdCsgXCJyZXR1cm4gbG9zczsgXFxuXCJcblx0XHRcdCsgXCJ9IFxcblwiXG5cdFx0XHQ7XG5cblxuZnVuY3Rpb24gU29mdG1heChsYXllciwgaW5kZXgpIHtcblx0dGhpcy5sID0gaW5kZXg7XG5cblx0dGhpcy5hY3RpdmF0aW9uID0gQWN0O1xuXG5cdHRoaXMuZ3JhZGllbnQgPSBHcmFkO1xuXHR0aGlzLmxvc3NGID0gTG9zcztcdFxuXG5cdHRoaXMuaW5wdXQgPSBudWxsO1xuXHR0aGlzLm91dHB1dCA9IG51bGw7XG5cdHRoaXMubG9zcyA9IG5ldyBURi5PdXRwdXRUZW5zb3IoR0wsIFsxXSk7XG59XG5Tb2Z0bWF4LnByb3RvdHlwZS5ydW4gPSBmdW5jdGlvbihpbnB1dCkge1xuXG5cdHRoaXMuaW5wdXQgPSBpbnB1dDtcblx0dGhpcy5vdXRwdXQgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCB0aGlzLmlucHV0LnNoYXBlKTtcblxuXHR0aGlzLm91dHB1dC5ydW4odGhpcy5hY3RpdmF0aW9uLCB7TzogdGhpcy5pbnB1dH0pO1xuXG5cdC8vY29uc29sZS5sb2coXCJvdXRwdXQgXCIgKyB0aGlzLmwgKyBcIjogXCIgKyB0aGlzLm91dHB1dC5yZWFkKCkuZGF0YSk7XG5cdHJldHVybiB0aGlzLm91dHB1dDtcbn1cblNvZnRtYXgucHJvdG90eXBlLnRyYWluID0gZnVuY3Rpb24oZXhwZWN0ZWQpIHtcblx0dmFyIHBhcnRpYWwgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCB0aGlzLmlucHV0LnNoYXBlKTtcblxuXHRpZiAoZXhwZWN0ZWQgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkpXG5cdFx0ZXhwZWN0ZWQgPSBuZXcgVEYuVGVuc29yKEdMLCBuZGFycmF5KGV4cGVjdGVkLCB0aGlzLmlucHV0LnNoYXBlKSk7XG5cblx0Ly8gY2FsY3VsYXRlIHVwc3RyZWFtIGVycm9yc1xuXHRwYXJ0aWFsLnJ1bih0aGlzLmdyYWRpZW50LCB7TzogdGhpcy5vdXRwdXQsIEU6IGV4cGVjdGVkfSk7XG5cblx0Ly8gY2FsY3VsYXRlIGJhdGNoIHRyYWluaW5nIGxvc3Ncblx0dGhpcy5sb3NzLnJ1bih0aGlzLmxvc3NGLCB7IEc6IHBhcnRpYWwgfSk7XG5cdC8vY29uc29sZS5sb2codGhpcy5sb3NzLnJlYWQoKSk7XG5cblx0cmV0dXJuIHBhcnRpYWw7XG59XG5cbm1vZHVsZS5leHBvcnRzID0gZnVuY3Rpb24odGVuc29yZmlyZSwgZ2xDb250ZXh0KSB7XG5cdFRGID0gdGVuc29yZmlyZTtcblx0R0wgPSBnbENvbnRleHQ7XG5cdHJldHVybiBTb2Z0bWF4O1xufSIsIi8vIFN0YW5kYXJkIE5vcm1hbCB2YXJpYXRlIHVzaW5nIEJveC1NdWxsZXIgdHJhbnNmb3JtLlxuZnVuY3Rpb24gcmFuZG9tKG1lYW4sIHN0ZERldikge1xuXHRtZWFuID0gbWVhbiB8fCAwO1xuXHRzdGREZXYgPSBzdGREZXYgfHwgMTtcbiAgICB2YXIgdSA9IDAsIHYgPSAwO1xuICAgIHdoaWxlKHUgPT09IDApIHUgPSBNYXRoLnJhbmRvbSgpOyAvL0NvbnZlcnRpbmcgWzAsMSkgdG8gKDAsMSlcbiAgICB3aGlsZSh2ID09PSAwKSB2ID0gTWF0aC5yYW5kb20oKTtcbiAgICAvL3JldHVybiAwLjQ7XG4gICAgcmV0dXJuIChNYXRoLnNxcnQoIC0yLjAgKiBNYXRoLmxvZyggdSApICkgKiBNYXRoLmNvcyggMi4wICogTWF0aC5QSSAqIHYgKSkgKiBzdGREZXYgKyBtZWFuO1xufVxuXG5tb2R1bGUuZXhwb3J0cyA9IGZ1bmN0aW9uIGdlbmVyYXRlV2VpZ2h0cyhzaGFwZSwgYmlhcykge1xuXHR2YXIgcmVzdWx0ID0gbmV3IEZsb2F0MzJBcnJheShzaGFwZVswXSAqIHNoYXBlWzFdICsgYmlhcyk7XG5cdHZhciBsID0gLTE7XG5cdHdoaWxlICgrK2wgPCByZXN1bHQubGVuZ3RoKSB7XG5cdFx0cmVzdWx0W2xdID0gcmFuZG9tKDAsIE1hdGguc3FydCgyIC8gc2hhcGVbMV0pKTtcblx0fVxuXHQvL2NvbnNvbGUubG9nKHJlc3VsdFswXSk7XG5cdHJldHVybiByZXN1bHQ7XG59IiwidmFyIHNwcmludGYgPSByZXF1aXJlKCdzcHJpbnRmJyk7XG5tb2R1bGUuZXhwb3J0cyA9IGZvcm1hdDtcblxuZnVuY3Rpb24gZm9ybWF0ICh4LCBieXRlcykge1xuICAgIGlmIChieXRlcyA9PT0gdW5kZWZpbmVkKSBieXRlcyA9IDg7XG4gICAgdmFyIHJmbXQgPSAnJScgKyBieXRlcyArICcuJyArIGJ5dGVzICsgJ3MnO1xuICAgIFxuICAgIGlmIChieXRlcyA8PSAwKSByZXR1cm4gdW5kZWZpbmVkO1xuICAgIGlmIChpc05hTih4KSkgcmV0dXJuIHNwcmludGYocmZtdCwgJ05hTicpO1xuICAgIGlmICh4ID09PSBJbmZpbml0eSkge1xuICAgICAgICBpZiAoYnl0ZXMgPT09IDEpIHJldHVybiB1bmRlZmluZWQ7XG4gICAgICAgIHJldHVybiBzcHJpbnRmKHJmbXQsIGJ5dGVzID49IDkgPyAnSW5maW5pdHknIDogJyBJbmYnKS5zbGljZSgwLCBieXRlcyk7XG4gICAgfVxuICAgIGlmICh4ID09PSAtSW5maW5pdHkpIHtcbiAgICAgICAgaWYgKGJ5dGVzID09PSAxKSByZXR1cm4gdW5kZWZpbmVkO1xuICAgICAgICByZXR1cm4gc3ByaW50ZihyZm10LCBieXRlcyA+PSA5ID8gJy1JbmZpbml0eScgOiAnLUluZicpLnNsaWNlKDAsIGJ5dGVzKTtcbiAgICB9XG4gICAgcmV0dXJuIHBhY2tmKHgsIGJ5dGVzKTtcbn07XG5cbmZ1bmN0aW9uIHNjaSAoeCwgYnl0ZXMpIHtcbiAgICB2YXIgbiA9IE1hdGgubWF4KDEsIGxvZzEwZihNYXRoLmFicyh4KSkpO1xuICAgIHZhciBzeiA9IGxvZzEwZihNYXRoLmFicyhuKSk7XG4gICAgXG4gICAgdmFyIGIgPSBNYXRoLnBvdygxMCxieXRlcysxKTtcbiAgICBpZiAoTWF0aC5hYnMoeCkgPCAxKSB7XG4gICAgICAgIHggPSBNYXRoLnJvdW5kKHggKiBiKSAvIGI7XG4gICAgfVxuICAgIGVsc2Uge1xuICAgICAgICB2YXIgdG4gPSBNYXRoLnBvdygxMCwgbiArIDEpO1xuICAgICAgICB4ID0gTWF0aC5yb3VuZCh4IC8gdG4gKiBiKSAvIGIgKiB0bjtcbiAgICB9XG4gICAgXG4gICAgdmFyIHM7XG4gICAgaWYgKGJ5dGVzIC0gc3ogLSA2ID09PSAtMSkge1xuICAgICAgICB4ID0gTWF0aC5yb3VuZCh4IC8gTWF0aC5wb3coMTAsIG4pKTtcbiAgICAgICAgeCA9IHggKiBNYXRoLnBvdygxMCwgbik7XG4gICAgICAgIHMgPSBzcHJpbnRmKCclMWUnLCB4KS5yZXBsYWNlKC9cXC5bXmVdKy8sICcnKTtcbiAgICB9XG4gICAgZWxzZSBpZiAoYnl0ZXMgLSBzeiAtIDYgPCAwKSByZXR1cm4gdW5kZWZpbmVkO1xuICAgIGVsc2Uge1xuICAgICAgICBzID0gc3ByaW50ZignJS4nICsgKGJ5dGVzIC0gc3ogLSA2KSArICdlJywgeCk7XG4gICAgfVxuICAgIGlmICh4ID4gMCkgcyA9ICcgJyArIHM7XG4gICAgcmV0dXJuIHBhZChzLCBieXRlcyk7XG59XG5cbmZ1bmN0aW9uIHBhZCAocywgYnl0ZXMpIHtcbiAgICByZXR1cm4gQXJyYXkoTWF0aC5tYXgoMCwgYnl0ZXMgLSBzLmxlbmd0aCArIDEpKS5qb2luKCcgJykgKyBzO1xufVxuXG5mdW5jdGlvbiBsb2cxMGYgKG4pIHtcbiAgICByZXR1cm4gTWF0aC5mbG9vcihNYXRoLmxvZyhuKSAvIE1hdGgubG9nKDEwKSk7XG59XG5cbmZ1bmN0aW9uIHBhY2tmICh4LCBieXRlcykge1xuICAgIHZhciBsYnl0ZXMgPSBNYXRoLm1heCgxLCBNYXRoLmZsb29yKChieXRlcyAtIDIpIC8gMikpO1xuICAgIHZhciByYnl0ZXMgPSBieXRlcyAtIGxieXRlcyAtIDI7XG4gICAgXG4gICAgaWYgKHggPT09IDAgJiYgYnl0ZXMgPCA0KSB7XG4gICAgICAgIHJldHVybiBwYWQoJzAnLCBieXRlcyk7XG4gICAgfVxuICAgIGVsc2UgaWYgKHggPT09IDApIHtcbiAgICAgICAgcmV0dXJuIHBhZCgnMC4nICsgQXJyYXkocmJ5dGVzKzEpLmpvaW4oJzAnKSwgYnl0ZXMpO1xuICAgIH1cbiAgICBcbiAgICBpZiAocmJ5dGVzIDw9IDApIHtcbiAgICAgICAgdmFyIHMgPSBzcHJpbnRmKCclJyArIGxieXRlcyArICdmJywgeCk7XG4gICAgICAgIGlmICh4ID49IDApIHMgPSAnICcgKyBzO1xuICAgICAgICBpZiAocy5sZW5ndGggPiBieXRlcykgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICAgICAgcmV0dXJuIHBhZChzLCBieXRlcyk7XG4gICAgfVxuICAgIGlmIChNYXRoLmFicyh4KSA8IE1hdGgucG93KDEwLDEtcmJ5dGVzKSkgcmV0dXJuIHNjaSh4LCBieXRlcyk7XG4gICAgXG4gICAgdmFyIGIgPSBNYXRoLnBvdygxMCxieXRlcy0zKTtcbiAgICB2YXIgdG4gPSBNYXRoLnBvdygxMCwgbG9nMTBmKE1hdGguYWJzKHgpKSk7XG4gICAgdmFyIHhyID0gTWF0aC5yb3VuZCh4IC8gdG4gKiBiKSAvIGIgKiB0bjtcbiAgICBcbiAgICB2YXIgcyA9IHNwcmludGYoJyUnICsgbGJ5dGVzICsgJy4nICsgcmJ5dGVzICsgJ2YnLCB4cik7XG4gICAgaWYgKHhyID4gMCkgcyA9ICcgJyArIHM7XG4gICAgcyA9IHMuc2xpY2UoMCwgYnl0ZXMpO1xuICAgIHZhciByID0gcy5zcGxpdCgnLicpWzFdO1xuICAgIGlmICghciB8fCByLmxlbmd0aCA8IDEpIHJldHVybiBzY2koeHIsIGJ5dGVzKTtcbiAgICByZXR1cm4gcGFkKHMsIGJ5dGVzKS5zbGljZSgwLCBieXRlcyk7XG59XG4iLCJcInVzZSBzdHJpY3RcIlxuXG5mdW5jdGlvbiBpb3RhKG4pIHtcbiAgdmFyIHJlc3VsdCA9IG5ldyBBcnJheShuKVxuICBmb3IodmFyIGk9MDsgaTxuOyArK2kpIHtcbiAgICByZXN1bHRbaV0gPSBpXG4gIH1cbiAgcmV0dXJuIHJlc3VsdFxufVxuXG5tb2R1bGUuZXhwb3J0cyA9IGlvdGEiLCIvKiFcbiAqIERldGVybWluZSBpZiBhbiBvYmplY3QgaXMgYSBCdWZmZXJcbiAqXG4gKiBAYXV0aG9yICAgRmVyb3NzIEFib3VraGFkaWplaCA8ZmVyb3NzQGZlcm9zcy5vcmc+IDxodHRwOi8vZmVyb3NzLm9yZz5cbiAqIEBsaWNlbnNlICBNSVRcbiAqL1xuXG4vLyBUaGUgX2lzQnVmZmVyIGNoZWNrIGlzIGZvciBTYWZhcmkgNS03IHN1cHBvcnQsIGJlY2F1c2UgaXQncyBtaXNzaW5nXG4vLyBPYmplY3QucHJvdG90eXBlLmNvbnN0cnVjdG9yLiBSZW1vdmUgdGhpcyBldmVudHVhbGx5XG5tb2R1bGUuZXhwb3J0cyA9IGZ1bmN0aW9uIChvYmopIHtcbiAgcmV0dXJuIG9iaiAhPSBudWxsICYmIChpc0J1ZmZlcihvYmopIHx8IGlzU2xvd0J1ZmZlcihvYmopIHx8ICEhb2JqLl9pc0J1ZmZlcilcbn1cblxuZnVuY3Rpb24gaXNCdWZmZXIgKG9iaikge1xuICByZXR1cm4gISFvYmouY29uc3RydWN0b3IgJiYgdHlwZW9mIG9iai5jb25zdHJ1Y3Rvci5pc0J1ZmZlciA9PT0gJ2Z1bmN0aW9uJyAmJiBvYmouY29uc3RydWN0b3IuaXNCdWZmZXIob2JqKVxufVxuXG4vLyBGb3IgTm9kZSB2MC4xMCBzdXBwb3J0LiBSZW1vdmUgdGhpcyBldmVudHVhbGx5LlxuZnVuY3Rpb24gaXNTbG93QnVmZmVyIChvYmopIHtcbiAgcmV0dXJuIHR5cGVvZiBvYmoucmVhZEZsb2F0TEUgPT09ICdmdW5jdGlvbicgJiYgdHlwZW9mIG9iai5zbGljZSA9PT0gJ2Z1bmN0aW9uJyAmJiBpc0J1ZmZlcihvYmouc2xpY2UoMCwgMCkpXG59XG4iLCJ2YXIgc2hvd2YgPSByZXF1aXJlKCdmaXhlZC13aWR0aC1mbG9hdCcpO1xudmFyIG5kYXJyYXkgPSByZXF1aXJlKCduZGFycmF5Jyk7XG5cbm1vZHVsZS5leHBvcnRzID0gZnVuY3Rpb24gKG0sIG9wdHMpIHtcbiAgICBpZiAoIW9wdHMpIG9wdHMgPSB7fTtcbiAgICBpZiAodHlwZW9mIG9wdHMgPT09ICdudW1iZXInKSBvcHRzID0geyB3aWR0aDogb3B0cyB9O1xuICAgIGlmICghb3B0cy53aWR0aCkgb3B0cy53aWR0aCA9IDg7XG5cbiAgICBpZiAobS5kaW1lbnNpb24gPT09IHVuZGVmaW5lZCkge1xuICAgICAgICBtID0gbmRhcnJheShtKTtcbiAgICB9XG5cbiAgICBpZiAobS5kaW1lbnNpb24gPT09IDEpIHJldHVybiBkMShtLCBvcHRzKTtcbiAgICBpZiAobS5kaW1lbnNpb24gPT09IDIpIHJldHVybiBkMihtLCBvcHRzKTtcbiAgICBpZiAobS5kaW1lbnNpb24gPT09IDMpIHJldHVybiBkMyhtLCBvcHRzKTtcbiAgICBpZiAobS5kaW1lbnNpb24gPT09IDQpIHJldHVybiBkNChtLCBvcHRzKTtcbn07XG5cbmZ1bmN0aW9uIGQxIChtLCBvcHRzKSB7XG4gICAgdmFyIHRlcm1zID0gW107XG4gICAgZm9yICh2YXIgaSA9IDA7IGkgPCBtLnNoYXBlWzBdOyBpKyspIHtcbiAgICAgICAgdGVybXMucHVzaChzaG93ZihtLmdldChpKSwgb3B0cy53aWR0aCkpO1xuICAgIH1cbiAgICByZXR1cm4gdGVybXMuam9pbignICcpO1xufVxuXG5mdW5jdGlvbiBkMiAobSwgb3B0cykge1xuICAgIHZhciByb3dzID0gW107XG4gICAgZm9yICh2YXIgeSA9IDA7IHkgPCBtLnNoYXBlWzBdOyB5KyspIHtcbiAgICAgICAgcm93cy5wdXNoKGQxKG0ucGljayh5LCBudWxsKSwgb3B0cykpO1xuICAgIH1cbiAgICByZXR1cm4gcm93cy5qb2luKCdcXG4nKTtcbn1cblxuZnVuY3Rpb24gZDMgKG0sIG9wdHMpIHtcbiAgICB2YXIgcm93cyA9IFtdO1xuICAgIGZvciAodmFyIHogPSAwOyB6IDwgbS5zaGFwZVswXTsgeisrKSB7XG4gICAgICAgIHJvd3MucHVzaChkMihtLnBpY2soeiwgbnVsbCwgbnVsbCksIG9wdHMpLCAnJyk7XG4gICAgfVxuICAgIHJldHVybiByb3dzLmpvaW4oJ1xcbicpO1xufVxuXG5mdW5jdGlvbiBkNCAobSwgb3B0cykge1xuICAgIHZhciByb3dzID0gW10sIGxlbiA9IDNcbiAgICBmb3IgKHZhciB3ID0gMDsgdyA8IG0uc2hhcGVbMF07IHcrKykge1xuICAgICAgICB2YXIgciA9IGQzKG0ucGljayh3LCBudWxsLCBudWxsLCBudWxsKSwgb3B0cylcbiAgICAgICAgcm93cy5wdXNoKHIpO1xuICAgICAgICB2YXIgbGluZXMgPSByLnNwbGl0KCdcXG4nKTtcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBsaW5lcy5sZW5ndGg7IGkrKykge1xuICAgICAgICAgICAgbGVuID0gTWF0aC5tYXgobGVuLCBsaW5lc1tpXS5sZW5ndGgpO1xuICAgICAgICB9XG4gICAgfVxuICAgIHJldHVybiByb3dzLmpvaW4oJ1xcbicgKyBBcnJheShsZW4rMSkuam9pbignLScpICsgJ1xcblxcbicpO1xufVxuIiwidmFyIGlvdGEgPSByZXF1aXJlKFwiaW90YS1hcnJheVwiKVxudmFyIGlzQnVmZmVyID0gcmVxdWlyZShcImlzLWJ1ZmZlclwiKVxuXG52YXIgaGFzVHlwZWRBcnJheXMgID0gKCh0eXBlb2YgRmxvYXQ2NEFycmF5KSAhPT0gXCJ1bmRlZmluZWRcIilcblxuZnVuY3Rpb24gY29tcGFyZTFzdChhLCBiKSB7XG4gIHJldHVybiBhWzBdIC0gYlswXVxufVxuXG5mdW5jdGlvbiBvcmRlcigpIHtcbiAgdmFyIHN0cmlkZSA9IHRoaXMuc3RyaWRlXG4gIHZhciB0ZXJtcyA9IG5ldyBBcnJheShzdHJpZGUubGVuZ3RoKVxuICB2YXIgaVxuICBmb3IoaT0wOyBpPHRlcm1zLmxlbmd0aDsgKytpKSB7XG4gICAgdGVybXNbaV0gPSBbTWF0aC5hYnMoc3RyaWRlW2ldKSwgaV1cbiAgfVxuICB0ZXJtcy5zb3J0KGNvbXBhcmUxc3QpXG4gIHZhciByZXN1bHQgPSBuZXcgQXJyYXkodGVybXMubGVuZ3RoKVxuICBmb3IoaT0wOyBpPHJlc3VsdC5sZW5ndGg7ICsraSkge1xuICAgIHJlc3VsdFtpXSA9IHRlcm1zW2ldWzFdXG4gIH1cbiAgcmV0dXJuIHJlc3VsdFxufVxuXG5mdW5jdGlvbiBjb21waWxlQ29uc3RydWN0b3IoZHR5cGUsIGRpbWVuc2lvbikge1xuICB2YXIgY2xhc3NOYW1lID0gW1wiVmlld1wiLCBkaW1lbnNpb24sIFwiZFwiLCBkdHlwZV0uam9pbihcIlwiKVxuICBpZihkaW1lbnNpb24gPCAwKSB7XG4gICAgY2xhc3NOYW1lID0gXCJWaWV3X05pbFwiICsgZHR5cGVcbiAgfVxuICB2YXIgdXNlR2V0dGVycyA9IChkdHlwZSA9PT0gXCJnZW5lcmljXCIpXG5cbiAgaWYoZGltZW5zaW9uID09PSAtMSkge1xuICAgIC8vU3BlY2lhbCBjYXNlIGZvciB0cml2aWFsIGFycmF5c1xuICAgIHZhciBjb2RlID1cbiAgICAgIFwiZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiKGEpe3RoaXMuZGF0YT1hO307XFxcbnZhciBwcm90bz1cIitjbGFzc05hbWUrXCIucHJvdG90eXBlO1xcXG5wcm90by5kdHlwZT0nXCIrZHR5cGUrXCInO1xcXG5wcm90by5pbmRleD1mdW5jdGlvbigpe3JldHVybiAtMX07XFxcbnByb3RvLnNpemU9MDtcXFxucHJvdG8uZGltZW5zaW9uPS0xO1xcXG5wcm90by5zaGFwZT1wcm90by5zdHJpZGU9cHJvdG8ub3JkZXI9W107XFxcbnByb3RvLmxvPXByb3RvLmhpPXByb3RvLnRyYW5zcG9zZT1wcm90by5zdGVwPVxcXG5mdW5jdGlvbigpe3JldHVybiBuZXcgXCIrY2xhc3NOYW1lK1wiKHRoaXMuZGF0YSk7fTtcXFxucHJvdG8uZ2V0PXByb3RvLnNldD1mdW5jdGlvbigpe307XFxcbnByb3RvLnBpY2s9ZnVuY3Rpb24oKXtyZXR1cm4gbnVsbH07XFxcbnJldHVybiBmdW5jdGlvbiBjb25zdHJ1Y3RfXCIrY2xhc3NOYW1lK1wiKGEpe3JldHVybiBuZXcgXCIrY2xhc3NOYW1lK1wiKGEpO31cIlxuICAgIHZhciBwcm9jZWR1cmUgPSBuZXcgRnVuY3Rpb24oY29kZSlcbiAgICByZXR1cm4gcHJvY2VkdXJlKClcbiAgfSBlbHNlIGlmKGRpbWVuc2lvbiA9PT0gMCkge1xuICAgIC8vU3BlY2lhbCBjYXNlIGZvciAwZCBhcnJheXNcbiAgICB2YXIgY29kZSA9XG4gICAgICBcImZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIihhLGQpIHtcXFxudGhpcy5kYXRhID0gYTtcXFxudGhpcy5vZmZzZXQgPSBkXFxcbn07XFxcbnZhciBwcm90bz1cIitjbGFzc05hbWUrXCIucHJvdG90eXBlO1xcXG5wcm90by5kdHlwZT0nXCIrZHR5cGUrXCInO1xcXG5wcm90by5pbmRleD1mdW5jdGlvbigpe3JldHVybiB0aGlzLm9mZnNldH07XFxcbnByb3RvLmRpbWVuc2lvbj0wO1xcXG5wcm90by5zaXplPTE7XFxcbnByb3RvLnNoYXBlPVxcXG5wcm90by5zdHJpZGU9XFxcbnByb3RvLm9yZGVyPVtdO1xcXG5wcm90by5sbz1cXFxucHJvdG8uaGk9XFxcbnByb3RvLnRyYW5zcG9zZT1cXFxucHJvdG8uc3RlcD1mdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfY29weSgpIHtcXFxucmV0dXJuIG5ldyBcIitjbGFzc05hbWUrXCIodGhpcy5kYXRhLHRoaXMub2Zmc2V0KVxcXG59O1xcXG5wcm90by5waWNrPWZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl9waWNrKCl7XFxcbnJldHVybiBUcml2aWFsQXJyYXkodGhpcy5kYXRhKTtcXFxufTtcXFxucHJvdG8udmFsdWVPZj1wcm90by5nZXQ9ZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX2dldCgpe1xcXG5yZXR1cm4gXCIrKHVzZUdldHRlcnMgPyBcInRoaXMuZGF0YS5nZXQodGhpcy5vZmZzZXQpXCIgOiBcInRoaXMuZGF0YVt0aGlzLm9mZnNldF1cIikrXG5cIn07XFxcbnByb3RvLnNldD1mdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfc2V0KHYpe1xcXG5yZXR1cm4gXCIrKHVzZUdldHRlcnMgPyBcInRoaXMuZGF0YS5zZXQodGhpcy5vZmZzZXQsdilcIiA6IFwidGhpcy5kYXRhW3RoaXMub2Zmc2V0XT12XCIpK1wiXFxcbn07XFxcbnJldHVybiBmdW5jdGlvbiBjb25zdHJ1Y3RfXCIrY2xhc3NOYW1lK1wiKGEsYixjLGQpe3JldHVybiBuZXcgXCIrY2xhc3NOYW1lK1wiKGEsZCl9XCJcbiAgICB2YXIgcHJvY2VkdXJlID0gbmV3IEZ1bmN0aW9uKFwiVHJpdmlhbEFycmF5XCIsIGNvZGUpXG4gICAgcmV0dXJuIHByb2NlZHVyZShDQUNIRURfQ09OU1RSVUNUT1JTW2R0eXBlXVswXSlcbiAgfVxuXG4gIHZhciBjb2RlID0gW1wiJ3VzZSBzdHJpY3QnXCJdXG5cbiAgLy9DcmVhdGUgY29uc3RydWN0b3IgZm9yIHZpZXdcbiAgdmFyIGluZGljZXMgPSBpb3RhKGRpbWVuc2lvbilcbiAgdmFyIGFyZ3MgPSBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7IHJldHVybiBcImlcIitpIH0pXG4gIHZhciBpbmRleF9zdHIgPSBcInRoaXMub2Zmc2V0K1wiICsgaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkge1xuICAgICAgICByZXR1cm4gXCJ0aGlzLnN0cmlkZVtcIiArIGkgKyBcIl0qaVwiICsgaVxuICAgICAgfSkuam9pbihcIitcIilcbiAgdmFyIHNoYXBlQXJnID0gaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkge1xuICAgICAgcmV0dXJuIFwiYlwiK2lcbiAgICB9KS5qb2luKFwiLFwiKVxuICB2YXIgc3RyaWRlQXJnID0gaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkge1xuICAgICAgcmV0dXJuIFwiY1wiK2lcbiAgICB9KS5qb2luKFwiLFwiKVxuICBjb2RlLnB1c2goXG4gICAgXCJmdW5jdGlvbiBcIitjbGFzc05hbWUrXCIoYSxcIiArIHNoYXBlQXJnICsgXCIsXCIgKyBzdHJpZGVBcmcgKyBcIixkKXt0aGlzLmRhdGE9YVwiLFxuICAgICAgXCJ0aGlzLnNoYXBlPVtcIiArIHNoYXBlQXJnICsgXCJdXCIsXG4gICAgICBcInRoaXMuc3RyaWRlPVtcIiArIHN0cmlkZUFyZyArIFwiXVwiLFxuICAgICAgXCJ0aGlzLm9mZnNldD1kfDB9XCIsXG4gICAgXCJ2YXIgcHJvdG89XCIrY2xhc3NOYW1lK1wiLnByb3RvdHlwZVwiLFxuICAgIFwicHJvdG8uZHR5cGU9J1wiK2R0eXBlK1wiJ1wiLFxuICAgIFwicHJvdG8uZGltZW5zaW9uPVwiK2RpbWVuc2lvbilcblxuICAvL3ZpZXcuc2l6ZTpcbiAgY29kZS5wdXNoKFwiT2JqZWN0LmRlZmluZVByb3BlcnR5KHByb3RvLCdzaXplJyx7Z2V0OmZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl9zaXplKCl7XFxcbnJldHVybiBcIitpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7IHJldHVybiBcInRoaXMuc2hhcGVbXCIraStcIl1cIiB9KS5qb2luKFwiKlwiKSxcblwifX0pXCIpXG5cbiAgLy92aWV3Lm9yZGVyOlxuICBpZihkaW1lbnNpb24gPT09IDEpIHtcbiAgICBjb2RlLnB1c2goXCJwcm90by5vcmRlcj1bMF1cIilcbiAgfSBlbHNlIHtcbiAgICBjb2RlLnB1c2goXCJPYmplY3QuZGVmaW5lUHJvcGVydHkocHJvdG8sJ29yZGVyJyx7Z2V0OlwiKVxuICAgIGlmKGRpbWVuc2lvbiA8IDQpIHtcbiAgICAgIGNvZGUucHVzaChcImZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl9vcmRlcigpe1wiKVxuICAgICAgaWYoZGltZW5zaW9uID09PSAyKSB7XG4gICAgICAgIGNvZGUucHVzaChcInJldHVybiAoTWF0aC5hYnModGhpcy5zdHJpZGVbMF0pPk1hdGguYWJzKHRoaXMuc3RyaWRlWzFdKSk/WzEsMF06WzAsMV19fSlcIilcbiAgICAgIH0gZWxzZSBpZihkaW1lbnNpb24gPT09IDMpIHtcbiAgICAgICAgY29kZS5wdXNoKFxuXCJ2YXIgczA9TWF0aC5hYnModGhpcy5zdHJpZGVbMF0pLHMxPU1hdGguYWJzKHRoaXMuc3RyaWRlWzFdKSxzMj1NYXRoLmFicyh0aGlzLnN0cmlkZVsyXSk7XFxcbmlmKHMwPnMxKXtcXFxuaWYoczE+czIpe1xcXG5yZXR1cm4gWzIsMSwwXTtcXFxufWVsc2UgaWYoczA+czIpe1xcXG5yZXR1cm4gWzEsMiwwXTtcXFxufWVsc2V7XFxcbnJldHVybiBbMSwwLDJdO1xcXG59XFxcbn1lbHNlIGlmKHMwPnMyKXtcXFxucmV0dXJuIFsyLDAsMV07XFxcbn1lbHNlIGlmKHMyPnMxKXtcXFxucmV0dXJuIFswLDEsMl07XFxcbn1lbHNle1xcXG5yZXR1cm4gWzAsMiwxXTtcXFxufX19KVwiKVxuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICBjb2RlLnB1c2goXCJPUkRFUn0pXCIpXG4gICAgfVxuICB9XG5cbiAgLy92aWV3LnNldChpMCwgLi4uLCB2KTpcbiAgY29kZS5wdXNoKFxuXCJwcm90by5zZXQ9ZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX3NldChcIithcmdzLmpvaW4oXCIsXCIpK1wiLHYpe1wiKVxuICBpZih1c2VHZXR0ZXJzKSB7XG4gICAgY29kZS5wdXNoKFwicmV0dXJuIHRoaXMuZGF0YS5zZXQoXCIraW5kZXhfc3RyK1wiLHYpfVwiKVxuICB9IGVsc2Uge1xuICAgIGNvZGUucHVzaChcInJldHVybiB0aGlzLmRhdGFbXCIraW5kZXhfc3RyK1wiXT12fVwiKVxuICB9XG5cbiAgLy92aWV3LmdldChpMCwgLi4uKTpcbiAgY29kZS5wdXNoKFwicHJvdG8uZ2V0PWZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl9nZXQoXCIrYXJncy5qb2luKFwiLFwiKStcIil7XCIpXG4gIGlmKHVzZUdldHRlcnMpIHtcbiAgICBjb2RlLnB1c2goXCJyZXR1cm4gdGhpcy5kYXRhLmdldChcIitpbmRleF9zdHIrXCIpfVwiKVxuICB9IGVsc2Uge1xuICAgIGNvZGUucHVzaChcInJldHVybiB0aGlzLmRhdGFbXCIraW5kZXhfc3RyK1wiXX1cIilcbiAgfVxuXG4gIC8vdmlldy5pbmRleDpcbiAgY29kZS5wdXNoKFxuICAgIFwicHJvdG8uaW5kZXg9ZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX2luZGV4KFwiLCBhcmdzLmpvaW4oKSwgXCIpe3JldHVybiBcIitpbmRleF9zdHIrXCJ9XCIpXG5cbiAgLy92aWV3LmhpKCk6XG4gIGNvZGUucHVzaChcInByb3RvLmhpPWZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl9oaShcIithcmdzLmpvaW4oXCIsXCIpK1wiKXtyZXR1cm4gbmV3IFwiK2NsYXNzTmFtZStcIih0aGlzLmRhdGEsXCIrXG4gICAgaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkge1xuICAgICAgcmV0dXJuIFtcIih0eXBlb2YgaVwiLGksXCIhPT0nbnVtYmVyJ3x8aVwiLGksXCI8MCk/dGhpcy5zaGFwZVtcIiwgaSwgXCJdOmlcIiwgaSxcInwwXCJdLmpvaW4oXCJcIilcbiAgICB9KS5qb2luKFwiLFwiKStcIixcIitcbiAgICBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7XG4gICAgICByZXR1cm4gXCJ0aGlzLnN0cmlkZVtcIitpICsgXCJdXCJcbiAgICB9KS5qb2luKFwiLFwiKStcIix0aGlzLm9mZnNldCl9XCIpXG5cbiAgLy92aWV3LmxvKCk6XG4gIHZhciBhX3ZhcnMgPSBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7IHJldHVybiBcImFcIitpK1wiPXRoaXMuc2hhcGVbXCIraStcIl1cIiB9KVxuICB2YXIgY192YXJzID0gaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkgeyByZXR1cm4gXCJjXCIraStcIj10aGlzLnN0cmlkZVtcIitpK1wiXVwiIH0pXG4gIGNvZGUucHVzaChcInByb3RvLmxvPWZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl9sbyhcIithcmdzLmpvaW4oXCIsXCIpK1wiKXt2YXIgYj10aGlzLm9mZnNldCxkPTAsXCIrYV92YXJzLmpvaW4oXCIsXCIpK1wiLFwiK2NfdmFycy5qb2luKFwiLFwiKSlcbiAgZm9yKHZhciBpPTA7IGk8ZGltZW5zaW9uOyArK2kpIHtcbiAgICBjb2RlLnB1c2goXG5cImlmKHR5cGVvZiBpXCIraStcIj09PSdudW1iZXInJiZpXCIraStcIj49MCl7XFxcbmQ9aVwiK2krXCJ8MDtcXFxuYis9Y1wiK2krXCIqZDtcXFxuYVwiK2krXCItPWR9XCIpXG4gIH1cbiAgY29kZS5wdXNoKFwicmV0dXJuIG5ldyBcIitjbGFzc05hbWUrXCIodGhpcy5kYXRhLFwiK1xuICAgIGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHtcbiAgICAgIHJldHVybiBcImFcIitpXG4gICAgfSkuam9pbihcIixcIikrXCIsXCIrXG4gICAgaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkge1xuICAgICAgcmV0dXJuIFwiY1wiK2lcbiAgICB9KS5qb2luKFwiLFwiKStcIixiKX1cIilcblxuICAvL3ZpZXcuc3RlcCgpOlxuICBjb2RlLnB1c2goXCJwcm90by5zdGVwPWZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl9zdGVwKFwiK2FyZ3Muam9pbihcIixcIikrXCIpe3ZhciBcIitcbiAgICBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7XG4gICAgICByZXR1cm4gXCJhXCIraStcIj10aGlzLnNoYXBlW1wiK2krXCJdXCJcbiAgICB9KS5qb2luKFwiLFwiKStcIixcIitcbiAgICBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7XG4gICAgICByZXR1cm4gXCJiXCIraStcIj10aGlzLnN0cmlkZVtcIitpK1wiXVwiXG4gICAgfSkuam9pbihcIixcIikrXCIsYz10aGlzLm9mZnNldCxkPTAsY2VpbD1NYXRoLmNlaWxcIilcbiAgZm9yKHZhciBpPTA7IGk8ZGltZW5zaW9uOyArK2kpIHtcbiAgICBjb2RlLnB1c2goXG5cImlmKHR5cGVvZiBpXCIraStcIj09PSdudW1iZXInKXtcXFxuZD1pXCIraStcInwwO1xcXG5pZihkPDApe1xcXG5jKz1iXCIraStcIiooYVwiK2krXCItMSk7XFxcbmFcIitpK1wiPWNlaWwoLWFcIitpK1wiL2QpXFxcbn1lbHNle1xcXG5hXCIraStcIj1jZWlsKGFcIitpK1wiL2QpXFxcbn1cXFxuYlwiK2krXCIqPWRcXFxufVwiKVxuICB9XG4gIGNvZGUucHVzaChcInJldHVybiBuZXcgXCIrY2xhc3NOYW1lK1wiKHRoaXMuZGF0YSxcIitcbiAgICBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7XG4gICAgICByZXR1cm4gXCJhXCIgKyBpXG4gICAgfSkuam9pbihcIixcIikrXCIsXCIrXG4gICAgaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkge1xuICAgICAgcmV0dXJuIFwiYlwiICsgaVxuICAgIH0pLmpvaW4oXCIsXCIpK1wiLGMpfVwiKVxuXG4gIC8vdmlldy50cmFuc3Bvc2UoKTpcbiAgdmFyIHRTaGFwZSA9IG5ldyBBcnJheShkaW1lbnNpb24pXG4gIHZhciB0U3RyaWRlID0gbmV3IEFycmF5KGRpbWVuc2lvbilcbiAgZm9yKHZhciBpPTA7IGk8ZGltZW5zaW9uOyArK2kpIHtcbiAgICB0U2hhcGVbaV0gPSBcImFbaVwiK2krXCJdXCJcbiAgICB0U3RyaWRlW2ldID0gXCJiW2lcIitpK1wiXVwiXG4gIH1cbiAgY29kZS5wdXNoKFwicHJvdG8udHJhbnNwb3NlPWZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl90cmFuc3Bvc2UoXCIrYXJncytcIil7XCIrXG4gICAgYXJncy5tYXAoZnVuY3Rpb24obixpZHgpIHsgcmV0dXJuIG4gKyBcIj0oXCIgKyBuICsgXCI9PT11bmRlZmluZWQ/XCIgKyBpZHggKyBcIjpcIiArIG4gKyBcInwwKVwifSkuam9pbihcIjtcIiksXG4gICAgXCJ2YXIgYT10aGlzLnNoYXBlLGI9dGhpcy5zdHJpZGU7cmV0dXJuIG5ldyBcIitjbGFzc05hbWUrXCIodGhpcy5kYXRhLFwiK3RTaGFwZS5qb2luKFwiLFwiKStcIixcIit0U3RyaWRlLmpvaW4oXCIsXCIpK1wiLHRoaXMub2Zmc2V0KX1cIilcblxuICAvL3ZpZXcucGljaygpOlxuICBjb2RlLnB1c2goXCJwcm90by5waWNrPWZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl9waWNrKFwiK2FyZ3MrXCIpe3ZhciBhPVtdLGI9W10sYz10aGlzLm9mZnNldFwiKVxuICBmb3IodmFyIGk9MDsgaTxkaW1lbnNpb247ICsraSkge1xuICAgIGNvZGUucHVzaChcImlmKHR5cGVvZiBpXCIraStcIj09PSdudW1iZXInJiZpXCIraStcIj49MCl7Yz0oYyt0aGlzLnN0cmlkZVtcIitpK1wiXSppXCIraStcIil8MH1lbHNle2EucHVzaCh0aGlzLnNoYXBlW1wiK2krXCJdKTtiLnB1c2godGhpcy5zdHJpZGVbXCIraStcIl0pfVwiKVxuICB9XG4gIGNvZGUucHVzaChcInZhciBjdG9yPUNUT1JfTElTVFthLmxlbmd0aCsxXTtyZXR1cm4gY3Rvcih0aGlzLmRhdGEsYSxiLGMpfVwiKVxuXG4gIC8vQWRkIHJldHVybiBzdGF0ZW1lbnRcbiAgY29kZS5wdXNoKFwicmV0dXJuIGZ1bmN0aW9uIGNvbnN0cnVjdF9cIitjbGFzc05hbWUrXCIoZGF0YSxzaGFwZSxzdHJpZGUsb2Zmc2V0KXtyZXR1cm4gbmV3IFwiK2NsYXNzTmFtZStcIihkYXRhLFwiK1xuICAgIGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHtcbiAgICAgIHJldHVybiBcInNoYXBlW1wiK2krXCJdXCJcbiAgICB9KS5qb2luKFwiLFwiKStcIixcIitcbiAgICBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7XG4gICAgICByZXR1cm4gXCJzdHJpZGVbXCIraStcIl1cIlxuICAgIH0pLmpvaW4oXCIsXCIpK1wiLG9mZnNldCl9XCIpXG5cbiAgLy9Db21waWxlIHByb2NlZHVyZVxuICB2YXIgcHJvY2VkdXJlID0gbmV3IEZ1bmN0aW9uKFwiQ1RPUl9MSVNUXCIsIFwiT1JERVJcIiwgY29kZS5qb2luKFwiXFxuXCIpKVxuICByZXR1cm4gcHJvY2VkdXJlKENBQ0hFRF9DT05TVFJVQ1RPUlNbZHR5cGVdLCBvcmRlcilcbn1cblxuZnVuY3Rpb24gYXJyYXlEVHlwZShkYXRhKSB7XG4gIGlmKGlzQnVmZmVyKGRhdGEpKSB7XG4gICAgcmV0dXJuIFwiYnVmZmVyXCJcbiAgfVxuICBpZihoYXNUeXBlZEFycmF5cykge1xuICAgIHN3aXRjaChPYmplY3QucHJvdG90eXBlLnRvU3RyaW5nLmNhbGwoZGF0YSkpIHtcbiAgICAgIGNhc2UgXCJbb2JqZWN0IEZsb2F0NjRBcnJheV1cIjpcbiAgICAgICAgcmV0dXJuIFwiZmxvYXQ2NFwiXG4gICAgICBjYXNlIFwiW29iamVjdCBGbG9hdDMyQXJyYXldXCI6XG4gICAgICAgIHJldHVybiBcImZsb2F0MzJcIlxuICAgICAgY2FzZSBcIltvYmplY3QgSW50OEFycmF5XVwiOlxuICAgICAgICByZXR1cm4gXCJpbnQ4XCJcbiAgICAgIGNhc2UgXCJbb2JqZWN0IEludDE2QXJyYXldXCI6XG4gICAgICAgIHJldHVybiBcImludDE2XCJcbiAgICAgIGNhc2UgXCJbb2JqZWN0IEludDMyQXJyYXldXCI6XG4gICAgICAgIHJldHVybiBcImludDMyXCJcbiAgICAgIGNhc2UgXCJbb2JqZWN0IFVpbnQ4QXJyYXldXCI6XG4gICAgICAgIHJldHVybiBcInVpbnQ4XCJcbiAgICAgIGNhc2UgXCJbb2JqZWN0IFVpbnQxNkFycmF5XVwiOlxuICAgICAgICByZXR1cm4gXCJ1aW50MTZcIlxuICAgICAgY2FzZSBcIltvYmplY3QgVWludDMyQXJyYXldXCI6XG4gICAgICAgIHJldHVybiBcInVpbnQzMlwiXG4gICAgICBjYXNlIFwiW29iamVjdCBVaW50OENsYW1wZWRBcnJheV1cIjpcbiAgICAgICAgcmV0dXJuIFwidWludDhfY2xhbXBlZFwiXG4gICAgfVxuICB9XG4gIGlmKEFycmF5LmlzQXJyYXkoZGF0YSkpIHtcbiAgICByZXR1cm4gXCJhcnJheVwiXG4gIH1cbiAgcmV0dXJuIFwiZ2VuZXJpY1wiXG59XG5cbnZhciBDQUNIRURfQ09OU1RSVUNUT1JTID0ge1xuICBcImZsb2F0MzJcIjpbXSxcbiAgXCJmbG9hdDY0XCI6W10sXG4gIFwiaW50OFwiOltdLFxuICBcImludDE2XCI6W10sXG4gIFwiaW50MzJcIjpbXSxcbiAgXCJ1aW50OFwiOltdLFxuICBcInVpbnQxNlwiOltdLFxuICBcInVpbnQzMlwiOltdLFxuICBcImFycmF5XCI6W10sXG4gIFwidWludDhfY2xhbXBlZFwiOltdLFxuICBcImJ1ZmZlclwiOltdLFxuICBcImdlbmVyaWNcIjpbXVxufVxuXG47KGZ1bmN0aW9uKCkge1xuICBmb3IodmFyIGlkIGluIENBQ0hFRF9DT05TVFJVQ1RPUlMpIHtcbiAgICBDQUNIRURfQ09OU1RSVUNUT1JTW2lkXS5wdXNoKGNvbXBpbGVDb25zdHJ1Y3RvcihpZCwgLTEpKVxuICB9XG59KTtcblxuZnVuY3Rpb24gd3JhcHBlZE5EQXJyYXlDdG9yKGRhdGEsIHNoYXBlLCBzdHJpZGUsIG9mZnNldCkge1xuICBpZihkYXRhID09PSB1bmRlZmluZWQpIHtcbiAgICB2YXIgY3RvciA9IENBQ0hFRF9DT05TVFJVQ1RPUlMuYXJyYXlbMF1cbiAgICByZXR1cm4gY3RvcihbXSlcbiAgfSBlbHNlIGlmKHR5cGVvZiBkYXRhID09PSBcIm51bWJlclwiKSB7XG4gICAgZGF0YSA9IFtkYXRhXVxuICB9XG4gIGlmKHNoYXBlID09PSB1bmRlZmluZWQpIHtcbiAgICBzaGFwZSA9IFsgZGF0YS5sZW5ndGggXVxuICB9XG4gIHZhciBkID0gc2hhcGUubGVuZ3RoXG4gIGlmKHN0cmlkZSA9PT0gdW5kZWZpbmVkKSB7XG4gICAgc3RyaWRlID0gbmV3IEFycmF5KGQpXG4gICAgZm9yKHZhciBpPWQtMSwgc3o9MTsgaT49MDsgLS1pKSB7XG4gICAgICBzdHJpZGVbaV0gPSBzelxuICAgICAgc3ogKj0gc2hhcGVbaV1cbiAgICB9XG4gIH1cbiAgaWYob2Zmc2V0ID09PSB1bmRlZmluZWQpIHtcbiAgICBvZmZzZXQgPSAwXG4gICAgZm9yKHZhciBpPTA7IGk8ZDsgKytpKSB7XG4gICAgICBpZihzdHJpZGVbaV0gPCAwKSB7XG4gICAgICAgIG9mZnNldCAtPSAoc2hhcGVbaV0tMSkqc3RyaWRlW2ldXG4gICAgICB9XG4gICAgfVxuICB9XG4gIHZhciBkdHlwZSA9IGFycmF5RFR5cGUoZGF0YSlcbiAgdmFyIGN0b3JfbGlzdCA9IENBQ0hFRF9DT05TVFJVQ1RPUlNbZHR5cGVdXG4gIHdoaWxlKGN0b3JfbGlzdC5sZW5ndGggPD0gZCsxKSB7XG4gICAgY3Rvcl9saXN0LnB1c2goY29tcGlsZUNvbnN0cnVjdG9yKGR0eXBlLCBjdG9yX2xpc3QubGVuZ3RoLTEpKVxuICB9XG4gIHZhciBjdG9yID0gY3Rvcl9saXN0W2QrMV1cbiAgcmV0dXJuIGN0b3IoZGF0YSwgc2hhcGUsIHN0cmlkZSwgb2Zmc2V0KVxufVxuXG5tb2R1bGUuZXhwb3J0cyA9IHdyYXBwZWROREFycmF5Q3RvclxuIiwiLyoqXG5zcHJpbnRmKCkgZm9yIEphdmFTY3JpcHQgMC43LWJldGExXG5odHRwOi8vd3d3LmRpdmVpbnRvamF2YXNjcmlwdC5jb20vcHJvamVjdHMvamF2YXNjcmlwdC1zcHJpbnRmXG5cbkNvcHlyaWdodCAoYykgQWxleGFuZHJ1IE1hcmFzdGVhbnUgPGFsZXhhaG9saWMgW2F0KSBnbWFpbCAoZG90XSBjb20+XG5BbGwgcmlnaHRzIHJlc2VydmVkLlxuXG5SZWRpc3RyaWJ1dGlvbiBhbmQgdXNlIGluIHNvdXJjZSBhbmQgYmluYXJ5IGZvcm1zLCB3aXRoIG9yIHdpdGhvdXRcbm1vZGlmaWNhdGlvbiwgYXJlIHBlcm1pdHRlZCBwcm92aWRlZCB0aGF0IHRoZSBmb2xsb3dpbmcgY29uZGl0aW9ucyBhcmUgbWV0OlxuICAgICogUmVkaXN0cmlidXRpb25zIG9mIHNvdXJjZSBjb2RlIG11c3QgcmV0YWluIHRoZSBhYm92ZSBjb3B5cmlnaHRcbiAgICAgIG5vdGljZSwgdGhpcyBsaXN0IG9mIGNvbmRpdGlvbnMgYW5kIHRoZSBmb2xsb3dpbmcgZGlzY2xhaW1lci5cbiAgICAqIFJlZGlzdHJpYnV0aW9ucyBpbiBiaW5hcnkgZm9ybSBtdXN0IHJlcHJvZHVjZSB0aGUgYWJvdmUgY29weXJpZ2h0XG4gICAgICBub3RpY2UsIHRoaXMgbGlzdCBvZiBjb25kaXRpb25zIGFuZCB0aGUgZm9sbG93aW5nIGRpc2NsYWltZXIgaW4gdGhlXG4gICAgICBkb2N1bWVudGF0aW9uIGFuZC9vciBvdGhlciBtYXRlcmlhbHMgcHJvdmlkZWQgd2l0aCB0aGUgZGlzdHJpYnV0aW9uLlxuICAgICogTmVpdGhlciB0aGUgbmFtZSBvZiBzcHJpbnRmKCkgZm9yIEphdmFTY3JpcHQgbm9yIHRoZVxuICAgICAgbmFtZXMgb2YgaXRzIGNvbnRyaWJ1dG9ycyBtYXkgYmUgdXNlZCB0byBlbmRvcnNlIG9yIHByb21vdGUgcHJvZHVjdHNcbiAgICAgIGRlcml2ZWQgZnJvbSB0aGlzIHNvZnR3YXJlIHdpdGhvdXQgc3BlY2lmaWMgcHJpb3Igd3JpdHRlbiBwZXJtaXNzaW9uLlxuXG5USElTIFNPRlRXQVJFIElTIFBST1ZJREVEIEJZIFRIRSBDT1BZUklHSFQgSE9MREVSUyBBTkQgQ09OVFJJQlVUT1JTIFwiQVMgSVNcIiBBTkRcbkFOWSBFWFBSRVNTIE9SIElNUExJRUQgV0FSUkFOVElFUywgSU5DTFVESU5HLCBCVVQgTk9UIExJTUlURUQgVE8sIFRIRSBJTVBMSUVEXG5XQVJSQU5USUVTIE9GIE1FUkNIQU5UQUJJTElUWSBBTkQgRklUTkVTUyBGT1IgQSBQQVJUSUNVTEFSIFBVUlBPU0UgQVJFXG5ESVNDTEFJTUVELiBJTiBOTyBFVkVOVCBTSEFMTCBBbGV4YW5kcnUgTWFyYXN0ZWFudSBCRSBMSUFCTEUgRk9SIEFOWVxuRElSRUNULCBJTkRJUkVDVCwgSU5DSURFTlRBTCwgU1BFQ0lBTCwgRVhFTVBMQVJZLCBPUiBDT05TRVFVRU5USUFMIERBTUFHRVNcbihJTkNMVURJTkcsIEJVVCBOT1QgTElNSVRFRCBUTywgUFJPQ1VSRU1FTlQgT0YgU1VCU1RJVFVURSBHT09EUyBPUiBTRVJWSUNFUztcbkxPU1MgT0YgVVNFLCBEQVRBLCBPUiBQUk9GSVRTOyBPUiBCVVNJTkVTUyBJTlRFUlJVUFRJT04pIEhPV0VWRVIgQ0FVU0VEIEFORFxuT04gQU5ZIFRIRU9SWSBPRiBMSUFCSUxJVFksIFdIRVRIRVIgSU4gQ09OVFJBQ1QsIFNUUklDVCBMSUFCSUxJVFksIE9SIFRPUlRcbihJTkNMVURJTkcgTkVHTElHRU5DRSBPUiBPVEhFUldJU0UpIEFSSVNJTkcgSU4gQU5ZIFdBWSBPVVQgT0YgVEhFIFVTRSBPRiBUSElTXG5TT0ZUV0FSRSwgRVZFTiBJRiBBRFZJU0VEIE9GIFRIRSBQT1NTSUJJTElUWSBPRiBTVUNIIERBTUFHRS5cblxuXG5DaGFuZ2Vsb2c6XG4yMDEwLjExLjA3IC0gMC43LWJldGExLW5vZGVcbiAgLSBjb252ZXJ0ZWQgaXQgdG8gYSBub2RlLmpzIGNvbXBhdGlibGUgbW9kdWxlXG5cbjIwMTAuMDkuMDYgLSAwLjctYmV0YTFcbiAgLSBmZWF0dXJlczogdnNwcmludGYsIHN1cHBvcnQgZm9yIG5hbWVkIHBsYWNlaG9sZGVyc1xuICAtIGVuaGFuY2VtZW50czogZm9ybWF0IGNhY2hlLCByZWR1Y2VkIGdsb2JhbCBuYW1lc3BhY2UgcG9sbHV0aW9uXG5cbjIwMTAuMDUuMjIgLSAwLjY6XG4gLSByZXZlcnRlZCB0byAwLjQgYW5kIGZpeGVkIHRoZSBidWcgcmVnYXJkaW5nIHRoZSBzaWduIG9mIHRoZSBudW1iZXIgMFxuIE5vdGU6XG4gVGhhbmtzIHRvIFJhcGhhZWwgUGlndWxsYSA8cmFwaCAoYXRdIG4zcmQgW2RvdCkgb3JnPiAoaHR0cDovL3d3dy5uM3JkLm9yZy8pXG4gd2hvIHdhcm5lZCBtZSBhYm91dCBhIGJ1ZyBpbiAwLjUsIEkgZGlzY292ZXJlZCB0aGF0IHRoZSBsYXN0IHVwZGF0ZSB3YXNcbiBhIHJlZ3Jlc3MuIEkgYXBwb2xvZ2l6ZSBmb3IgdGhhdC5cblxuMjAxMC4wNS4wOSAtIDAuNTpcbiAtIGJ1ZyBmaXg6IDAgaXMgbm93IHByZWNlZWRlZCB3aXRoIGEgKyBzaWduXG4gLSBidWcgZml4OiB0aGUgc2lnbiB3YXMgbm90IGF0IHRoZSByaWdodCBwb3NpdGlvbiBvbiBwYWRkZWQgcmVzdWx0cyAoS2FtYWwgQWJkYWxpKVxuIC0gc3dpdGNoZWQgZnJvbSBHUEwgdG8gQlNEIGxpY2Vuc2VcblxuMjAwNy4xMC4yMSAtIDAuNDpcbiAtIHVuaXQgdGVzdCBhbmQgcGF0Y2ggKERhdmlkIEJhaXJkKVxuXG4yMDA3LjA5LjE3IC0gMC4zOlxuIC0gYnVnIGZpeDogbm8gbG9uZ2VyIHRocm93cyBleGNlcHRpb24gb24gZW1wdHkgcGFyYW1lbnRlcnMgKEhhbnMgUHVmYWwpXG5cbjIwMDcuMDkuMTEgLSAwLjI6XG4gLSBmZWF0dXJlOiBhZGRlZCBhcmd1bWVudCBzd2FwcGluZ1xuXG4yMDA3LjA0LjAzIC0gMC4xOlxuIC0gaW5pdGlhbCByZWxlYXNlXG4qKi9cblxudmFyIHNwcmludGYgPSAoZnVuY3Rpb24oKSB7XG5cdGZ1bmN0aW9uIGdldF90eXBlKHZhcmlhYmxlKSB7XG5cdFx0cmV0dXJuIE9iamVjdC5wcm90b3R5cGUudG9TdHJpbmcuY2FsbCh2YXJpYWJsZSkuc2xpY2UoOCwgLTEpLnRvTG93ZXJDYXNlKCk7XG5cdH1cblx0ZnVuY3Rpb24gc3RyX3JlcGVhdChpbnB1dCwgbXVsdGlwbGllcikge1xuXHRcdGZvciAodmFyIG91dHB1dCA9IFtdOyBtdWx0aXBsaWVyID4gMDsgb3V0cHV0Wy0tbXVsdGlwbGllcl0gPSBpbnB1dCkgey8qIGRvIG5vdGhpbmcgKi99XG5cdFx0cmV0dXJuIG91dHB1dC5qb2luKCcnKTtcblx0fVxuXG5cdHZhciBzdHJfZm9ybWF0ID0gZnVuY3Rpb24oKSB7XG5cdFx0aWYgKCFzdHJfZm9ybWF0LmNhY2hlLmhhc093blByb3BlcnR5KGFyZ3VtZW50c1swXSkpIHtcblx0XHRcdHN0cl9mb3JtYXQuY2FjaGVbYXJndW1lbnRzWzBdXSA9IHN0cl9mb3JtYXQucGFyc2UoYXJndW1lbnRzWzBdKTtcblx0XHR9XG5cdFx0cmV0dXJuIHN0cl9mb3JtYXQuZm9ybWF0LmNhbGwobnVsbCwgc3RyX2Zvcm1hdC5jYWNoZVthcmd1bWVudHNbMF1dLCBhcmd1bWVudHMpO1xuXHR9O1xuXG5cdC8vIGNvbnZlcnQgb2JqZWN0IHRvIHNpbXBsZSBvbmUgbGluZSBzdHJpbmcgd2l0aG91dCBpbmRlbnRhdGlvbiBvclxuXHQvLyBuZXdsaW5lcy4gTm90ZSB0aGF0IHRoaXMgaW1wbGVtZW50YXRpb24gZG9lcyBub3QgcHJpbnQgYXJyYXlcblx0Ly8gdmFsdWVzIHRvIHRoZWlyIGFjdHVhbCBwbGFjZSBmb3Igc3BhcnNlIGFycmF5cy4gXG5cdC8vXG5cdC8vIEZvciBleGFtcGxlIHNwYXJzZSBhcnJheSBsaWtlIHRoaXNcblx0Ly8gICAgbCA9IFtdXG5cdC8vICAgIGxbNF0gPSAxXG5cdC8vIFdvdWxkIGJlIHByaW50ZWQgYXMgXCJbMV1cIiBpbnN0ZWFkIG9mIFwiWywgLCAsICwgMV1cIlxuXHQvLyBcblx0Ly8gSWYgYXJndW1lbnQgJ3NlZW4nIGlzIG5vdCBudWxsIGFuZCBhcnJheSB0aGUgZnVuY3Rpb24gd2lsbCBjaGVjayBmb3IgXG5cdC8vIGNpcmN1bGFyIG9iamVjdCByZWZlcmVuY2VzIGZyb20gYXJndW1lbnQuXG5cdHN0cl9mb3JtYXQub2JqZWN0X3N0cmluZ2lmeSA9IGZ1bmN0aW9uKG9iaiwgZGVwdGgsIG1heGRlcHRoLCBzZWVuKSB7XG5cdFx0dmFyIHN0ciA9ICcnO1xuXHRcdGlmIChvYmogIT0gbnVsbCkge1xuXHRcdFx0c3dpdGNoKCB0eXBlb2Yob2JqKSApIHtcblx0XHRcdGNhc2UgJ2Z1bmN0aW9uJzogXG5cdFx0XHRcdHJldHVybiAnW0Z1bmN0aW9uJyArIChvYmoubmFtZSA/ICc6ICcrb2JqLm5hbWUgOiAnJykgKyAnXSc7XG5cdFx0XHQgICAgYnJlYWs7XG5cdFx0XHRjYXNlICdvYmplY3QnOlxuXHRcdFx0XHRpZiAoIG9iaiBpbnN0YW5jZW9mIEVycm9yKSB7IHJldHVybiAnWycgKyBvYmoudG9TdHJpbmcoKSArICddJyB9O1xuXHRcdFx0XHRpZiAoZGVwdGggPj0gbWF4ZGVwdGgpIHJldHVybiAnW09iamVjdF0nXG5cdFx0XHRcdGlmIChzZWVuKSB7XG5cdFx0XHRcdFx0Ly8gYWRkIG9iamVjdCB0byBzZWVuIGxpc3Rcblx0XHRcdFx0XHRzZWVuID0gc2Vlbi5zbGljZSgwKVxuXHRcdFx0XHRcdHNlZW4ucHVzaChvYmopO1xuXHRcdFx0XHR9XG5cdFx0XHRcdGlmIChvYmoubGVuZ3RoICE9IG51bGwpIHsgLy9hcnJheVxuXHRcdFx0XHRcdHN0ciArPSAnWyc7XG5cdFx0XHRcdFx0dmFyIGFyciA9IFtdXG5cdFx0XHRcdFx0Zm9yICh2YXIgaSBpbiBvYmopIHtcblx0XHRcdFx0XHRcdGlmIChzZWVuICYmIHNlZW4uaW5kZXhPZihvYmpbaV0pID49IDApIGFyci5wdXNoKCdbQ2lyY3VsYXJdJyk7XG5cdFx0XHRcdFx0XHRlbHNlIGFyci5wdXNoKHN0cl9mb3JtYXQub2JqZWN0X3N0cmluZ2lmeShvYmpbaV0sIGRlcHRoKzEsIG1heGRlcHRoLCBzZWVuKSk7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHRcdHN0ciArPSBhcnIuam9pbignLCAnKSArICddJztcblx0XHRcdFx0fSBlbHNlIGlmICgnZ2V0TW9udGgnIGluIG9iaikgeyAvLyBkYXRlXG5cdFx0XHRcdFx0cmV0dXJuICdEYXRlKCcgKyBvYmogKyAnKSc7XG5cdFx0XHRcdH0gZWxzZSB7IC8vIG9iamVjdFxuXHRcdFx0XHRcdHN0ciArPSAneyc7XG5cdFx0XHRcdFx0dmFyIGFyciA9IFtdXG5cdFx0XHRcdFx0Zm9yICh2YXIgayBpbiBvYmopIHsgXG5cdFx0XHRcdFx0XHRpZihvYmouaGFzT3duUHJvcGVydHkoaykpIHtcblx0XHRcdFx0XHRcdFx0aWYgKHNlZW4gJiYgc2Vlbi5pbmRleE9mKG9ialtrXSkgPj0gMCkgYXJyLnB1c2goayArICc6IFtDaXJjdWxhcl0nKTtcblx0XHRcdFx0XHRcdFx0ZWxzZSBhcnIucHVzaChrICsnOiAnICtzdHJfZm9ybWF0Lm9iamVjdF9zdHJpbmdpZnkob2JqW2tdLCBkZXB0aCsxLCBtYXhkZXB0aCwgc2VlbikpOyBcblx0XHRcdFx0XHRcdH1cblx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0c3RyICs9IGFyci5qb2luKCcsICcpICsgJ30nO1xuXHRcdFx0XHR9XG5cdFx0XHRcdHJldHVybiBzdHI7XG5cdFx0XHRcdGJyZWFrO1xuXHRcdFx0Y2FzZSAnc3RyaW5nJzpcdFx0XHRcdFxuXHRcdFx0XHRyZXR1cm4gJ1wiJyArIG9iaiArICdcIic7XG5cdFx0XHRcdGJyZWFrXG5cdFx0XHR9XG5cdFx0fVxuXHRcdHJldHVybiAnJyArIG9iajtcblx0fVxuXG5cdHN0cl9mb3JtYXQuZm9ybWF0ID0gZnVuY3Rpb24ocGFyc2VfdHJlZSwgYXJndikge1xuXHRcdHZhciBjdXJzb3IgPSAxLCB0cmVlX2xlbmd0aCA9IHBhcnNlX3RyZWUubGVuZ3RoLCBub2RlX3R5cGUgPSAnJywgYXJnLCBvdXRwdXQgPSBbXSwgaSwgaywgbWF0Y2gsIHBhZCwgcGFkX2NoYXJhY3RlciwgcGFkX2xlbmd0aDtcblx0XHRmb3IgKGkgPSAwOyBpIDwgdHJlZV9sZW5ndGg7IGkrKykge1xuXHRcdFx0bm9kZV90eXBlID0gZ2V0X3R5cGUocGFyc2VfdHJlZVtpXSk7XG5cdFx0XHRpZiAobm9kZV90eXBlID09PSAnc3RyaW5nJykge1xuXHRcdFx0XHRvdXRwdXQucHVzaChwYXJzZV90cmVlW2ldKTtcblx0XHRcdH1cblx0XHRcdGVsc2UgaWYgKG5vZGVfdHlwZSA9PT0gJ2FycmF5Jykge1xuXHRcdFx0XHRtYXRjaCA9IHBhcnNlX3RyZWVbaV07IC8vIGNvbnZlbmllbmNlIHB1cnBvc2VzIG9ubHlcblx0XHRcdFx0aWYgKG1hdGNoWzJdKSB7IC8vIGtleXdvcmQgYXJndW1lbnRcblx0XHRcdFx0XHRhcmcgPSBhcmd2W2N1cnNvcl07XG5cdFx0XHRcdFx0Zm9yIChrID0gMDsgayA8IG1hdGNoWzJdLmxlbmd0aDsgaysrKSB7XG5cdFx0XHRcdFx0XHRpZiAoIWFyZy5oYXNPd25Qcm9wZXJ0eShtYXRjaFsyXVtrXSkpIHtcblx0XHRcdFx0XHRcdFx0dGhyb3cgbmV3IEVycm9yKHNwcmludGYoJ1tzcHJpbnRmXSBwcm9wZXJ0eSBcIiVzXCIgZG9lcyBub3QgZXhpc3QnLCBtYXRjaFsyXVtrXSkpO1xuXHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdFx0YXJnID0gYXJnW21hdGNoWzJdW2tdXTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblx0XHRcdFx0ZWxzZSBpZiAobWF0Y2hbMV0pIHsgLy8gcG9zaXRpb25hbCBhcmd1bWVudCAoZXhwbGljaXQpXG5cdFx0XHRcdFx0YXJnID0gYXJndlttYXRjaFsxXV07XG5cdFx0XHRcdH1cblx0XHRcdFx0ZWxzZSB7IC8vIHBvc2l0aW9uYWwgYXJndW1lbnQgKGltcGxpY2l0KVxuXHRcdFx0XHRcdGFyZyA9IGFyZ3ZbY3Vyc29yKytdO1xuXHRcdFx0XHR9XG5cblx0XHRcdFx0aWYgKC9bXnNPXS8udGVzdChtYXRjaFs4XSkgJiYgKGdldF90eXBlKGFyZykgIT0gJ251bWJlcicpKSB7XG5cdFx0XHRcdFx0dGhyb3cgbmV3IEVycm9yKHNwcmludGYoJ1tzcHJpbnRmXSBleHBlY3RpbmcgbnVtYmVyIGJ1dCBmb3VuZCAlcyBcIicgKyBhcmcgKyAnXCInLCBnZXRfdHlwZShhcmcpKSk7XG5cdFx0XHRcdH1cblx0XHRcdFx0c3dpdGNoIChtYXRjaFs4XSkge1xuXHRcdFx0XHRcdGNhc2UgJ2InOiBhcmcgPSBhcmcudG9TdHJpbmcoMik7IGJyZWFrO1xuXHRcdFx0XHRcdGNhc2UgJ2MnOiBhcmcgPSBTdHJpbmcuZnJvbUNoYXJDb2RlKGFyZyk7IGJyZWFrO1xuXHRcdFx0XHRcdGNhc2UgJ2QnOiBhcmcgPSBwYXJzZUludChhcmcsIDEwKTsgYnJlYWs7XG5cdFx0XHRcdFx0Y2FzZSAnZSc6IGFyZyA9IG1hdGNoWzddID8gYXJnLnRvRXhwb25lbnRpYWwobWF0Y2hbN10pIDogYXJnLnRvRXhwb25lbnRpYWwoKTsgYnJlYWs7XG5cdFx0XHRcdFx0Y2FzZSAnZic6IGFyZyA9IG1hdGNoWzddID8gcGFyc2VGbG9hdChhcmcpLnRvRml4ZWQobWF0Y2hbN10pIDogcGFyc2VGbG9hdChhcmcpOyBicmVhaztcblx0XHRcdFx0ICAgIGNhc2UgJ08nOiBhcmcgPSBzdHJfZm9ybWF0Lm9iamVjdF9zdHJpbmdpZnkoYXJnLCAwLCBwYXJzZUludChtYXRjaFs3XSkgfHwgNSk7IGJyZWFrO1xuXHRcdFx0XHRcdGNhc2UgJ28nOiBhcmcgPSBhcmcudG9TdHJpbmcoOCk7IGJyZWFrO1xuXHRcdFx0XHRcdGNhc2UgJ3MnOiBhcmcgPSAoKGFyZyA9IFN0cmluZyhhcmcpKSAmJiBtYXRjaFs3XSA/IGFyZy5zdWJzdHJpbmcoMCwgbWF0Y2hbN10pIDogYXJnKTsgYnJlYWs7XG5cdFx0XHRcdFx0Y2FzZSAndSc6IGFyZyA9IE1hdGguYWJzKGFyZyk7IGJyZWFrO1xuXHRcdFx0XHRcdGNhc2UgJ3gnOiBhcmcgPSBhcmcudG9TdHJpbmcoMTYpOyBicmVhaztcblx0XHRcdFx0XHRjYXNlICdYJzogYXJnID0gYXJnLnRvU3RyaW5nKDE2KS50b1VwcGVyQ2FzZSgpOyBicmVhaztcblx0XHRcdFx0fVxuXHRcdFx0XHRhcmcgPSAoL1tkZWZdLy50ZXN0KG1hdGNoWzhdKSAmJiBtYXRjaFszXSAmJiBhcmcgPj0gMCA/ICcrJysgYXJnIDogYXJnKTtcblx0XHRcdFx0cGFkX2NoYXJhY3RlciA9IG1hdGNoWzRdID8gbWF0Y2hbNF0gPT0gJzAnID8gJzAnIDogbWF0Y2hbNF0uY2hhckF0KDEpIDogJyAnO1xuXHRcdFx0XHRwYWRfbGVuZ3RoID0gbWF0Y2hbNl0gLSBTdHJpbmcoYXJnKS5sZW5ndGg7XG5cdFx0XHRcdHBhZCA9IG1hdGNoWzZdID8gc3RyX3JlcGVhdChwYWRfY2hhcmFjdGVyLCBwYWRfbGVuZ3RoKSA6ICcnO1xuXHRcdFx0XHRvdXRwdXQucHVzaChtYXRjaFs1XSA/IGFyZyArIHBhZCA6IHBhZCArIGFyZyk7XG5cdFx0XHR9XG5cdFx0fVxuXHRcdHJldHVybiBvdXRwdXQuam9pbignJyk7XG5cdH07XG5cblx0c3RyX2Zvcm1hdC5jYWNoZSA9IHt9O1xuXG5cdHN0cl9mb3JtYXQucGFyc2UgPSBmdW5jdGlvbihmbXQpIHtcblx0XHR2YXIgX2ZtdCA9IGZtdCwgbWF0Y2ggPSBbXSwgcGFyc2VfdHJlZSA9IFtdLCBhcmdfbmFtZXMgPSAwO1xuXHRcdHdoaWxlIChfZm10KSB7XG5cdFx0XHRpZiAoKG1hdGNoID0gL15bXlxceDI1XSsvLmV4ZWMoX2ZtdCkpICE9PSBudWxsKSB7XG5cdFx0XHRcdHBhcnNlX3RyZWUucHVzaChtYXRjaFswXSk7XG5cdFx0XHR9XG5cdFx0XHRlbHNlIGlmICgobWF0Y2ggPSAvXlxceDI1ezJ9Ly5leGVjKF9mbXQpKSAhPT0gbnVsbCkge1xuXHRcdFx0XHRwYXJzZV90cmVlLnB1c2goJyUnKTtcblx0XHRcdH1cblx0XHRcdGVsc2UgaWYgKChtYXRjaCA9IC9eXFx4MjUoPzooWzEtOV1cXGQqKVxcJHxcXCgoW15cXCldKylcXCkpPyhcXCspPygwfCdbXiRdKT8oLSk/KFxcZCspPyg/OlxcLihcXGQrKSk/KFtiLWZvc091eFhdKS8uZXhlYyhfZm10KSkgIT09IG51bGwpIHtcblx0XHRcdFx0aWYgKG1hdGNoWzJdKSB7XG5cdFx0XHRcdFx0YXJnX25hbWVzIHw9IDE7XG5cdFx0XHRcdFx0dmFyIGZpZWxkX2xpc3QgPSBbXSwgcmVwbGFjZW1lbnRfZmllbGQgPSBtYXRjaFsyXSwgZmllbGRfbWF0Y2ggPSBbXTtcblx0XHRcdFx0XHRpZiAoKGZpZWxkX21hdGNoID0gL14oW2Etel9dW2Etel9cXGRdKikvaS5leGVjKHJlcGxhY2VtZW50X2ZpZWxkKSkgIT09IG51bGwpIHtcblx0XHRcdFx0XHRcdGZpZWxkX2xpc3QucHVzaChmaWVsZF9tYXRjaFsxXSk7XG5cdFx0XHRcdFx0XHR3aGlsZSAoKHJlcGxhY2VtZW50X2ZpZWxkID0gcmVwbGFjZW1lbnRfZmllbGQuc3Vic3RyaW5nKGZpZWxkX21hdGNoWzBdLmxlbmd0aCkpICE9PSAnJykge1xuXHRcdFx0XHRcdFx0XHRpZiAoKGZpZWxkX21hdGNoID0gL15cXC4oW2Etel9dW2Etel9cXGRdKikvaS5leGVjKHJlcGxhY2VtZW50X2ZpZWxkKSkgIT09IG51bGwpIHtcblx0XHRcdFx0XHRcdFx0XHRmaWVsZF9saXN0LnB1c2goZmllbGRfbWF0Y2hbMV0pO1xuXHRcdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0XHRcdGVsc2UgaWYgKChmaWVsZF9tYXRjaCA9IC9eXFxbKFxcZCspXFxdLy5leGVjKHJlcGxhY2VtZW50X2ZpZWxkKSkgIT09IG51bGwpIHtcblx0XHRcdFx0XHRcdFx0XHRmaWVsZF9saXN0LnB1c2goZmllbGRfbWF0Y2hbMV0pO1xuXHRcdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0XHRcdGVsc2Uge1xuXHRcdFx0XHRcdFx0XHRcdHRocm93IG5ldyBFcnJvcignW3NwcmludGZdICcgKyByZXBsYWNlbWVudF9maWVsZCk7XG5cdFx0XHRcdFx0XHRcdH1cblx0XHRcdFx0XHRcdH1cblx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0ZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ1tzcHJpbnRmXSAnICsgcmVwbGFjZW1lbnRfZmllbGQpO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0XHRtYXRjaFsyXSA9IGZpZWxkX2xpc3Q7XG5cdFx0XHRcdH1cblx0XHRcdFx0ZWxzZSB7XG5cdFx0XHRcdFx0YXJnX25hbWVzIHw9IDI7XG5cdFx0XHRcdH1cblx0XHRcdFx0aWYgKGFyZ19uYW1lcyA9PT0gMykge1xuXHRcdFx0XHRcdHRocm93IG5ldyBFcnJvcignW3NwcmludGZdIG1peGluZyBwb3NpdGlvbmFsIGFuZCBuYW1lZCBwbGFjZWhvbGRlcnMgaXMgbm90ICh5ZXQpIHN1cHBvcnRlZCcpO1xuXHRcdFx0XHR9XG5cdFx0XHRcdHBhcnNlX3RyZWUucHVzaChtYXRjaCk7XG5cdFx0XHR9XG5cdFx0XHRlbHNlIHtcblx0XHRcdFx0dGhyb3cgbmV3IEVycm9yKCdbc3ByaW50Zl0gJyArIF9mbXQpO1xuXHRcdFx0fVxuXHRcdFx0X2ZtdCA9IF9mbXQuc3Vic3RyaW5nKG1hdGNoWzBdLmxlbmd0aCk7XG5cdFx0fVxuXHRcdHJldHVybiBwYXJzZV90cmVlO1xuXHR9O1xuXG5cdHJldHVybiBzdHJfZm9ybWF0O1xufSkoKTtcblxudmFyIHZzcHJpbnRmID0gZnVuY3Rpb24oZm10LCBhcmd2KSB7XG5cdHZhciBhcmd2Q2xvbmUgPSBhcmd2LnNsaWNlKCk7XG5cdGFyZ3ZDbG9uZS51bnNoaWZ0KGZtdCk7XG5cdHJldHVybiBzcHJpbnRmLmFwcGx5KG51bGwsIGFyZ3ZDbG9uZSk7XG59O1xuXG5tb2R1bGUuZXhwb3J0cyA9IHNwcmludGY7XG5zcHJpbnRmLnNwcmludGYgPSBzcHJpbnRmO1xuc3ByaW50Zi52c3ByaW50ZiA9IHZzcHJpbnRmO1xuIiwiaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xuXG5leHBvcnQgZGVmYXVsdCB7XG5cdGhhcmRfc2lnbW9pZDogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvaGFyZF9zaWdtb2lkLmdsc2wnLCAndXRmOCcpLFxuXHRsaW5lYXI6IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL2xpbmVhci5nbHNsJywgJ3V0ZjgnKSxcblx0cmVsdTogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvcmVsdS5nbHNsJywgJ3V0ZjgnKSxcblx0cmdiOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9yZ2IuZ2xzbCcsICd1dGY4JyksXG5cdHNpZ21vaWQ6IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3NpZ21vaWQuZ2xzbCcsICd1dGY4JyksXG5cdHRhbmg6IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3RhbmguZ2xzbCcsICd1dGY4JyksXG59IiwiaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xuXG5leHBvcnQgY29uc3QgZW5jb2RlU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvZW5jb2RlLmdsc2wnLCAndXRmOCcpO1xuZXhwb3J0IGNvbnN0IGRlY29kZVNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL2RlY29kZS5nbHNsJywgJ3V0ZjgnKTtcblxuZXhwb3J0IGZ1bmN0aW9uIGluaXQoc2hhcGUsIGZvcm1hdCl7XG5cdHJldHVybiB7XG5cdFx0cmFuZ2U6IGZvcm1hdC5yYW5nZSB8fCA0MDk2XG5cdH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGVuY29kZShidWYsIHZhbHVlLCBpbmZvKXtcblx0dmFyIHogPSBNYXRoLm1pbigxLCBNYXRoLm1heCgwLCB2YWx1ZSAvIGluZm8ucmFuZ2UgKyAwLjUpKTtcblx0YnVmWzBdID0gKHogKiAyNTYgKiAyNTYgKiAyNTYgKiAyNTYpICUgMjU2XG5cdGJ1ZlsxXSA9ICh6ICogMjU2ICogMjU2ICogMjU2KSAlIDI1NlxuXHRidWZbMl0gPSAoeiAqIDI1NiAqIDI1NikgJSAyNTZcblx0YnVmWzNdID0gKHogKiAyNTYpICUgMjU2XG59XG5cblxuZXhwb3J0IGZ1bmN0aW9uIGRlY29kZShidWYpe1xuXHRyZXR1cm4gYnVmWzBdIC8gMjU2LjAgLyAyNTYuMCAvIDI1Ni4wIC8gMjU2LjAgK1xuXHRcdCAgIGJ1ZlsxXSAvIDI1Ni4wIC8gMjU2LjAgLyAyNTYuMCArXG5cdFx0ICAgYnVmWzJdIC8gMjU2LjAgLyAyNTYuMCArXG5cdFx0ICAgYnVmWzNdIC8gMjU2LjA7XG59XG4iLCJpbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5cbmV4cG9ydCBjb25zdCBlbmNvZGVTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9lbmNvZGUuZ2xzbCcsICd1dGY4Jyk7XG5leHBvcnQgY29uc3QgZGVjb2RlU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvZGVjb2RlLmdsc2wnLCAndXRmOCcpO1xuXG5leHBvcnQgZnVuY3Rpb24gaW5pdChzaGFwZSwgZm9ybWF0KXtcblx0cmV0dXJuIHsgfVxufVxuXG52YXIgdG1wX2Zsb2F0ID0gbmV3IEZsb2F0MzJBcnJheSgxKSxcblx0dG1wX2ludCA9IG5ldyBVaW50OEFycmF5KHRtcF9mbG9hdC5idWZmZXIpO1xuXG5leHBvcnQgZnVuY3Rpb24gZW5jb2RlKGJ1ZiwgdmFsdWUpe1xuXHR0bXBfZmxvYXRbMF0gPSB2YWx1ZTtcblx0YnVmLnNldCh0bXBfaW50LCAwKVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZGVjb2RlKGJ1Zil7XG5cdHRtcF9pbnQuc2V0KGJ1Zilcblx0cmV0dXJuIHRtcF9mbG9hdFswXVxufSIsImltcG9ydCAqIGFzIHBhY2tfc3RyaWRlIGZyb20gJy4vcGFjay9zdHJpZGUvaW5kZXguanMnXG5pbXBvcnQgKiBhcyBwYWNrX3RpbGUgZnJvbSAnLi9wYWNrL3RpbGUvaW5kZXguanMnXG5cbmltcG9ydCAqIGFzIGNvZGVjX2ZpeG51bSBmcm9tICcuL2NvZGVjL2ZpeG51bS9pbmRleC5qcydcbmltcG9ydCAqIGFzIGNvZGVjX3NvZnRmbG9hdCBmcm9tICcuL2NvZGVjL3NvZnRmbG9hdC9pbmRleC5qcydcblxuaW1wb3J0IGFjdGl2YXRpb25zIGZyb20gJy4vYWN0aXZhdGlvbi9pbmRleC5qcydcblxuaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xuXG5leHBvcnQgZGVmYXVsdCB7XG5cdHBhY2s6IHtcblx0XHRzdHJpZGU6IHBhY2tfc3RyaWRlLFxuXHRcdHRpbGU6IHBhY2tfdGlsZVxuXHR9LFxuXG5cdHJlYWRfc2hpbTogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvcGFjay9yZWFkX3NoaW0uZ2xzbCcsICd1dGY4JyksXG5cdHdyaXRlX3NoaW06IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3BhY2svd3JpdGVfc2hpbS5nbHNsJywgJ3V0ZjgnKSxcblxuXHRjb2RlYzoge1xuXHRcdGZpeG51bTogY29kZWNfZml4bnVtLFxuXHRcdHNvZnRmbG9hdDogY29kZWNfc29mdGZsb2F0LFxuXHR9LFxuXHRhY3RpdmF0aW9uczogYWN0aXZhdGlvbnNcbn0iLCJpbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5pbXBvcnQgbmRhcnJheSBmcm9tICduZGFycmF5J1xuXG5leHBvcnQgY29uc3QgcmVhZFNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3JlYWQuZ2xzbCcsICd1dGY4Jyk7XG5leHBvcnQgY29uc3Qgd3JpdGVTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy93cml0ZS5nbHNsJywgJ3V0ZjgnKTtcblxuZXhwb3J0IGZ1bmN0aW9uIGluaXQoc2hhcGUpe1xuICAgIC8vIHZhciBsZW5ndGggPSA0ICogTWF0aC5jZWlsKHNoYXBlWzJdIC8gNCkgKiBzaGFwZVszXSAqIHNoYXBlWzFdICogc2hhcGVbMF07XG4gICAgLy8gdmFyIGNvbHMgPSBNYXRoLmNlaWwoTWF0aC5zcXJ0KGxlbmd0aCkgLyA0KSAqIDQ7XG5cbiAgICB2YXIgbGVuZ3RoID0gc2hhcGVbMl0gKiBzaGFwZVszXSAqIHNoYXBlWzFdICogc2hhcGVbMF07XG4gICAgdmFyIGNvbHMgPSBNYXRoLmNlaWwoTWF0aC5zcXJ0KGxlbmd0aCkpO1xuICAgIHZhciB0ZXhTaXplID0gW2NvbHMsIE1hdGguY2VpbChsZW5ndGggLyBjb2xzKV1cbiAgICByZXR1cm4ge1xuICAgICAgICB0ZXhTaXplOiB0ZXhTaXplLFxuICAgICAgICBzaGFwZTogc2hhcGUsXG4gICAgICAgIC8vIHZlYzQoMSwgQHNoYXBlLngsIEBzaGFwZS54ICogQHNoYXBlLnksIEBzaGFwZS54ICogQHNoYXBlLnkgKiBAc2hhcGUueilcbiAgICAgICAgc3RyaWRlOiBbMSwgc2hhcGVbMF0sIHNoYXBlWzBdICogc2hhcGVbMV0sIHNoYXBlWzBdICogc2hhcGVbMV0gKiBzaGFwZVsyXV1cbiAgICB9XG59XG5cblxuZXhwb3J0IGZ1bmN0aW9uIHBhY2soaW5mbywgYXJyYXksIGVuY29kZTEsIGZvcm1hdCl7XG4gICAgLy8gcmV0dXJuIFVpbnQ4QXJyYXkgb3IgRmxvYXQzMkFycmF5XG4gICAgYXJyYXkgPSBuZGFycmF5KGFycmF5LmRhdGEsIFxuICAgICAgICBhcnJheS5zaGFwZS5jb25jYXQoWzEsIDEsIDEsIDFdKS5zbGljZSgwLCA0KSxcbiAgICAgICAgYXJyYXkuc3RyaWRlLmNvbmNhdChbMSwgMSwgMSwgMV0pLnNsaWNlKDAsIDQpLFxuICAgICAgICBhcnJheS5vZmZzZXQpXG5cbiAgICB2YXIgc2hhcGUgPSBpbmZvLnNoYXBlO1xuICAgIHZhciBsZW5ndGggPSBpbmZvLnRleFNpemVbMF0gKiBpbmZvLnRleFNpemVbMV0gKiA0O1xuXG4gICAgaWYoZm9ybWF0LnR5cGUgPT09ICdmbG9hdDMyJyl7XG4gICAgICAgIHZhciBkYXRhID0gbmV3IEZsb2F0MzJBcnJheShsZW5ndGgpOyAgICBcbiAgICB9ZWxzZSBpZihmb3JtYXQudHlwZSA9PT0gJ3VpbnQ4Jyl7XG4gICAgICAgIHZhciBkYXRhID0gbmV3IFVpbnQ4QXJyYXkobGVuZ3RoKTsgICAgXG4gICAgfVxuXG4gICAgZm9yKHZhciB4ID0gMDsgeCA8IHNoYXBlWzBdOyB4Kyspe1xuICAgICAgICBmb3IodmFyIHkgPSAwOyB5IDwgc2hhcGVbMV07IHkrKyl7XG4gICAgICAgICAgICBmb3IodmFyIHogPSAwOyB6IDwgc2hhcGVbMl07IHorKyl7XG4gICAgICAgICAgICAgICAgZm9yKHZhciB3ID0gMDsgdyA8IHNoYXBlWzNdOyB3Kyspe1xuICAgICAgICAgICAgICAgICAgICB2YXIgdGlsZSAgPSB4ICsgXG4gICAgICAgICAgICAgICAgICAgICAgICB5ICogc2hhcGVbMF0gKyBcbiAgICAgICAgICAgICAgICAgICAgICAgIHogKiBzaGFwZVswXSAqIHNoYXBlWzFdICtcbiAgICAgICAgICAgICAgICAgICAgICAgIHcgKiBzaGFwZVswXSAqIHNoYXBlWzFdICogc2hhcGVbMl07XG5cbiAgICAgICAgICAgICAgICAgICAgZW5jb2RlMShkYXRhLnN1YmFycmF5KDQqdGlsZSwgNCp0aWxlKzQpLCBhcnJheS5nZXQoeCwgeSwgeiwgdyksIGluZm8pXG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIGRhdGE7XG59XG5cblxuZXhwb3J0IGZ1bmN0aW9uIHVucGFjayhpbmZvLCBkYXRhLCBkZWNvZGUxLCB0eXBlKXtcbiAgICBpZih0eXBlICE9ICdmbG9hdDMyJykgdGhyb3cgbmV3IEVycm9yKCdub3QgaW1wbCcpO1xuXG4gICAgdmFyIHNoYXBlID0gaW5mby5zaGFwZTtcbiAgICB2YXIgbGVuZ3RoID0gc2hhcGUucmVkdWNlKChhLCBiKSA9PiBhICogYik7XG5cbiAgICB2YXIgYXJyYXkgPSBuZGFycmF5KG5ldyBGbG9hdDMyQXJyYXkobGVuZ3RoKSwgXG4gICAgICAgIHNoYXBlLmNvbmNhdChbMSwgMSwgMSwgMV0pLnNsaWNlKDAsIDQpKVxuXG5cbiAgICBmb3IodmFyIHggPSAwOyB4IDwgc2hhcGVbMF07IHgrKyl7XG4gICAgICAgIGZvcih2YXIgeSA9IDA7IHkgPCBzaGFwZVsxXTsgeSsrKXtcbiAgICAgICAgICAgIGZvcih2YXIgeiA9IDA7IHogPCBzaGFwZVsyXTsgeisrKXtcbiAgICAgICAgICAgICAgICBmb3IodmFyIHcgPSAwOyB3IDwgc2hhcGVbM107IHcrKyl7XG4gICAgICAgICAgICAgICAgICAgIHZhciB0aWxlICA9IHggKyBcbiAgICAgICAgICAgICAgICAgICAgICAgIHkgKiBzaGFwZVswXSArIFxuICAgICAgICAgICAgICAgICAgICAgICAgeiAqIHNoYXBlWzBdICogc2hhcGVbMV0gK1xuICAgICAgICAgICAgICAgICAgICAgICAgdyAqIHNoYXBlWzBdICogc2hhcGVbMV0gKiBzaGFwZVsyXTtcblxuICAgICAgICAgICAgICAgICAgICBhcnJheS5zZXQoeCwgeSwgeiwgdywgZGVjb2RlMShkYXRhLnN1YmFycmF5KDQqdGlsZSwgNCp0aWxlKzQpLCBpbmZvKSlcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGFycmF5O1xufSIsImltcG9ydCB7IHJlYWRGaWxlU3luYyB9IGZyb20gJ2ZzJztcblxuZXhwb3J0IGNvbnN0IHJlYWRTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9yZWFkLmdsc2wnLCAndXRmOCcpO1xuZXhwb3J0IGNvbnN0IHdyaXRlU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvd3JpdGUuZ2xzbCcsICd1dGY4Jyk7XG5pbXBvcnQgbmRhcnJheSBmcm9tICduZGFycmF5J1xuXG5cbmV4cG9ydCBmdW5jdGlvbiBpbml0KHNoYXBlKXtcbiAgICB2YXIgd2lkdGggPSBzaGFwZVswXTtcbiAgICAvLyB3ZSBwaWNrIHRoZSBudW1iZXIgb2YgY29sdW1ucyBzbyB3ZSBjYW4ga2VlcFxuICAgIC8vIHRoZSB0ZXh0dXJlIGFzIHNxdWFyZSBhcyBwb3NzaWJsZSwgd2l0aCB0aGVcbiAgICAvLyBtaW5pbWFsIGFtb3VudCBvZiB3YXN0ZWQgc3BhY2UuXG5cbiAgICB2YXIgdGlsZXMgPSBzaGFwZVsyXSAqIHNoYXBlWzNdLFxuICAgICAgICBjb2xzID0gTWF0aC5tYXgoMSwgTWF0aC5taW4odGlsZXMsIE1hdGguY2VpbChcbiAgICAgICAgICAgIE1hdGguc3FydChzaGFwZVswXSAqIHNoYXBlWzFdICogdGlsZXMpIC8gd2lkdGgpKSk7XG5cbiAgICB2YXIgdGV4U2l6ZSA9IFt3aWR0aCAqIGNvbHMsIHNoYXBlWzFdICogTWF0aC5jZWlsKHRpbGVzIC8gY29scyldXG5cbiAgICByZXR1cm4ge1xuICAgICAgICB0ZXhTaXplOiB0ZXhTaXplLFxuICAgICAgICBjb2xzOiBjb2xzLFxuICAgICAgICBzaGFwZTogc2hhcGUsXG4gICAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gcGFjayhpbmZvLCBuZGFycmF5KXtcbiAgICAvLyByZXR1cm4gVWludDhBcnJheSBvciBGbG9hdDMyQXJyYXlcblxuXG4vLyB1bmlmb3JtIHNhbXBsZXIyRCBAX3RleDtcbi8vIHVuaWZvcm0gaXZlYzIgQF90ZXhTaXplO1xuLy8gdW5pZm9ybSBpdmVjNCBAX3NoYXBlO1xuLy8gdW5pZm9ybSBpbnQgQF9jb2xzO1xuXG4gICAgLy8gcmV0dXJuIHtcbiAgICAvLyAgdGV4OlxuICAgIC8vICB0ZXhTaXplOlxuICAgIC8vICBzaGFwZTpcbiAgICAvLyAgY29sczpcbiAgICAvLyB9XG4gICAgdGhyb3cgbmV3IEVycm9yKFwibm90IGltcGxlbWVudGVkOiBmb3JtYXQvMS00L3BhY2svdGlsZS9pbmRleC5qczpwYWNrXCIpXG59XG5cblxuZXhwb3J0IGZ1bmN0aW9uIHVucGFjayhpbmZvLCBhcnIpe1xuICAgIC8vIHJldHVybiBuZGFycmF5XG4gICAgdGhyb3cgbmV3IEVycm9yKFwibm90IGltcGxlbWVudGVkOiBmb3JtYXQvMS00L3BhY2svdGlsZS9pbmRleC5qczp1bnBhY2tcIilcbn0iLCJpbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5cbmV4cG9ydCBkZWZhdWx0IHtcblx0aGFyZF9zaWdtb2lkOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9oYXJkX3NpZ21vaWQuZ2xzbCcsICd1dGY4JyksXG5cdGxpbmVhcjogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvbGluZWFyLmdsc2wnLCAndXRmOCcpLFxuXHRyZWx1OiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9yZWx1Lmdsc2wnLCAndXRmOCcpLFxuXHRyZ2I6IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3JnYi5nbHNsJywgJ3V0ZjgnKSxcblx0c2lnbW9pZDogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvc2lnbW9pZC5nbHNsJywgJ3V0ZjgnKSxcblx0dGFuaDogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvdGFuaC5nbHNsJywgJ3V0ZjgnKSxcbn0iLCJpbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5cbmV4cG9ydCBjb25zdCBlbmNvZGVTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9lbmNvZGUuZ2xzbCcsICd1dGY4Jyk7XG5leHBvcnQgY29uc3QgZGVjb2RlU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvZGVjb2RlLmdsc2wnLCAndXRmOCcpO1xuXG5leHBvcnQgZnVuY3Rpb24gaW5pdChzaGFwZSwgZm9ybWF0KXtcblx0cmV0dXJuIHtcblx0XHRyYW5nZTogW1xuXHRcdFx0aXNGaW5pdGUoZm9ybWF0Lm1pbikgPyBmb3JtYXQubWluIDogMCxcblx0XHRcdGlzRmluaXRlKGZvcm1hdC5tYXgpID8gZm9ybWF0Lm1heCA6IDFcblx0XHRdXG5cdFx0Ly8gbWF4OiAsXG5cdFx0Ly8gbWluOiAsXG5cdH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGVuY29kZShkYXRhLCByLCBnLCBiLCBhLCBpbmZvKXtcblxuXHRkYXRhWzBdID0gTWF0aC5yb3VuZCgyNTUgKiBNYXRoLm1pbigxLCBNYXRoLm1heCgwLCAociAtIGluZm8ucmFuZ2VbMF0pLyhpbmZvLnJhbmdlWzFdIC0gaW5mby5yYW5nZVswXSkgKSkpXG5cdGRhdGFbMV0gPSBNYXRoLnJvdW5kKDI1NSAqIE1hdGgubWluKDEsIE1hdGgubWF4KDAsIChnIC0gaW5mby5yYW5nZVswXSkvKGluZm8ucmFuZ2VbMV0gLSBpbmZvLnJhbmdlWzBdKSApKSlcblx0ZGF0YVsyXSA9IE1hdGgucm91bmQoMjU1ICogTWF0aC5taW4oMSwgTWF0aC5tYXgoMCwgKGIgLSBpbmZvLnJhbmdlWzBdKS8oaW5mby5yYW5nZVsxXSAtIGluZm8ucmFuZ2VbMF0pICkpKVxuXHRkYXRhWzNdID0gTWF0aC5yb3VuZCgyNTUgKiBNYXRoLm1pbigxLCBNYXRoLm1heCgwLCAoYSAtIGluZm8ucmFuZ2VbMF0pLyhpbmZvLnJhbmdlWzFdIC0gaW5mby5yYW5nZVswXSkgKSkpXG5cdC8vIGNvbnNvbGUubG9nKGRhdGFbMF0sIGRhdGFbMV0sIGRhdGFbMl0pXG59XG5cblxuZXhwb3J0IGZ1bmN0aW9uIGRlY29kZShkYXRhLCByLCBnLCBiLCBhLCBpbmZvKXtcblx0ZGF0YVswXSA9IChyIC8gMjU1KSAqIChpbmZvLnJhbmdlWzFdIC0gaW5mby5yYW5nZVswXSkgKyBpbmZvLnJhbmdlWzBdO1xuXHRkYXRhWzFdID0gKGcgLyAyNTUpICogKGluZm8ucmFuZ2VbMV0gLSBpbmZvLnJhbmdlWzBdKSArIGluZm8ucmFuZ2VbMF07XG5cdGRhdGFbMl0gPSAoYiAvIDI1NSkgKiAoaW5mby5yYW5nZVsxXSAtIGluZm8ucmFuZ2VbMF0pICsgaW5mby5yYW5nZVswXTtcblx0ZGF0YVszXSA9IChhIC8gMjU1KSAqIChpbmZvLnJhbmdlWzFdIC0gaW5mby5yYW5nZVswXSkgKyBpbmZvLnJhbmdlWzBdO1xufSIsImltcG9ydCB7IHJlYWRGaWxlU3luYyB9IGZyb20gJ2ZzJztcblxuZXhwb3J0IGNvbnN0IGVuY29kZVNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL2VuY29kZS5nbHNsJywgJ3V0ZjgnKTtcbmV4cG9ydCBjb25zdCBkZWNvZGVTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9kZWNvZGUuZ2xzbCcsICd1dGY4Jyk7XG5cbmV4cG9ydCBmdW5jdGlvbiBpbml0KHNoYXBlLCBmb3JtYXQpe1xuXHRyZXR1cm4geyB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBlbmNvZGUoZGF0YSwgciwgZywgYiwgYSl7XG5cdGRhdGFbMF0gPSByO1xuXHRkYXRhWzFdID0gZztcblx0ZGF0YVsyXSA9IGI7XG5cdGRhdGFbM10gPSBhO1xufVxuXG5cbmV4cG9ydCBmdW5jdGlvbiBkZWNvZGUoZGF0YSwgciwgZywgYiwgYSl7XG5cdGRhdGFbMF0gPSByO1xuXHRkYXRhWzFdID0gZztcblx0ZGF0YVsyXSA9IGI7XG5cdGRhdGFbM10gPSBhO1xufSIsImltcG9ydCAqIGFzIHBhY2tfc3RyaWRlIGZyb20gJy4vcGFjay9zdHJpZGUvaW5kZXguanMnXG5pbXBvcnQgKiBhcyBwYWNrX3RpbGUgZnJvbSAnLi9wYWNrL3RpbGUvaW5kZXguanMnXG5cbmltcG9ydCAqIGFzIGNvZGVjX3JhdyBmcm9tICcuL2NvZGVjL3Jhdy9pbmRleC5qcydcbmltcG9ydCAqIGFzIGNvZGVjX2xpbnF1YW50IGZyb20gJy4vY29kZWMvbGlucXVhbnQvaW5kZXguanMnXG5cbmltcG9ydCBhY3RpdmF0aW9ucyBmcm9tICcuL2FjdGl2YXRpb24vaW5kZXguanMnXG5cbmltcG9ydCB7IHJlYWRGaWxlU3luYyB9IGZyb20gJ2ZzJztcblxuZXhwb3J0IGRlZmF1bHQge1xuXHRwYWNrOiB7XG5cdFx0c3RyaWRlOiBwYWNrX3N0cmlkZSxcblx0XHR0aWxlOiBwYWNrX3RpbGVcblx0fSxcblxuXG5cdHJlYWRfc2hpbTogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvcGFjay9yZWFkX3NoaW0uZ2xzbCcsICd1dGY4JyksXG5cdHdyaXRlX3NoaW06IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3BhY2svd3JpdGVfc2hpbS5nbHNsJywgJ3V0ZjgnKSxcblxuXHRjb2RlYzoge1xuXHRcdHJhdzogY29kZWNfcmF3LFxuXHRcdGxpbnF1YW50OiBjb2RlY19saW5xdWFudCxcblx0fSxcblx0YWN0aXZhdGlvbnM6IGFjdGl2YXRpb25zXG59IiwiaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xuXG5leHBvcnQgY29uc3QgcmVhZFNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3JlYWQuZ2xzbCcsICd1dGY4Jyk7XG5leHBvcnQgY29uc3Qgd3JpdGVTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy93cml0ZS5nbHNsJywgJ3V0ZjgnKTtcbmltcG9ydCBuZGFycmF5IGZyb20gJ25kYXJyYXknXG5cbmV4cG9ydCBmdW5jdGlvbiBpbml0KHNoYXBlKXtcbiAgICB2YXIgbGVuZ3RoID0gTWF0aC5jZWlsKHNoYXBlWzJdIC8gNCkgKiBzaGFwZVszXSAqIHNoYXBlWzFdICogc2hhcGVbMF07XG4gICAgdmFyIGNvbHMgPSBNYXRoLmNlaWwoTWF0aC5zcXJ0KGxlbmd0aCkpO1xuICAgIHZhciB0ZXhTaXplID0gW2NvbHMsIE1hdGguY2VpbChsZW5ndGggLyBjb2xzKV1cblxuICAgIGNvbnNvbGUuYXNzZXJ0KHRleFNpemVbMF0gKiB0ZXhTaXplWzFdID49IGxlbmd0aClcbiAgICByZXR1cm4ge1xuICAgICAgICB0ZXhTaXplOiB0ZXhTaXplLFxuICAgICAgICBzaGFwZTogc2hhcGUsXG5cbiAgICAgICAgc3RyaWRlOiBbXG4gICAgICAgICAgICAxLCBcbiAgICAgICAgICAgIHNoYXBlWzBdLCBcbiAgICAgICAgICAgIHNoYXBlWzBdICogc2hhcGVbMV0gLyA0LCAgLy8gdGhlIC80IGlzIGJlY2F1c2Ugb2YgdGhlIGNvbG9yIGNoYW5uZWxcbiAgICAgICAgICAgIHNoYXBlWzBdICogc2hhcGVbMV0gKiBNYXRoLmNlaWwoc2hhcGVbMl0gLyA0KVxuICAgICAgICBdLFxuICAgICAgICAvLyBkZWN2ZWM6IFsxLCBzaGFwZVswXSwgc2hhcGVbMF0gKiBzaGFwZVsxXSwgc2hhcGVbMF0gKiBzaGFwZVsxXSAqIE1hdGguY2VpbChzaGFwZVsyXSAvIDQpXVxuICAgIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHBhY2soaW5mbywgYXJyYXksIGVuY29kZTQsIGZvcm1hdCl7XG4gICAgLy8gcmV0dXJuIFVpbnQ4QXJyYXkgb3IgRmxvYXQzMkFycmF5XG5cbiAgICBhcnJheSA9IG5kYXJyYXkoYXJyYXkuZGF0YSwgXG4gICAgICAgIGFycmF5LnNoYXBlLmNvbmNhdChbMSwgMSwgMSwgMV0pLnNsaWNlKDAsIDQpLFxuICAgICAgICBhcnJheS5zdHJpZGUuY29uY2F0KFsxLCAxLCAxLCAxXSkuc2xpY2UoMCwgNCksXG4gICAgICAgIGFycmF5Lm9mZnNldClcbiAgICBcbiAgICB2YXIgW3dpZHRoLCBoZWlnaHRdID0gaW5mby50ZXhTaXplLFxuICAgICAgICBsZW5ndGggPSB3aWR0aCAqIGhlaWdodCAqIDQ7XG4gICAgdmFyIHNoYXBlID0gaW5mby5zaGFwZTtcblxuICAgIGlmKGZvcm1hdC50eXBlID09PSAnZmxvYXQzMicpe1xuICAgICAgICB2YXIgZGF0YSA9IG5ldyBGbG9hdDMyQXJyYXkobGVuZ3RoKTsgICAgXG4gICAgfWVsc2UgaWYoZm9ybWF0LnR5cGUgPT09ICd1aW50OCcpe1xuICAgICAgICB2YXIgZGF0YSA9IG5ldyBVaW50OEFycmF5KGxlbmd0aCk7ICAgIFxuICAgIH1cblxuICAgIHZhciBjaGFucyA9IE1hdGguY2VpbChpbmZvLnNoYXBlWzJdIC8gNCk7XG5cbiAgICBmb3IodmFyIGkgPSAwOyBpIDwgaW5mby5zaGFwZVswXTsgaSsrKXtcbiAgICAgICAgZm9yKHZhciBqID0gMDsgaiA8IGluZm8uc2hhcGVbMV07IGorKyl7XG4gICAgICAgICAgICBmb3IodmFyIGsgPSAwOyBrIDwgY2hhbnM7IGsrKyl7XG4gICAgICAgICAgICAgICAgdmFyIGIgPSBNYXRoLm1pbihrKjQrNCwgc2hhcGVbMl0pLWsqNDtcbiAgICAgICAgICAgICAgICBmb3IodmFyIHcgPSAwOyB3IDwgaW5mby5zaGFwZVszXTsgdysrKXtcblxuICAgICAgICAgICAgICAgICAgICB2YXIgdGlsZSAgPSBpICsgXG4gICAgICAgICAgICAgICAgICAgICAgICBqICogc2hhcGVbMF0gKyBcbiAgICAgICAgICAgICAgICAgICAgICAgIGsgKiBzaGFwZVswXSAqIHNoYXBlWzFdICtcbiAgICAgICAgICAgICAgICAgICAgICAgIHcgKiBzaGFwZVswXSAqIHNoYXBlWzFdICogY2hhbnM7XG5cblxuICAgICAgICAgICAgICAgICAgICB2YXIgcG9zID0gNCAqIHRpbGU7XG4gICAgICAgICAgICAgICAgICAgIGVuY29kZTQoXG4gICAgICAgICAgICAgICAgICAgICAgICBkYXRhLnN1YmFycmF5KHBvcywgcG9zICsgNCksXG4gICAgICAgICAgICAgICAgICAgICAgICBiIDwgMSA/IDAgOiBhcnJheS5nZXQoaSwgaiwgNCprKzAsIHcpLCBcbiAgICAgICAgICAgICAgICAgICAgICAgIGIgPCAyID8gMCA6IGFycmF5LmdldChpLCBqLCA0KmsrMSwgdyksIFxuICAgICAgICAgICAgICAgICAgICAgICAgYiA8IDMgPyAwIDogYXJyYXkuZ2V0KGksIGosIDQqaysyLCB3KSwgXG4gICAgICAgICAgICAgICAgICAgICAgICBiIDwgNCA/IDAgOiBhcnJheS5nZXQoaSwgaiwgNCprKzMsIHcpLCBpbmZvKVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiBkYXRhXG59XG5cblxuZXhwb3J0IGZ1bmN0aW9uIHVucGFjayhpbmZvLCBkYXRhLCBkZWNvZGU0LCB0eXBlKXtcblxuXG5cbiAgICB2YXIgc2hhcGUgPSBpbmZvLnNoYXBlO1xuICAgIHZhciBzaGFwZWxlbmd0aCA9IHNoYXBlLnJlZHVjZSgoYSwgYikgPT4gYSAqIGIpO1xuICAgIFxuICAgIHZhciBbd2lkdGgsIGhlaWdodF0gPSBpbmZvLnRleFNpemUsXG4gICAgICAgIGxlbmd0aCA9IHdpZHRoICogaGVpZ2h0ICogNDtcbiAgICB2YXIgY2hhbnMgPSBNYXRoLmNlaWwoaW5mby5zaGFwZVsyXSAvIDQpO1xuXG4gICAgLy8gaWYodHlwZSA9PT0gJ2Zsb2F0MzInKXtcbiAgICB2YXIgYXJyYXkgPSBuZGFycmF5KG5ldyBGbG9hdDMyQXJyYXkoc2hhcGVsZW5ndGgpLCBzaGFwZSlcbiAgICB2YXIgYnVmID0gbmV3IEZsb2F0MzJBcnJheSg0KTtcbiAgICAvLyB9ZWxzZSBpZih0eXBlID09ICd1aW50OCcpe1xuICAgIC8vICAgICB2YXIgYXJyYXkgPSBuZGFycmF5KG5ldyBVaW50OEFycmF5KHNoYXBlbGVuZ3RoKSwgc2hhcGUpXG4gICAgLy8gICAgIHZhciBidWYgPSBuZXcgVWludDhBcnJheSg0KTtcbiAgICAvLyB9ZWxzZSB0aHJvdyBuZXcgRXJyb3IoJ3VuaW1wbGVtZW50ZWQgdHlwZScpO1xuICAgIFxuXG4gICAgZm9yKHZhciBpID0gMDsgaSA8IGluZm8uc2hhcGVbMF07IGkrKyl7XG4gICAgICAgIGZvcih2YXIgaiA9IDA7IGogPCBpbmZvLnNoYXBlWzFdOyBqKyspe1xuICAgICAgICAgICAgZm9yKHZhciBrID0gMDsgayA8IGNoYW5zOyBrKyspe1xuICAgICAgICAgICAgICAgIHZhciBiID0gTWF0aC5taW4oayo0KzQsIHNoYXBlWzJdKS1rKjQ7XG4gICAgICAgICAgICAgICAgZm9yKHZhciB3ID0gMDsgdyA8IGluZm8uc2hhcGVbM107IHcrKyl7XG5cbiAgICAgICAgICAgICAgICAgICAgdmFyIHRpbGUgID0gXG4gICAgICAgICAgICAgICAgICAgICAgICBpICsgXG4gICAgICAgICAgICAgICAgICAgICAgICBqICogc2hhcGVbMF0gKyBcbiAgICAgICAgICAgICAgICAgICAgICAgIGsgKiBzaGFwZVswXSAqIHNoYXBlWzFdICtcbiAgICAgICAgICAgICAgICAgICAgICAgIHcgKiBzaGFwZVswXSAqIHNoYXBlWzFdICogY2hhbnM7XG5cbiAgICAgICAgICAgICAgICAgICAgZGVjb2RlNChidWYsIFxuICAgICAgICAgICAgICAgICAgICAgICAgZGF0YVs0ICogdGlsZSArIDBdLFxuICAgICAgICAgICAgICAgICAgICAgICAgZGF0YVs0ICogdGlsZSArIDFdLFxuICAgICAgICAgICAgICAgICAgICAgICAgZGF0YVs0ICogdGlsZSArIDJdLFxuICAgICAgICAgICAgICAgICAgICAgICAgZGF0YVs0ICogdGlsZSArIDNdLCBpbmZvKVxuXG5cbiAgICAgICAgICAgICAgICAgICAgZm9yKHZhciB4ID0gMDsgeCA8IGI7IHgrKyl7XG4gICAgICAgICAgICAgICAgICAgICAgICBhcnJheS5zZXQoaSwgaiwgNCprK3gsIHcsIGJ1Zlt4XSlcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiBhcnJheTtcblxufVxuIiwiaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xuXG5leHBvcnQgY29uc3QgcmVhZFNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3JlYWQuZ2xzbCcsICd1dGY4Jyk7XG5leHBvcnQgY29uc3Qgd3JpdGVTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy93cml0ZS5nbHNsJywgJ3V0ZjgnKTtcblxuZXhwb3J0IGZ1bmN0aW9uIGluaXQoc2hhcGUpe1xuICAgIHZhciB3aWR0aCA9IHNoYXBlWzBdOyAvLyB2YXIgd2lkdGggPSBzaGFwZVswXSAqIDQ7ICAgIFxuICAgIC8vIHdlIHBpY2sgdGhlIG51bWJlciBvZiBjb2x1bW5zIHNvIHdlIGNhbiBrZWVwXG4gICAgLy8gdGhlIHRleHR1cmUgYXMgc3F1YXJlIGFzIHBvc3NpYmxlLCB3aXRoIHRoZVxuICAgIC8vIG1pbmltYWwgYW1vdW50IG9mIHdhc3RlZCBzcGFjZS5cblxuICAgIHZhciB0aWxlcyA9IE1hdGguY2VpbChzaGFwZVsyXSAvIDQpICogc2hhcGVbM10sXG4gICAgICAgIGNvbHMgPSBNYXRoLm1heCgxLCBNYXRoLm1pbih0aWxlcywgTWF0aC5yb3VuZChcbiAgICAgICAgICAgIE1hdGguc3FydChzaGFwZVswXSAqIHNoYXBlWzFdICogdGlsZXMpIC8gd2lkdGgpKSk7XG5cbiAgICB2YXIgdGV4U2l6ZSA9IFt3aWR0aCAqIGNvbHMsIHNoYXBlWzFdICogTWF0aC5jZWlsKHRpbGVzIC8gY29scyldXG5cbiAgICByZXR1cm4ge1xuICAgIFx0dGV4U2l6ZTogdGV4U2l6ZSxcbiAgICBcdGNvbHM6IGNvbHMsXG4gICAgXHRzaGFwZTogc2hhcGUsXG4gICAgfVxufVxuXG5pbXBvcnQgbmRhcnJheSBmcm9tIFwibmRhcnJheVwiXG5cbmV4cG9ydCBmdW5jdGlvbiBwYWNrKGluZm8sIGFycmF5LCBlbmNvZGU0LCBmb3JtYXQpe1xuICAgIGFycmF5ID0gbmRhcnJheShhcnJheS5kYXRhLCBcbiAgICAgICAgYXJyYXkuc2hhcGUuY29uY2F0KFsxLCAxLCAxLCAxXSkuc2xpY2UoMCwgNCksXG4gICAgICAgIGFycmF5LnN0cmlkZS5jb25jYXQoWzEsIDEsIDEsIDFdKS5zbGljZSgwLCA0KSxcbiAgICAgICAgYXJyYXkub2Zmc2V0KVxuXG4gICAgdmFyIHNoYXBlID0gYXJyYXkuc2hhcGUsXG4gICAgICAgIHRpbGVzID0gTWF0aC5jZWlsKHNoYXBlWzJdIC8gNCkgKiBzaGFwZVszXSxcbiAgICAgICAgdHcgPSBzaGFwZVswXSxcbiAgICAgICAgdGggPSBzaGFwZVsxXSxcbiAgICAgICAgY29scyA9IGluZm8uY29scyxcbiAgICAgICAgW3dpZHRoLCBoZWlnaHRdID0gaW5mby50ZXhTaXplLFxuICAgICAgICBjaHVua3MgPSBNYXRoLmNlaWwoc2hhcGVbMl0gLyA0KSxcbiAgICAgICAgbGVuZ3RoID0gd2lkdGggKiBoZWlnaHQgKiA0O1xuXG4gICAgaWYoZm9ybWF0LnR5cGUgPT09ICdmbG9hdDMyJyl7XG4gICAgICAgIHZhciBkYXRhID0gbmV3IEZsb2F0MzJBcnJheShsZW5ndGgpOyAgICBcbiAgICB9ZWxzZSBpZihmb3JtYXQudHlwZSA9PT0gJ3VpbnQ4Jyl7XG4gICAgICAgIHZhciBkYXRhID0gbmV3IFVpbnQ4QXJyYXkobGVuZ3RoKTsgICAgXG4gICAgfVxuICAgIFxuXG4gICAgZm9yKHZhciB6ID0gMDsgeiA8IGNodW5rczsgeisrKXtcbiAgICAgICAgZm9yKHZhciB3ID0gMDsgdyA8IHNoYXBlWzNdOyB3Kyspe1xuICAgICAgICAgICAgdmFyIHRpbGUgPSB3ICogY2h1bmtzICsgejtcbiAgICAgICAgICAgIHZhciBiID0gTWF0aC5taW4oeio0KzQsIHNoYXBlWzJdKS16KjQ7XG4gICAgICAgICAgICBcbiAgICAgICAgICAgIHZhciBpaCA9IHRoICogTWF0aC5mbG9vcih0aWxlIC8gY29scyk7XG4gICAgICAgICAgICB2YXIgancgPSB0dyAqICh0aWxlICUgY29scyk7XG5cbiAgICAgICAgICAgIGZvcih2YXIgaSA9IDA7IGkgPCB0dzsgaSsrKXtcbiAgICAgICAgICAgICAgICBmb3IodmFyIGogPSAwOyBqIDwgdGg7IGorKyl7XG5cbiAgICAgICAgICAgICAgICAgICAgdmFyIHBvcyA9IDQgKiAoKGloK2opICogd2lkdGggKyBqdyArIGkpO1xuICAgICAgICAgICAgICAgICAgICBlbmNvZGU0KFxuICAgICAgICAgICAgICAgICAgICAgICAgZGF0YS5zdWJhcnJheShwb3MsIHBvcyArIDQpLFxuICAgICAgICAgICAgICAgICAgICAgICAgYiA8IDEgPyAwIDogYXJyYXkuZ2V0KGksIGosIDQqeiswLCB3KSwgXG4gICAgICAgICAgICAgICAgICAgICAgICBiIDwgMiA/IDAgOiBhcnJheS5nZXQoaSwgaiwgNCp6KzEsIHcpLCBcbiAgICAgICAgICAgICAgICAgICAgICAgIGIgPCAzID8gMCA6IGFycmF5LmdldChpLCBqLCA0KnorMiwgdyksIFxuICAgICAgICAgICAgICAgICAgICAgICAgYiA8IDQgPyAwIDogYXJyYXkuZ2V0KGksIGosIDQqeiszLCB3KSwgaW5mbylcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGRhdGE7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB1bnBhY2soaW5mbywgZGF0YSwgZGVjb2RlNCwgdHlwZSl7XG4gICAgdGhyb3cgbmV3IEVycm9yKFwibm90IGltcGxlbWVudGVkOiBmb3JtYXQvNC00L3BhY2svdGlsZS9pbmRleC5qczp1bnBhY2tcIilcbn0iLCJpbXBvcnQgRm9ybWF0NDQgZnJvbSAnLi80LTQvaW5kZXguanMnXG5pbXBvcnQgRm9ybWF0MTQgZnJvbSAnLi8xLTQvaW5kZXguanMnXG5cbmV4cG9ydCBkZWZhdWx0IHtcblx0JzQ6NCc6IEZvcm1hdDQ0LFxuXHQnMTo0JzogRm9ybWF0MTQsXG59IiwiLy8gZG8geW91IGV2ZXIgaG9wZSB0aGF0IHBlcmhhcHMgaW5kZXggZmlsZXMgc2hvdWxkIFxuLy8gYWN0dWFsbHkgYmUgaW5kZXggZmlsZXMgbGFja2luZyBhbnkgaW1wbGVtZW50YXRpb24gXG4vLyBjb2RlPyB3ZWxsLCB0b2RheSB5b3UncmUgaW4gbHVjayFcblxuZXhwb3J0IHsgVGVuc29yLCBPdXRwdXRUZW5zb3IsIEluUGxhY2VUZW5zb3IgfSBmcm9tICcuL3RlbnNvci9pbmRleC5qcydcbmV4cG9ydCB7IFJ1biwgQ29tcGlsZSB9IGZyb20gJy4vcnVudGltZS9pbmRleC5qcydcbmV4cG9ydCB7IGNyZWF0ZUdMIH0gZnJvbSAnLi91dGlsLmpzJyIsIi8vIGNvZGUgZm9yIHByZXR0eSBwcmludGluZyBzaGFkZXIgZXJyb3JzIGZyb20gcmVnbFxuXG5leHBvcnQgZnVuY3Rpb24gY2hlY2tMaW5rRXJyb3IgKGdsLCBwcm9ncmFtLCBmcmFnU2hhZGVyLCB2ZXJ0U2hhZGVyLCBjb21tYW5kKSB7XG4gICAgaWYgKCFnbC5nZXRQcm9ncmFtUGFyYW1ldGVyKHByb2dyYW0sIGdsLkxJTktfU1RBVFVTKSkge1xuICAgICAgICB2YXIgZXJyTG9nID0gZ2wuZ2V0UHJvZ3JhbUluZm9Mb2cocHJvZ3JhbSlcbiAgICAgICAgdmFyIGZyYWdQYXJzZSA9IHBhcnNlU291cmNlKGZyYWdTaGFkZXIsIGNvbW1hbmQpXG4gICAgICAgIHZhciB2ZXJ0UGFyc2UgPSBwYXJzZVNvdXJjZSh2ZXJ0U2hhZGVyLCBjb21tYW5kKVxuXG4gICAgICAgIHZhciBoZWFkZXIgPSAnRXJyb3IgbGlua2luZyBwcm9ncmFtIHdpdGggdmVydGV4IHNoYWRlciwgXCInICtcbiAgICAgICAgICAgIHZlcnRQYXJzZVswXS5uYW1lICsgJ1wiLCBhbmQgZnJhZ21lbnQgc2hhZGVyIFwiJyArIGZyYWdQYXJzZVswXS5uYW1lICsgJ1wiJ1xuXG4gICAgICAgIGlmICh0eXBlb2YgZG9jdW1lbnQgIT09ICd1bmRlZmluZWQnKSB7XG4gICAgICAgICAgICBjb25zb2xlLmxvZygnJWMnICsgaGVhZGVyICsgJ1xcbiVjJyArIGVyckxvZyxcbiAgICAgICAgICAgICAgICAnY29sb3I6cmVkO3RleHQtZGVjb3JhdGlvbjp1bmRlcmxpbmU7Zm9udC13ZWlnaHQ6Ym9sZCcsXG4gICAgICAgICAgICAgICAgJ2NvbG9yOnJlZCcpXG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBjb25zb2xlLmxvZyhoZWFkZXIgKyAnXFxuJyArIGVyckxvZylcbiAgICAgICAgfVxuXG4gICAgICAgIGNvbnNvbGUubG9nKGZyYWdTaGFkZXIpO1xuICAgICAgICBcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKGhlYWRlcilcbiAgICB9XG59XG5cblxuZXhwb3J0IGZ1bmN0aW9uIGNoZWNrU2hhZGVyRXJyb3IgKGdsLCBzaGFkZXIsIHNvdXJjZSwgdHlwZSwgY29tbWFuZCkge1xuICAgIGlmICghZ2wuZ2V0U2hhZGVyUGFyYW1ldGVyKHNoYWRlciwgZ2wuQ09NUElMRV9TVEFUVVMpKSB7XG4gICAgICAgIHZhciBlcnJMb2cgPSBnbC5nZXRTaGFkZXJJbmZvTG9nKHNoYWRlcilcbiAgICAgICAgdmFyIHR5cGVOYW1lID0gdHlwZSA9PT0gZ2wuRlJBR01FTlRfU0hBREVSID8gJ2ZyYWdtZW50JyA6ICd2ZXJ0ZXgnXG4gICAgICAgIC8vIGNoZWNrQ29tbWFuZFR5cGUoc291cmNlLCAnc3RyaW5nJywgdHlwZU5hbWUgKyAnIHNoYWRlciBzb3VyY2UgbXVzdCBiZSBhIHN0cmluZycsIGNvbW1hbmQpXG5cbiAgICAgICAgdmFyIGZpbGVzID0gcGFyc2VTb3VyY2Uoc291cmNlLCBjb21tYW5kKVxuICAgICAgICB2YXIgZXJyb3JzID0gcGFyc2VFcnJvckxvZyhlcnJMb2cpXG4gICAgICAgIGFubm90YXRlRmlsZXMoZmlsZXMsIGVycm9ycylcblxuICAgICAgICBPYmplY3Qua2V5cyhmaWxlcykuZm9yRWFjaChmdW5jdGlvbiAoZmlsZU51bWJlcikge1xuICAgICAgICAgICAgdmFyIGZpbGUgPSBmaWxlc1tmaWxlTnVtYmVyXVxuICAgICAgICAgICAgaWYgKCFmaWxlLmhhc0Vycm9ycykge1xuICAgICAgICAgICAgICAgIHJldHVyblxuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICB2YXIgc3RyaW5ncyA9IFsnJ11cbiAgICAgICAgICAgIHZhciBzdHlsZXMgPSBbJyddXG5cbiAgICAgICAgICAgIGZ1bmN0aW9uIHB1c2ggKHN0ciwgc3R5bGUpIHtcbiAgICAgICAgICAgICAgICBzdHJpbmdzLnB1c2goc3RyKVxuICAgICAgICAgICAgICAgIHN0eWxlcy5wdXNoKHN0eWxlIHx8ICcnKVxuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBwdXNoKCdmaWxlIG51bWJlciAnICsgZmlsZU51bWJlciArICc6ICcgKyBmaWxlLm5hbWUgKyAnXFxuJywgJ2NvbG9yOnJlZDt0ZXh0LWRlY29yYXRpb246dW5kZXJsaW5lO2ZvbnQtd2VpZ2h0OmJvbGQnKVxuXG4gICAgICAgICAgICBmaWxlLmxpbmVzLmZvckVhY2goZnVuY3Rpb24gKGxpbmUpIHtcbiAgICAgICAgICAgICAgICBpZiAobGluZS5lcnJvcnMubGVuZ3RoID4gMCkge1xuICAgICAgICAgICAgICAgICAgICBwdXNoKGxlZnRQYWQobGluZS5udW1iZXIsIDQpICsgJ3wgICcsICdiYWNrZ3JvdW5kLWNvbG9yOnllbGxvdzsgZm9udC13ZWlnaHQ6Ym9sZCcpXG4gICAgICAgICAgICAgICAgICAgIHB1c2gobGluZS5saW5lICsgJ1xcbicsICdjb2xvcjpyZWQ7IGJhY2tncm91bmQtY29sb3I6eWVsbG93OyBmb250LXdlaWdodDpib2xkJylcblxuICAgICAgICAgICAgICAgICAgICAvLyB0cnkgdG8gZ3Vlc3MgdG9rZW5cbiAgICAgICAgICAgICAgICAgICAgdmFyIG9mZnNldCA9IDBcbiAgICAgICAgICAgICAgICAgICAgbGluZS5lcnJvcnMuZm9yRWFjaChmdW5jdGlvbiAoZXJyb3IpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciBtZXNzYWdlID0gZXJyb3IubWVzc2FnZVxuICAgICAgICAgICAgICAgICAgICAgICAgdmFyIHRva2VuID0gL15cXHMqXFwnKC4qKVxcJ1xccypcXDpcXHMqKC4qKSQvLmV4ZWMobWVzc2FnZSlcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmICh0b2tlbikge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZhciB0b2tlblBhdCA9IHRva2VuWzFdXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgbWVzc2FnZSA9IHRva2VuWzJdXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgc3dpdGNoICh0b2tlblBhdCkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjYXNlICdhc3NpZ24nOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdG9rZW5QYXQgPSAnPSdcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGJyZWFrXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIG9mZnNldCA9IE1hdGgubWF4KGxpbmUubGluZS5pbmRleE9mKHRva2VuUGF0LCBvZmZzZXQpLCAwKVxuICAgICAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBvZmZzZXQgPSAwXG4gICAgICAgICAgICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgICAgICAgICAgIHB1c2gobGVmdFBhZCgnfCAnLCA2KSlcbiAgICAgICAgICAgICAgICAgICAgICAgIHB1c2gobGVmdFBhZCgnXl5eJywgb2Zmc2V0ICsgMykgKyAnXFxuJywgJ2ZvbnQtd2VpZ2h0OmJvbGQnKVxuICAgICAgICAgICAgICAgICAgICAgICAgcHVzaChsZWZ0UGFkKCd8ICcsIDYpKVxuICAgICAgICAgICAgICAgICAgICAgICAgcHVzaChtZXNzYWdlICsgJ1xcbicsICdmb250LXdlaWdodDpib2xkJylcbiAgICAgICAgICAgICAgICAgICAgfSlcbiAgICAgICAgICAgICAgICAgICAgcHVzaChsZWZ0UGFkKCd8ICcsIDYpICsgJ1xcbicpXG4gICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgcHVzaChsZWZ0UGFkKGxpbmUubnVtYmVyLCA0KSArICd8ICAnKVxuICAgICAgICAgICAgICAgICAgICBwdXNoKGxpbmUubGluZSArICdcXG4nLCAnY29sb3I6cmVkJylcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9KVxuICAgICAgICAgICAgaWYgKHR5cGVvZiBkb2N1bWVudCAhPT0gJ3VuZGVmaW5lZCcpIHtcbiAgICAgICAgICAgICAgICBzdHlsZXNbMF0gPSBzdHJpbmdzLmpvaW4oJyVjJylcbiAgICAgICAgICAgICAgICBjb25zb2xlLmxvZy5hcHBseShjb25zb2xlLCBzdHlsZXMpXG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKHN0cmluZ3Muam9pbignJykpXG4gICAgICAgICAgICB9XG4gICAgICAgIH0pXG5cbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdFcnJvciBjb21waWxpbmcgJyArIHR5cGVOYW1lICsgJyBzaGFkZXIsICcgKyBmaWxlc1swXS5uYW1lKVxuICAgIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNoZWNrRnJhbWVidWZmZXJFcnJvcihnbCl7XG4gICAgXG4gICAgdmFyIHN0YXR1cyA9IGdsLmNoZWNrRnJhbWVidWZmZXJTdGF0dXMoZ2wuRlJBTUVCVUZGRVIpO1xuICAgIGlmKHN0YXR1cyAhPSBnbC5GUkFNRUJVRkZFUl9DT01QTEVURSl7XG4gICAgICAgIHZhciBzdGF0dXNDb2RlID0ge31cbiAgICAgICAgc3RhdHVzQ29kZVtnbC5GUkFNRUJVRkZFUl9DT01QTEVURV0gPSAnY29tcGxldGUnXG4gICAgICAgIHN0YXR1c0NvZGVbZ2wuRlJBTUVCVUZGRVJfSU5DT01QTEVURV9BVFRBQ0hNRU5UXSA9ICdpbmNvbXBsZXRlIGF0dGFjaG1lbnQnXG4gICAgICAgIHN0YXR1c0NvZGVbZ2wuRlJBTUVCVUZGRVJfSU5DT01QTEVURV9ESU1FTlNJT05TXSA9ICdpbmNvbXBsZXRlIGRpbWVuc2lvbnMnXG4gICAgICAgIHN0YXR1c0NvZGVbZ2wuRlJBTUVCVUZGRVJfSU5DT01QTEVURV9NSVNTSU5HX0FUVEFDSE1FTlRdID0gJ2luY29tcGxldGUsIG1pc3NpbmcgYXR0YWNobWVudCdcbiAgICAgICAgc3RhdHVzQ29kZVtnbC5GUkFNRUJVRkZFUl9VTlNVUFBPUlRFRF0gPSAndW5zdXBwb3J0ZWQnXG4gICAgICAgIHRocm93IG5ldyBFcnJvcignZnJhbWVidWZmZXIgY29uZmlndXJhdGlvbiBub3Qgc3VwcG9ydGVkLCBzdGF0dXMgPSAnICsgc3RhdHVzQ29kZVtzdGF0dXNdKVxuICAgIH1cbn1cblxuXG5mdW5jdGlvbiBsZWZ0UGFkIChzdHIsIG4pIHtcbiAgICBzdHIgPSBzdHIgKyAnJ1xuICAgIHdoaWxlIChzdHIubGVuZ3RoIDwgbikge1xuICAgICAgICBzdHIgPSAnICcgKyBzdHJcbiAgICB9XG4gICAgcmV0dXJuIHN0clxufVxuXG5mdW5jdGlvbiBTaGFkZXJGaWxlICgpIHtcbiAgICB0aGlzLm5hbWUgPSAndW5rbm93bidcbiAgICB0aGlzLmxpbmVzID0gW11cbiAgICB0aGlzLmluZGV4ID0ge31cbiAgICB0aGlzLmhhc0Vycm9ycyA9IGZhbHNlXG59XG5cbmZ1bmN0aW9uIFNoYWRlckxpbmUgKG51bWJlciwgbGluZSkge1xuICAgIHRoaXMubnVtYmVyID0gbnVtYmVyXG4gICAgdGhpcy5saW5lID0gbGluZVxuICAgIHRoaXMuZXJyb3JzID0gW11cbn1cblxuZnVuY3Rpb24gU2hhZGVyRXJyb3IgKGZpbGVOdW1iZXIsIGxpbmVOdW1iZXIsIG1lc3NhZ2UpIHtcbiAgICB0aGlzLmZpbGUgPSBmaWxlTnVtYmVyXG4gICAgdGhpcy5saW5lID0gbGluZU51bWJlclxuICAgIHRoaXMubWVzc2FnZSA9IG1lc3NhZ2Vcbn1cblxuZnVuY3Rpb24gcGFyc2VTb3VyY2UgKHNvdXJjZSwgY29tbWFuZCkge1xuICAgIHZhciBsaW5lcyA9IHNvdXJjZS5zcGxpdCgnXFxuJylcbiAgICB2YXIgbGluZU51bWJlciA9IDFcbiAgICB2YXIgZmlsZU51bWJlciA9IDBcbiAgICB2YXIgZmlsZXMgPSB7XG4gICAgICAgIHVua25vd246IG5ldyBTaGFkZXJGaWxlKCksXG4gICAgICAgIDA6IG5ldyBTaGFkZXJGaWxlKClcbiAgICB9XG4gICAgZmlsZXMudW5rbm93bi5uYW1lID0gZmlsZXNbMF0ubmFtZSA9ICd1bmtub3duJ1xuICAgIGZpbGVzLnVua25vd24ubGluZXMucHVzaChuZXcgU2hhZGVyTGluZSgwLCAnJykpXG4gICAgZm9yICh2YXIgaSA9IDA7IGkgPCBsaW5lcy5sZW5ndGg7ICsraSkge1xuICAgICAgICB2YXIgbGluZSA9IGxpbmVzW2ldXG4gICAgICAgIHZhciBwYXJ0cyA9IC9eXFxzKlxcI1xccyooXFx3KylcXHMrKC4rKVxccyokLy5leGVjKGxpbmUpXG4gICAgICAgIGlmIChwYXJ0cykge1xuICAgICAgICAgICAgc3dpdGNoIChwYXJ0c1sxXSkge1xuICAgICAgICAgICAgICAgIGNhc2UgJ2xpbmUnOlxuICAgICAgICAgICAgICAgICAgICB2YXIgbGluZU51bWJlckluZm8gPSAvKFxcZCspKFxccytcXGQrKT8vLmV4ZWMocGFydHNbMl0pXG4gICAgICAgICAgICAgICAgICAgIGlmIChsaW5lTnVtYmVySW5mbykge1xuICAgICAgICAgICAgICAgICAgICAgICAgbGluZU51bWJlciA9IGxpbmVOdW1iZXJJbmZvWzFdIHwgMFxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKGxpbmVOdW1iZXJJbmZvWzJdKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgZmlsZU51bWJlciA9IGxpbmVOdW1iZXJJbmZvWzJdIHwgMFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGlmICghKGZpbGVOdW1iZXIgaW4gZmlsZXMpKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZpbGVzW2ZpbGVOdW1iZXJdID0gbmV3IFNoYWRlckZpbGUoKVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBicmVha1xuICAgICAgICAgICAgICAgIGNhc2UgJ2RlZmluZSc6XG4gICAgICAgICAgICAgICAgICAgIHZhciBuYW1lSW5mbyA9IC9TSEFERVJfTkFNRShfQjY0KT9cXHMrKC4qKSQvLmV4ZWMocGFydHNbMl0pXG4gICAgICAgICAgICAgICAgICAgIGlmIChuYW1lSW5mbykge1xuICAgICAgICAgICAgICAgICAgICAgICAgZmlsZXNbZmlsZU51bWJlcl0ubmFtZSA9IChuYW1lSW5mb1sxXVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA/IGRlY29kZUI2NChuYW1lSW5mb1syXSlcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgOiBuYW1lSW5mb1syXSlcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBicmVha1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIGZpbGVzW2ZpbGVOdW1iZXJdLmxpbmVzLnB1c2gobmV3IFNoYWRlckxpbmUobGluZU51bWJlcisrLCBsaW5lKSlcbiAgICB9XG4gICAgT2JqZWN0LmtleXMoZmlsZXMpLmZvckVhY2goZnVuY3Rpb24gKGZpbGVOdW1iZXIpIHtcbiAgICAgICAgdmFyIGZpbGUgPSBmaWxlc1tmaWxlTnVtYmVyXVxuICAgICAgICBmaWxlLmxpbmVzLmZvckVhY2goZnVuY3Rpb24gKGxpbmUpIHtcbiAgICAgICAgICAgIGZpbGUuaW5kZXhbbGluZS5udW1iZXJdID0gbGluZVxuICAgICAgICB9KVxuICAgIH0pXG4gICAgcmV0dXJuIGZpbGVzXG59XG5cbmZ1bmN0aW9uIHBhcnNlRXJyb3JMb2cgKGVyckxvZykge1xuICAgIHZhciByZXN1bHQgPSBbXVxuICAgIGVyckxvZy5zcGxpdCgnXFxuJykuZm9yRWFjaChmdW5jdGlvbiAoZXJyTXNnKSB7XG4gICAgICAgIGlmIChlcnJNc2cubGVuZ3RoIDwgNSkge1xuICAgICAgICAgICAgcmV0dXJuXG4gICAgICAgIH1cbiAgICAgICAgdmFyIHBhcnRzID0gL15FUlJPUlxcOlxccysoXFxkKylcXDooXFxkKylcXDpcXHMqKC4qKSQvLmV4ZWMoZXJyTXNnKVxuICAgICAgICBpZiAocGFydHMpIHtcbiAgICAgICAgICAgIHJlc3VsdC5wdXNoKG5ldyBTaGFkZXJFcnJvcihcbiAgICAgICAgICAgICAgICBwYXJ0c1sxXSB8IDAsXG4gICAgICAgICAgICAgICAgcGFydHNbMl0gfCAwLFxuICAgICAgICAgICAgICAgIHBhcnRzWzNdLnRyaW0oKSkpXG4gICAgICAgIH0gZWxzZSBpZiAoZXJyTXNnLmxlbmd0aCA+IDApIHtcbiAgICAgICAgICAgIHJlc3VsdC5wdXNoKG5ldyBTaGFkZXJFcnJvcigndW5rbm93bicsIDAsIGVyck1zZykpXG4gICAgICAgIH1cbiAgICB9KVxuICAgIHJldHVybiByZXN1bHRcbn1cblxuZnVuY3Rpb24gYW5ub3RhdGVGaWxlcyAoZmlsZXMsIGVycm9ycykge1xuICAgIGVycm9ycy5mb3JFYWNoKGZ1bmN0aW9uIChlcnJvcikge1xuICAgICAgICB2YXIgZmlsZSA9IGZpbGVzW2Vycm9yLmZpbGVdXG4gICAgICAgIGlmIChmaWxlKSB7XG4gICAgICAgICAgICB2YXIgbGluZSA9IGZpbGUuaW5kZXhbZXJyb3IubGluZV1cbiAgICAgICAgICAgIGlmIChsaW5lKSB7XG4gICAgICAgICAgICAgICAgbGluZS5lcnJvcnMucHVzaChlcnJvcilcbiAgICAgICAgICAgICAgICBmaWxlLmhhc0Vycm9ycyA9IHRydWVcbiAgICAgICAgICAgICAgICByZXR1cm5cbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBmaWxlcy51bmtub3duLmhhc0Vycm9ycyA9IHRydWVcbiAgICAgICAgZmlsZXMudW5rbm93bi5saW5lc1swXS5lcnJvcnMucHVzaChlcnJvcilcbiAgICB9KVxufVxuIiwiLy8gaW1wb3J0IHsgVGVuc29yLCBPdXRwdXRUZW5zb3IsIEluUGxhY2VUZW5zb3IgfSBmcm9tICcuLi90ZW5zb3IvaW5kZXguanMnXG5pbXBvcnQgQmFzZVRlbnNvciBmcm9tICcuLi90ZW5zb3IvYmFzZS5qcydcblxuaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xuXG5jb25zdCBURU5TT1JfRlJBR01FTlRfSEVBREVSID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvLi4vZm9ybWF0L3V0aWwuZ2xzbCcsICd1dGY4JylcblxuXG5leHBvcnQgZGVmYXVsdCBmdW5jdGlvbiBhc3NlbWJsZUZyYWdtZW50U2hhZGVyKHNoYWRlckdlbiwgb3V0cHV0LCB1bmlmb3Jtcyl7XG4gICAgdmFyIHRlbnNvclNoYWRlciA9IHNoYWRlckdlbih1bmlmb3Jtcywgb3V0cHV0KTtcbiAgICBcbiAgICB2YXIgZnJhZ21lbnRTaGFkZXIgPSBURU5TT1JfRlJBR01FTlRfSEVBREVSO1xuICAgIGZvcihsZXQgdW5pZm9ybSBpbiB1bmlmb3Jtcyl7XG4gICAgICAgIGlmKHVuaWZvcm1zW3VuaWZvcm1dIGluc3RhbmNlb2YgQmFzZVRlbnNvcil7XG4gICAgICAgICAgICBsZXQgdGVuc29yID0gdW5pZm9ybXNbdW5pZm9ybV07XG5cbiAgICAgICAgICAgIGZyYWdtZW50U2hhZGVyICs9IHRlbnNvci5fZm9ybWF0LmNvZGVjLmRlY29kZVNoYWRlci5yZXBsYWNlKC9AL2csIHVuaWZvcm0gKyAnXycpICsgJ1xcbidcbiAgICAgICAgICAgIGZyYWdtZW50U2hhZGVyICs9IHRlbnNvci5fZm9ybWF0LnBhY2sucmVhZFNoYWRlci5yZXBsYWNlKC9AL2csIHVuaWZvcm0gKyAnXycpICsgJ1xcbidcblxuICAgICAgICAgICAgaWYoKHRlbnNvci5mb3JtYXQuZGVuc2l0eSA9PSAnMTo0JyAmJiAobmV3IFJlZ0V4cCh1bmlmb3JtICsgJ19yZWFkNFxcXFxiJykpLnRlc3QodGVuc29yU2hhZGVyKSkgfHwgXG4gICAgICAgICAgICAgICAgKHRlbnNvci5mb3JtYXQuZGVuc2l0eSA9PSAnNDo0JyAmJiAobmV3IFJlZ0V4cCh1bmlmb3JtICsgJ19yZWFkXFxcXGInKSkudGVzdCh0ZW5zb3JTaGFkZXIpKSl7XG4gICAgICAgICAgICAgICAgZnJhZ21lbnRTaGFkZXIgKz0gdGVuc29yLl9mb3JtYXQucmVhZF9zaGltLnJlcGxhY2UoL0AvZywgdW5pZm9ybSArICdfJykgKyAnXFxuJztcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgIH1cblxuICAgIHZhciBhY3RpdmF0aW9uID0gKHR5cGVvZiB1bmlmb3Jtcy5fYWN0aXZhdGlvbiA9PSAnc3RyaW5nJyAmJiB1bmlmb3Jtcy5fYWN0aXZhdGlvbiAhPSAnbGluZWFyJykgP1xuICAgICAgICB1bmlmb3Jtcy5fYWN0aXZhdGlvbi50b0xvd2VyQ2FzZSgpIDogJ2xpbmVhcic7XG5cbiAgICBpZighKGFjdGl2YXRpb24gaW4gb3V0cHV0Ll9mb3JtYXQuYWN0aXZhdGlvbnMpKVxuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ1Vua25vd24gYWN0aXZhdGlvbiB0eXBlICcgKyBhY3RpdmF0aW9uKTtcblxuICAgIGZyYWdtZW50U2hhZGVyICs9IG91dHB1dC5fZm9ybWF0LmFjdGl2YXRpb25zW2FjdGl2YXRpb25dLnJlcGxhY2UoL0AvZywgJ291dF8nKSArICdcXG4nO1xuICAgIGZyYWdtZW50U2hhZGVyICs9IG91dHB1dC5fZm9ybWF0LmNvZGVjLmVuY29kZVNoYWRlci5yZXBsYWNlKC9AL2csICdvdXRfJykgKyAnXFxuJztcbiAgICBmcmFnbWVudFNoYWRlciArPSBvdXRwdXQuX2Zvcm1hdC5wYWNrLndyaXRlU2hhZGVyLnJlcGxhY2UoL0AvZywgJ291dF8nKSArICdcXG4nO1xuXG5cbiAgICBpZigob3V0cHV0LmZvcm1hdC5kZW5zaXR5ID09ICcxOjQnICYmIC9wcm9jZXNzNFxcYi8udGVzdCh0ZW5zb3JTaGFkZXIpKSB8fCBcbiAgICAgICAgKG91dHB1dC5mb3JtYXQuZGVuc2l0eSA9PSAnNDo0JyAmJiAvcHJvY2Vzc1xcYi8udGVzdCh0ZW5zb3JTaGFkZXIpKSl7XG4gICAgICAgIGZyYWdtZW50U2hhZGVyICs9IG91dHB1dC5fZm9ybWF0LndyaXRlX3NoaW0ucmVwbGFjZSgvQC9nLCAnb3V0XycpICsgJ1xcbic7XG4gICAgfVxuXG4gICAgZnJhZ21lbnRTaGFkZXIgKz0gdGVuc29yU2hhZGVyLnJlcGxhY2UoL0AvZywgJ291dF8nKVxuXG4gICAgLy8gY29uc29sZS5sb2coZnJhZ21lbnRTaGFkZXIpXG5cbiAgICByZXR1cm4gZnJhZ21lbnRTaGFkZXI7XG59IiwiaW1wb3J0IGdldFRlbnNvclByb2dyYW0gZnJvbSAnLi9wcm9ncmFtLmpzJ1xuaW1wb3J0IGFzc2VtYmxlRnJhZ21lbnRTaGFkZXIgZnJvbSAnLi9mcmFnLmpzJ1xuaW1wb3J0IHsgVGVuc29yLCBPdXRwdXRUZW5zb3IsIEluUGxhY2VUZW5zb3IgfSBmcm9tICcuLi90ZW5zb3IvaW5kZXguanMnXG5pbXBvcnQgeyBjaGVja0ZyYW1lYnVmZmVyRXJyb3IgfSBmcm9tICcuL2NoZWNrLmpzJ1xuaW1wb3J0IFROU0wgZnJvbSAnLi90bnNsLmpzJ1xuaW1wb3J0IHsgYmVnaW5UaW1lciwgZW5kVGltZXIsIG5vdyB9IGZyb20gJy4vdGltZXIuanMnXG5cblxuZXhwb3J0IGZ1bmN0aW9uIENvbXBpbGUoc2hhZGVyR2VuLCBvdXRwdXQsIHVuaWZvcm1zID0ge30pe1xuICAgIHZhciBzdGFydFRpbWUgPSBub3coKTtcbiAgICBpZighKG91dHB1dCBpbnN0YW5jZW9mIE91dHB1dFRlbnNvcikpIFxuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXCJGaXJzdCBhcmd1bWVudCBtdXN0IGJlIGFuIGluc3RhbmNlIG9mIE91dHB1dFRlbnNvclwiKTtcbiAgICBcbiAgICBpZih0eXBlb2Ygc2hhZGVyR2VuID09PSAnc3RyaW5nJykgc2hhZGVyR2VuID0gVE5TTChzaGFkZXJHZW4pO1xuICAgIFxuICAgIHZhciBnbCA9IG91dHB1dC5nbDtcbiAgICB2YXIgcHJvZ3JhbSA9IGdldFRlbnNvclByb2dyYW0oZ2wsIGFzc2VtYmxlRnJhZ21lbnRTaGFkZXIoc2hhZGVyR2VuLCBvdXRwdXQsIHVuaWZvcm1zKSk7XG4gICAgdmFyIGNvbXBpbGVUaW1lID0gbm93KCkgLSBzdGFydFRpbWU7XG4gICAgLy8gY29uc29sZS5sb2coJ0NvbXBpbGUgVGltZScsIGNvbXBpbGVUaW1lKVxuICAgIHJldHVybiBwcm9ncmFtO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gUnVuKHNoYWRlckdlbiwgb3V0cHV0LCB1bmlmb3JtcyA9IHt9LCBjYWxsYmFjayA9IG51bGwpe1xuICAgIHZhciB0cCA9IENvbXBpbGUoc2hhZGVyR2VuLCBvdXRwdXQsIHVuaWZvcm1zKTtcblxuICAgIHZhciBnbCA9IG91dHB1dC5nbDtcbiAgICBcbiAgICBpZihjYWxsYmFjayAmJiB0eXBlb2YgY2FsbGJhY2sgIT0gJ2Z1bmN0aW9uJykgdGhyb3cgbmV3IEVycm9yKCdDYWxsYmFjayBtdXN0IGJlIGEgZnVuY3Rpb24nKTtcbiAgICBpZihjYWxsYmFjayl7XG4gICAgICAgIGJlZ2luVGltZXIoZ2wsIHtcbiAgICAgICAgICAgIHNoYWRlcjogc2hhZGVyR2VuLFxuICAgICAgICAgICAgb3V0cHV0OiBvdXRwdXRcbiAgICAgICAgfSlcbiAgICB9XG5cbiAgICBnbC51c2VQcm9ncmFtKHRwLnByb2dyYW0pO1xuICAgIGdsLmRpc2FibGUoZ2wuREVQVEhfVEVTVCk7XG4gICAgZ2wuZGlzYWJsZShnbC5CTEVORCk7XG5cbiAgICB2YXIgc2V0VW5pZm9ybSA9IHRwLnNldFVuaWZvcm0sXG4gICAgICAgIHRleEluZGV4ID0gMCxcbiAgICAgICAgbXVzdFN3YXAgPSBmYWxzZTtcbiAgICAgICAgXG4gICAgZm9yKGxldCBuYW1lIGluIHVuaWZvcm1zKXtcbiAgICAgICAgaWYobmFtZS5zdGFydHNXaXRoKCdfJykpIGNvbnRpbnVlO1xuICAgICAgICBcbiAgICAgICAgaWYoKG5hbWUgKyAnX3RleCcpIGluIHRwLnVuaWZvcm1UeXBlcyl7XG4gICAgICAgICAgICBsZXQgdGVuc29yID0gdW5pZm9ybXNbbmFtZV07XG4gICAgICAgICAgICBpZih0ZW5zb3IuZ2wgIT09IG91dHB1dC5nbCkgdGhyb3cgbmV3IEVycm9yKCdVbmlmb3JtcyBtdXN0IGJlbG9uZyB0byBzYW1lIEdMIGNvbnRleHQgYXMgb3V0cHV0Jyk7XG4gICAgICAgICAgICBpZih0ZW5zb3IgPT09IG91dHB1dCkgbXVzdFN3YXAgPSB0cnVlO1xuXG4gICAgICAgICAgICBmb3IobGV0IHVuaWZvcm0gaW4gdGVuc29yLmluZm8pe1xuICAgICAgICAgICAgICAgIHNldFVuaWZvcm0obmFtZSArICdfJyArIHVuaWZvcm0sIHRlbnNvci5pbmZvW3VuaWZvcm1dKVxuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBnbC5hY3RpdmVUZXh0dXJlKGdsWydURVhUVVJFJyArIHRleEluZGV4XSk7XG4gICAgICAgICAgICBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0ZW5zb3IudGV4KTtcbiAgICAgICAgICAgIHNldFVuaWZvcm0obmFtZSArICdfdGV4JywgdGV4SW5kZXgpO1xuXG4gICAgICAgICAgICB0ZXhJbmRleCsrXG4gICAgICAgIH1lbHNlIGlmKG5hbWUgaW4gdHAudW5pZm9ybVR5cGVzKXtcbiAgICAgICAgICAgIHNldFVuaWZvcm0obmFtZSwgdW5pZm9ybXNbbmFtZV0pXG4gICAgICAgIH1lbHNle1xuICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKFwiVW5rbm93biB1bmlmb3JtIFwiICsgbmFtZSk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBPcmRpbmFyaWx5IHdlIGNhbid0IHdyaXRlIHRvIHRoZSBzYW1lIHRleHR1cmUgdGhhdCB3ZSdyZSB1c2luZyBhc1xuICAgIC8vIGFuIGlucHV0LCBhcyB0aGlzIGNvdWxkIGxlYWQgdG8gYWxsIHNvcnRzIG9mIHRlcnJpYmxlIHJhY2UgY29uZGl0aW9ucyxcbiAgICAvLyB1bmRlZmluZWQgYmVoYXZpb3IsIGFuZCBpbnZhbGlkIHN0YXRlLiBJblBsYWNlVGVuc29ycyBhY3R1YWxseSBjb25zaXN0XG4gICAgLy8gb2YgYSBwYWlyIG9mIHRleHR1cmVzIHdoaWNoIGFyZSBzd2FwcGVkIGZvciB0aGVzZSBpbi1wbGFjZSBvcGVyYXRpb25zLiBcbiAgICBpZihtdXN0U3dhcCkgb3V0cHV0LnN3YXAoKTtcblxuICAgIGZvcihsZXQgdW5pZm9ybSBpbiBvdXRwdXQuaW5mbyl7XG4gICAgICAgIHNldFVuaWZvcm0oJ291dF8nICsgdW5pZm9ybSwgb3V0cHV0LmluZm9bdW5pZm9ybV0pXG4gICAgfVxuXG4gICAgZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBvdXRwdXQuZmJvKTtcbiAgICBnbC52aWV3cG9ydCgwLCAwLCBvdXRwdXQuaW5mby50ZXhTaXplWzBdLCBvdXRwdXQuaW5mby50ZXhTaXplWzFdKTtcbiAgICBnbC5kcmF3QXJyYXlzKGdsLlRSSUFOR0xFX1NUUklQLCAwLCA0KTsgLy8gZHJhdyB0byBmcmFtZWJ1ZmZlclxuXG4gICAgY2hlY2tGcmFtZWJ1ZmZlckVycm9yKGdsKTtcbiAgICBcbiAgICAvLyB2YXIgcnVuVGltZSA9IG5vdygpIC0gc3RhcnRUaW1lO1xuICAgIC8vIHRpbWVyLmVuZCgpXG4gICAgaWYoY2FsbGJhY2spe1xuICAgICAgICBlbmRUaW1lcihnbCwgZnVuY3Rpb24oaW5mbyl7XG4gICAgICAgICAgICAvLyBjb25zb2xlLmxvZygnR1BVIHRpbWU6ICcsIGluZm8pXG4gICAgICAgICAgICBjYWxsYmFjayhpbmZvKTtcbiAgICAgICAgfSkgICAgXG4gICAgfVxuICAgIC8vIGNvbnNvbGUubG9nKCdDUFUgUnVuIFRpbWUnLCBydW5UaW1lKVxuXG4gICAgcmV0dXJuIG91dHB1dDtcbn0iLCJpbXBvcnQgeyBjaGVja0xpbmtFcnJvciwgY2hlY2tTaGFkZXJFcnJvciB9IGZyb20gJy4vY2hlY2suanMnXG5cbmNvbnN0IFRFTlNPUl9WRVJURVhfU0hBREVSID0gYFxuICAgIHByZWNpc2lvbiBoaWdocCBmbG9hdDtcbiAgICBhdHRyaWJ1dGUgdmVjMiBhX3Bvc2l0aW9uO1xuICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgZ2xfUG9zaXRpb24gPSB2ZWM0KGFfcG9zaXRpb24sIDAsIDEpO1xuICAgIH1cbmBcblxuXG5jb25zdCBVTklGT1JNX1NFVFRFUlMgPSB7IHZlYzQ6ICc0ZnYnLCB2ZWMzOiAnM2Z2JywgdmVjMjogJzJmdicsIGZsb2F0OiAnMWYnLFxuICAgICAgICAgICAgICAgICAgICAgICAgICBpdmVjNDogJzRpdicsIGl2ZWMzOiAnM2l2JywgaXZlYzI6ICcyaXYnLCBpbnQ6ICcxaScsXG4gICAgICAgICAgICAgICAgICAgICAgICAgIHNhbXBsZXIyRDogJzFpJyB9O1xuXG5leHBvcnQgZGVmYXVsdCBmdW5jdGlvbiBnZXRUZW5zb3JQcm9ncmFtKGdsLCBmcmFnbWVudFNoYWRlcil7XG4gICAgaWYoIWdsLl90ZW5zb3JQcm9ncmFtcykgZ2wuX3RlbnNvclByb2dyYW1zID0ge307XG4gICAgaWYoZnJhZ21lbnRTaGFkZXIgaW4gZ2wuX3RlbnNvclByb2dyYW1zKXtcbiAgICAgICAgcmV0dXJuIGdsLl90ZW5zb3JQcm9ncmFtc1tmcmFnbWVudFNoYWRlcl1cbiAgICB9XG4gICAgdmFyIHByb2dyYW0gPSBjcmVhdGVUZW5zb3JQcm9ncmFtKGdsLCBmcmFnbWVudFNoYWRlcik7XG4gICAgZ2wuX3RlbnNvclByb2dyYW1zW2ZyYWdtZW50U2hhZGVyXSA9IHByb2dyYW07XG4gICAgcmV0dXJuIHByb2dyYW07XG59XG5cbmZ1bmN0aW9uIGNyZWF0ZVRlbnNvclByb2dyYW0oZ2wsIGZyYWdtZW50U2hhZGVyKXtcbiAgICB2YXIgcHJvZ3JhbSA9IGNyZWF0ZVNoYWRlclByb2dyYW0oZ2wsIFRFTlNPUl9WRVJURVhfU0hBREVSLCBmcmFnbWVudFNoYWRlcik7XG4gICAgXG4gICAgZ2wudXNlUHJvZ3JhbShwcm9ncmFtKTtcbiAgICBiaW5kQXR0cmlidXRlQnVmZmVyKGdsLCBwcm9ncmFtKTtcblxuICAgIHZhciB1bmlmb3JtVHlwZXMgPSBleHRyYWN0VW5pZm9ybURlY2xhcmF0aW9ucyhmcmFnbWVudFNoYWRlciksXG4gICAgICAgIHVuaWZvcm1Mb2NzID0ge307XG5cbiAgICBmdW5jdGlvbiBhZGRVbmlmb3JtKG5hbWUsIHR5cGUpe1xuICAgICAgICB1bmlmb3JtTG9jc1tuYW1lXSA9IHsgbG9jOiBnbC5nZXRVbmlmb3JtTG9jYXRpb24ocHJvZ3JhbSwgbmFtZSksIHR5cGU6IHR5cGUgfVxuICAgIH1cblxuICAgIGZvcihsZXQgbmFtZSBpbiB1bmlmb3JtVHlwZXMpe1xuICAgICAgICBsZXQgdHlwZSA9IHVuaWZvcm1UeXBlc1tuYW1lXTtcbiAgICAgICAgaWYoKHR5cGUpIGluIFVOSUZPUk1fU0VUVEVSUyl7XG4gICAgICAgICAgICBhZGRVbmlmb3JtKG5hbWUsIHR5cGUpO1xuICAgICAgICB9ZWxzZSB0aHJvdyBuZXcgRXJyb3IoXCJVbmtub3duIHVuaWZvcm0gdHlwZSBcIiArIHR5cGUpO1xuICAgIH1cblxuICAgIGZ1bmN0aW9uIHNldFVuaWZvcm0obmFtZSwgdmFsdWUpe1xuICAgICAgICBpZighKG5hbWUgaW4gdW5pZm9ybUxvY3MpKXtcbiAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcihcIkNvdWxkIG5vdCBmaW5kIHVuaWZvcm0gXCIgKyBuYW1lKTtcbiAgICAgICAgfVxuICAgICAgICBnbFsndW5pZm9ybScgKyBVTklGT1JNX1NFVFRFUlNbdW5pZm9ybUxvY3NbbmFtZV0udHlwZV1dKHVuaWZvcm1Mb2NzW25hbWVdLmxvYywgdmFsdWUpXG4gICAgfVxuXG4gICAgcmV0dXJuIHtcbiAgICAgICAgcHJvZ3JhbTogcHJvZ3JhbSxcbiAgICAgICAgdW5pZm9ybUxvY3M6IHVuaWZvcm1Mb2NzLFxuICAgICAgICB1bmlmb3JtVHlwZXM6IHVuaWZvcm1UeXBlcyxcbiAgICAgICAgc2V0VW5pZm9ybTogc2V0VW5pZm9ybSxcbiAgICB9XG59XG5cblxuZXhwb3J0IGZ1bmN0aW9uIGJpbmRBdHRyaWJ1dGVCdWZmZXIoZ2wsIHByb2dyYW0pIHtcbiAgICBnbC5iaW5kQnVmZmVyKGdsLkFSUkFZX0JVRkZFUiwgZ2wuY3JlYXRlQnVmZmVyKCkpO1xuICAgIGdsLmJ1ZmZlckRhdGEoZ2wuQVJSQVlfQlVGRkVSLCBuZXcgRmxvYXQzMkFycmF5KFsgLTEsLTEsIDEsLTEsIC0xLCAxLCAxLCAxXSksIGdsLlNUQVRJQ19EUkFXKTtcblxuICAgIHZhciBwb3NpdGlvbkxvY2F0aW9uID0gZ2wuZ2V0QXR0cmliTG9jYXRpb24ocHJvZ3JhbSwgXCJhX3Bvc2l0aW9uXCIpO1xuICAgIGdsLmVuYWJsZVZlcnRleEF0dHJpYkFycmF5KHBvc2l0aW9uTG9jYXRpb24pO1xuICAgIGdsLnZlcnRleEF0dHJpYlBvaW50ZXIocG9zaXRpb25Mb2NhdGlvbiwgMiwgZ2wuRkxPQVQsIGZhbHNlLCAwLCAwKTtcbn1cblxuXG5mdW5jdGlvbiBleHRyYWN0VW5pZm9ybURlY2xhcmF0aW9ucyhzdHIpe1xuICAgIHZhciB1bmlmb3JtcyA9IHt9O1xuICAgIHN0ciA9IHN0ci5yZXBsYWNlKC8oKD86XFwvXFwqKD86W14qXXwoPzpcXCorW14qXFwvXSkpKlxcKitcXC8pfCg/OlxcL1xcLy4qKSkvZywgJycpXG4gICAgc3RyID0gc3RyLnJlcGxhY2UoL1xcL1xcLy4qXFxuL2csICcnKVxuICAgIHZhciBtLCByZSA9IC91bmlmb3JtXFxzKihbXFx3X10rKVxccyooW1xcd19dKykvZztcbiAgICB3aGlsZSAobSA9IHJlLmV4ZWMoc3RyKSkgdW5pZm9ybXNbbVsyXV0gPSBtWzFdO1xuICAgIHJldHVybiB1bmlmb3Jtcztcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVNoYWRlclByb2dyYW0oZ2wsIHZlcnRleFNvdXJjZSwgZnJhZ21lbnRTb3VyY2UpIHtcbiAgICB2YXIgdmVydGV4U2hhZGVyID0gY29tcGlsZVNoYWRlcihnbCwgdmVydGV4U291cmNlLCBnbC5WRVJURVhfU0hBREVSKTtcbiAgICB2YXIgZnJhZ21lbnRTaGFkZXIgPSBjb21waWxlU2hhZGVyKGdsLCBmcmFnbWVudFNvdXJjZSwgZ2wuRlJBR01FTlRfU0hBREVSKTtcblxuICAgIC8vIHZhciBkZWJ1ZyA9IGdsLmdldEV4dGVuc2lvbignV0VCR0xfZGVidWdfc2hhZGVycycpXG4gICAgLy8gY29uc29sZS5sb2coZGVidWcuZ2V0VHJhbnNsYXRlZFNoYWRlclNvdXJjZSh2ZXJ0ZXhTaGFkZXIpKTtcbiAgICAvLyBjb25zb2xlLmxvZyhkZWJ1Zy5nZXRUcmFuc2xhdGVkU2hhZGVyU291cmNlKGZyYWdtZW50U2hhZGVyKSk7XG5cbiAgICB2YXIgcHJvZ3JhbSA9IGdsLmNyZWF0ZVByb2dyYW0oKTtcbiAgICBnbC5hdHRhY2hTaGFkZXIocHJvZ3JhbSwgdmVydGV4U2hhZGVyKTtcbiAgICBnbC5hdHRhY2hTaGFkZXIocHJvZ3JhbSwgZnJhZ21lbnRTaGFkZXIpO1xuICAgIGdsLmxpbmtQcm9ncmFtKHByb2dyYW0pO1xuXG4gICAgLy8gaW50ZXJlc3RpbmdseSBlbm91Z2ggaXQgc2VlbXMgbGlrZSBTYWZhcmkgbmV2ZXIgZW1pdHNcbiAgICAvLyBhIHNoYWRlciBwcm9ncmFtIGxpbmsgZXJyb3IuIFxuICAgIGNoZWNrTGlua0Vycm9yKGdsLCBwcm9ncmFtLCBmcmFnbWVudFNvdXJjZSwgdmVydGV4U291cmNlKTtcblxuICAgIHJldHVybiBwcm9ncmFtO1xufVxuXG5cbmZ1bmN0aW9uIGNvbXBpbGVTaGFkZXIoZ2wsIHNoYWRlclNvdXJjZSwgc2hhZGVyVHlwZSkge1xuICAgIHZhciBzaGFkZXIgPSBnbC5jcmVhdGVTaGFkZXIoc2hhZGVyVHlwZSk7XG4gICAgZ2wuc2hhZGVyU291cmNlKHNoYWRlciwgc2hhZGVyU291cmNlKTtcbiAgICBnbC5jb21waWxlU2hhZGVyKHNoYWRlcik7XG4gICAgdmFyIHN1Y2Nlc3MgPSBnbC5nZXRTaGFkZXJQYXJhbWV0ZXIoc2hhZGVyLCBnbC5DT01QSUxFX1NUQVRVUyk7XG4gICAgY2hlY2tTaGFkZXJFcnJvcihnbCwgc2hhZGVyLCBzaGFkZXJTb3VyY2UsIHNoYWRlclR5cGUpXG4gICAgcmV0dXJuIHNoYWRlcjtcbn1cblxuXG4iLCJleHBvcnQgZnVuY3Rpb24gbm93KCkge1xuICAgIGlmICh0eXBlb2YgcGVyZm9ybWFuY2UgPT09ICd1bmRlZmluZWQnKSB7XG4gICAgICAgIHJldHVybiBEYXRlLm5vdygpXG4gICAgfSBlbHNlIHtcbiAgICAgICAgcmV0dXJuIHBlcmZvcm1hbmNlLm5vdygpO1xuICAgIH1cbn1cblxuZnVuY3Rpb24gZ2V0VGltZXIoZ2wpe1xuXHRpZihnbC5OT19QUk9GSUxFKSByZXR1cm47XG5cdGlmKHR5cGVvZiBnbC5USU1FUl9QT09MID09PSAndW5kZWZpbmVkJyl7XG5cdFx0dmFyIGV4dFRpbWVyID0gZ2wuZ2V0RXh0ZW5zaW9uKCdleHRfZGlzam9pbnRfdGltZXJfcXVlcnknKTtcblx0XHRpZighZXh0VGltZXIgfHwgIWV4dFRpbWVyLmNyZWF0ZVF1ZXJ5RVhUKXtcblx0XHRcdGdsLk5PX1BST0ZJTEUgPSB0cnVlO1xuXHRcdFx0cmV0dXJuO1xuXHRcdH1cblx0XHRnbC5USU1FUl9QT09MID0gY3JlYXRlVGltZXIoZ2wpXG5cdH1cblx0cmV0dXJuIGdsLlRJTUVSX1BPT0w7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBiZWdpblRpbWVyKGdsLCBpbmZvPXt9KXtcblx0dmFyIHRpbWVyID0gZ2V0VGltZXIoZ2wpO1xuXHRpZih0aW1lcil7XG5cdFx0dGltZXIuYmVnaW4oaW5mbylcblx0fVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZW5kVGltZXIoZ2wsIGNhbGxiYWNrKXtcblx0dmFyIHRpbWVyID0gZ2V0VGltZXIoZ2wpO1xuXHRpZih0aW1lcil7XG5cdFx0dGltZXIuZW5kKGNhbGxiYWNrKVxuXHR9ZWxzZSBpZihjYWxsYmFjayl7XG5cdFx0Y29uc29sZS53YXJuKFwiQ2FuIG5vdCB0cmlnZ2VyIGNhbGxiYWNrOiBpbXBsZW1lbnRhdGlvbiBkb2VzIG5vdCBzdXBwb3J0IGV4dF9kaXNqb2ludF90aW1lcl9xdWVyeVwiKVxuXHR9XG59XG5cbmZ1bmN0aW9uIGNyZWF0ZVRpbWVyKGdsKXtcdFxuXHR2YXIgZXh0VGltZXIgPSBnbC5nZXRFeHRlbnNpb24oJ2V4dF9kaXNqb2ludF90aW1lcl9xdWVyeScpO1xuXG5cdHZhciBxdWVyeVBvb2wgPSBbXVxuICAgIGZ1bmN0aW9uIGFsbG9jUXVlcnkgKCkge1xuICAgICAgICByZXR1cm4gcXVlcnlQb29sLnBvcCgpIHx8IGV4dFRpbWVyLmNyZWF0ZVF1ZXJ5RVhUKClcbiAgICB9XG4gICAgZnVuY3Rpb24gZnJlZVF1ZXJ5IChxdWVyeSkge1xuICAgICAgICBxdWVyeVBvb2wucHVzaChxdWVyeSlcbiAgICB9XG5cblx0dmFyIHBlbmRpbmdRdWVyaWVzID0gW11cblx0ZnVuY3Rpb24gYmVnaW5RdWVyeSAoaW5mbykge1xuXHRcdHZhciBxdWVyeSA9IGFsbG9jUXVlcnkoKVxuXHRcdGV4dFRpbWVyLmJlZ2luUXVlcnlFWFQoZXh0VGltZXIuVElNRV9FTEFQU0VEX0VYVCwgcXVlcnkpXG5cdFx0cGVuZGluZ1F1ZXJpZXMucHVzaChbcXVlcnksIGluZm9dKVxuXHR9XG5cblx0ZnVuY3Rpb24gZW5kUXVlcnkgKCkge1xuXHRcdGV4dFRpbWVyLmVuZFF1ZXJ5RVhUKGV4dFRpbWVyLlRJTUVfRUxBUFNFRF9FWFQpXG5cdH1cblxuXHRmdW5jdGlvbiBjYWxsYmFjayhpbmZvLCB0aW1lKXtcblx0XHR2YXIgZm4gPSBpbmZvLmNhbGxiYWNrO1xuXHRcdGluZm8uZ3B1VGltZSA9IHRpbWU7XG5cdFx0ZGVsZXRlIGluZm8uY2FsbGJhY2s7XG5cdFx0aWYoZm4pIGZuKGluZm8pO1xuXHR9XG5cblx0ZnVuY3Rpb24gbW9uaXRvclBlbmRpbmcoKXtcblx0XHRmb3IgKHZhciBpID0gMDsgaSA8IHBlbmRpbmdRdWVyaWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBcdFx0dmFyIHF1ZXJ5ID0gcGVuZGluZ1F1ZXJpZXNbaV1bMF1cbiAgICAgIFx0XHRpZiAoZXh0VGltZXIuZ2V0UXVlcnlPYmplY3RFWFQocXVlcnksIGV4dFRpbWVyLlFVRVJZX1JFU1VMVF9BVkFJTEFCTEVfRVhUKSkge1xuICAgICAgICBcdFx0dmFyIHF1ZXJ5VGltZSA9IGV4dFRpbWVyLmdldFF1ZXJ5T2JqZWN0RVhUKHF1ZXJ5LCBleHRUaW1lci5RVUVSWV9SRVNVTFRfRVhUKVxuICAgICAgICBcdFx0Y2FsbGJhY2socGVuZGluZ1F1ZXJpZXNbaV1bMV0sIHF1ZXJ5VGltZSAvIDFlNilcbiAgICAgICAgXHRcdGZyZWVRdWVyeShxdWVyeSlcbiAgICAgICAgXHRcdHBlbmRpbmdRdWVyaWVzLnNwbGljZShpLCAxKVxuICAgICAgICBcdFx0aS0tXG4gICAgICBcdFx0fVxuXHQgICAgfVxuXHR9XG5cblxuXHR2YXIgaXNQb2xsaW5nID0gZmFsc2U7XG5cdGZ1bmN0aW9uIGxvb3AoKXtcblx0XHRpZihwZW5kaW5nUXVlcmllcy5sZW5ndGggPiAwKXtcblx0XHRcdG1vbml0b3JQZW5kaW5nKClcblx0XHRcdHJlcXVlc3RBbmltYXRpb25GcmFtZShsb29wKVxuXHRcdH1lbHNle1xuXHRcdFx0aXNQb2xsaW5nID0gZmFsc2U7XG5cdFx0fVxuXHR9XG5cblx0dmFyIGN1cnJlbnRJbmZvID0gbnVsbDtcbiAgICByZXR1cm4ge1xuICAgIFx0YmVnaW4oaW5mbyA9IHt9KXtcbiAgICBcdFx0aWYoY3VycmVudEluZm8pIHRocm93IG5ldyBFcnJvcignYmVnaW5UaW1lciB3YXMgY2FsbGVkIGJlZm9yZSBwcmV2aW91cyBlbmRUaW1lcicpO1xuICAgIFx0XHRjdXJyZW50SW5mbyA9IGluZm9cbiAgICBcdFx0aW5mby5jcHVTdGFydFRpbWUgPSBub3coKTtcbiAgICBcdFx0YmVnaW5RdWVyeShjdXJyZW50SW5mbylcbiAgICBcdH0sXG5cbiAgICBcdGVuZChmbil7XG4gICAgXHRcdGN1cnJlbnRJbmZvLmNwdVRpbWUgPSBub3coKSAtIGN1cnJlbnRJbmZvLmNwdVN0YXJ0VGltZVxuICAgIFx0XHRkZWxldGUgY3VycmVudEluZm8uY3B1U3RhcnRUaW1lO1xuICAgIFx0XHRjdXJyZW50SW5mby5jYWxsYmFjayA9IGZuO1xuICAgIFx0XHRjdXJyZW50SW5mbyA9IG51bGw7XG4gICAgXHRcdGVuZFF1ZXJ5KClcblxuICAgIFx0XHRpZihpc1BvbGxpbmcgPT09IGZhbHNlKXtcbiAgICBcdFx0XHRpc1BvbGxpbmcgPSB0cnVlO1xuICAgIFx0XHRcdHJlcXVlc3RBbmltYXRpb25GcmFtZShsb29wKVxuICAgIFx0XHR9XG4gICAgXHR9XG4gICAgfVxufSIsIi8vIFROU0wgKHByb25vdW5jZWQgdGluc2VsKVxuLy8gaXMgYSBkb21haW4gc3BlY2lmaWMgbGFuZ3VhZ2UgYmFzZWQgb24gR0xTTFxuLy8gZm9yIGhlbHBpbmcgd2l0aCB0aGUgd3JpdGluZyBjb2RlIHRoYXRcbi8vIGNvbXB1dGVzIHdpdGggdGVuc29ycy4gXG5cbi8vIEEgbGltaXRhdGlvbiBvZiBHTFNMIGlzIHRoYXQgdGhlIGNvbmRpdGlvblxuLy8gb2YgYW55IGxvb3AgaGFzIHRvIGJlIHN0YXRpY2FsbHkga25vd24gXG4vLyAoZS5nLiBjb3VudGVycyB1cCB0byBhIGZpeGVkIGNvbnN0YW50XG4vLyB2YWx1ZSkgd2hpY2ggaXMgcHJvYmxlbWF0aWMgaWYgd2Ugd2FudFxuLy8gdG8gd3JpdGUgZ2VuZXJhbCBjb2RlIHRoYXQgZGVwZW5kcyBvblxuLy8gdGhlIHNpemUgb2YgdGhlIGlucHV0IHRlbnNvcnNcblxuLy8gVE5TTCBhZGRzIHRoZSBmb2xsb3dpbmcgc3ludGF4OlxuLy8gICAgICAjKGltYWdlLnNoYXBlKVxuLy8gd2hpY2ggd2lsbCBiZSByZXBsYWNlZCB3aXRoIGFuIGl2ZWM0XG4vLyBjb250YWluaW5nIHRoZSBzaGFwZSBvZiB0aGUgaW5wdXQgdGVuc29yIFwiaW1hZ2VcIlxuLy8gYXV0b21hdGljYWxseVxuXG5leHBvcnQgZGVmYXVsdCBmdW5jdGlvbiBUTlNMKHN0cil7XG4gICAgaWYodHlwZW9mIHN0ciAhPSAnc3RyaW5nJykgXG4gICAgICAgIHRocm93IG5ldyBFcnJvcignVE5TTCBzaGFkZXIgcHJlcHJvY2Vzc29yIG9ubHkgYWNjZXB0cyBzdHJpbmdzJyk7XG4gICAgXG4gICAgcmV0dXJuIGZ1bmN0aW9uKHVuaWZvcm1zLCBvdXRwdXQpe1xuICAgICAgICByZXR1cm4gc3RyXG4gICAgICAgIC8vIGNvbW1lbnQgb3V0IHRoZSB0ZW5zb3Igc3RydWN0IGRlZmluaXRpb25zXG4gICAgICAgIC5yZXBsYWNlKC91bmlmb3JtXFxzKlRlbnNvclxccyooW1xcd19dKylcXHMqOy9nLCAnLyogKFRlbnNvciAkMSkgKi8nKVxuXG4gICAgICAgIC8vIHRoaXMgaXMgdGhlIG1hY3JvIHN5bnRheFxuICAgICAgICAucmVwbGFjZSgvXFwjXFwoKFtcXHdcXC5cXHNdKylcXCkvZywgZnVuY3Rpb24oYWxsLCBib2R5KXtcbiAgICAgICAgICAgIHZhciBvYmogPSB1bmlmb3JtcztcbiAgICAgICAgICAgIGZvcihsZXQgcGFydCBvZiBib2R5LnNwbGl0KCcuJykpXG4gICAgICAgICAgICAgICAgb2JqID0gb2JqW3BhcnQudHJpbSgpXTtcbiAgICAgICAgICAgIGlmKHR5cGVvZiBvYmogPT0gJ251bWJlcicpe1xuICAgICAgICAgICAgICAgIHJldHVybiBvYmoudG9TdHJpbmcoKVxuICAgICAgICAgICAgfWVsc2UgaWYoQXJyYXkuaXNBcnJheShvYmopICYmIG9iai5sZW5ndGggPD0gNCAmJiBvYmoubGVuZ3RoID4gMSl7XG4gICAgICAgICAgICAgICAgcmV0dXJuIChvYmouZXZlcnkoTnVtYmVyLmlzSW50ZWdlcikgPyAnaScgOiAnJykgKyBcbiAgICAgICAgICAgICAgICAgICAgJ3ZlYycgKyBvYmoubGVuZ3RoICsgJygnICsgb2JqLmpvaW4oJywnKSArICcpJ1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdDYW4gbm90IGlubGluZSBleHByZXNzaW9uICcgKyBib2R5KTtcbiAgICAgICAgfSlcbiAgICAgICAgLy8gdGVuc29yLnJlYWQ0KHgsIDApID0+IHRlbnNvci5yZWFkNChpdmVjNCh4LCAwLCAwLCAwKSlcbiAgICAgICAgLy8gdGhpcyB0cmFuc2Zvcm1hdGlvbiB0YWtlcyBwbGFjZSB3aGVuIHRoZXJlIGFyZSAyIG9yIG1vcmUgYXJndW1lbnRzXG4gICAgICAgIC8vIGFzIG90aGVyd2lzZSBpdCdzIG5vdCBwb3NzaWJsZSB0byBzdGF0aWNhbGx5IGRldGVybWluZSB3aGV0aGVyIHggaXNcbiAgICAgICAgLy8gb2YgdHlwZSBpdmVjNCBvciBhIG51bWJlclxuICAgICAgICAucmVwbGFjZSgvXFxiKFxcdyspXFxzKlxcLlxccyoocmVhZDQ/KVxcYlxccypcXCgoW15cXChcXCldKylcXCkvZywgZnVuY3Rpb24oYWxsLCBuYW1lLCBwcm9wLCBhcmcpe1xuICAgICAgICAgICAgaWYobmFtZSBpbiB1bmlmb3JtcyAmJiB1bmlmb3Jtc1tuYW1lXS5zaGFwZSl7XG4gICAgICAgICAgICAgICAgdmFyIHBhcnRzID0gYXJnLnNwbGl0KCcsJyksXG4gICAgICAgICAgICAgICAgICAgIHBhZGRlZCA9IHBhcnRzLmNvbmNhdChbJzAnLCAnMCcsICcwJywgJzAnXS5zbGljZSgwLCA0IC0gcGFydHMubGVuZ3RoKSk7XG4gICAgICAgICAgICAgICAgaWYocGFydHMubGVuZ3RoIDwgMiB8fCBwYXJ0cy5sZW5ndGggPiA0KSByZXR1cm4gYWxsO1xuICAgICAgICAgICAgICAgIHZhciB2ZWMgPSAnaXZlYzQoJyArIHBhZGRlZC5qb2luKCcsJykgKyAnKSc7XG4gICAgICAgICAgICAgICAgcmV0dXJuIG5hbWUgKyAnXycgKyBwcm9wICsgJygnICsgdmVjICsgJyknO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmV0dXJuIGFsbDtcbiAgICAgICAgfSlcblxuICAgICAgICAvLyB0ZW5zb3Iuc2hhcGUgPT4gdGVuc29yX3NoYXBlXG4gICAgICAgIC5yZXBsYWNlKC9cXGIoXFx3KylcXHMqXFwuXFxzKihcXHcrKVxcYi9nLCBmdW5jdGlvbihhbGwsIG5hbWUsIHByb3Ape1xuICAgICAgICAgICAgaWYobmFtZSBpbiB1bmlmb3JtcyAmJiB1bmlmb3Jtc1tuYW1lXS5zaGFwZSl7XG4gICAgICAgICAgICAgICAgcmV0dXJuIG5hbWUgKyAnXycgKyBwcm9wO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcmV0dXJuIGFsbDtcbiAgICAgICAgfSlcbiAgICAgICAgLy8gLnJlcGxhY2UoL1xcI1xccyooXFx3KylcXHMqXFxbKC4qPylcXF0vZywgZnVuY3Rpb24oYWxsLCB0ZW5zb3IsIGJvZHkpe1xuICAgICAgICAvLyAgICAgcmV0dXJuIHRlbnNvciArICdfcmVhZChpdmVjNCgnICsgYm9keSArICcpKSdcbiAgICAgICAgLy8gfSlcbiAgICB9XG59XG4iLCJpbXBvcnQgeyBtYWtlVGV4dHVyZSwgbWFrZUZyYW1lQnVmZmVyLCBjaGVja1JlbmRlckZsb2F0IH0gZnJvbSAnLi9oZWxwZXJzLmpzJ1xuaW1wb3J0IEZvcm1hdHMgZnJvbSAnLi4vZm9ybWF0L2luZGV4LmpzJ1xuXG4vLyBUaGUgdGVuc29yIGZvcm1hdCBpcyBhIEpTT04gb2JqZWN0IHRoYXQgc3BlY2lmaWVzIGhvdyBcbi8vIHRoZSB0ZW5zb3IgaXMgcmVwcmVzZW50ZWQgYXMgYSB0ZXh0dXJlXG4vLyBpdCBjb25zaXN0cyBvZiBzZXZlcmFsIGtleXM6XG5cbi8vICAgICB0eXBlOiB1aW50OCB8IGZsb2F0MzJcbi8vICAgICBkZW5zaXR5OiA0OjQgfCAxOjRcbi8vICAgICBwYWNrOiBzdHJpZGUgfCB0aWxlXG4vLyAgICAgY29kZWM6IFxuLy9cdFx0XHRzb2Z0ZmxvYXQgfCBmaXhudW0gKDE6NClcbi8vICAgICAgICAgIHJhdyB8IGxpbnF1YW50ICg0OjQpXG5cbmV4cG9ydCBkZWZhdWx0IGNsYXNzIEJhc2VUZW5zb3Ige1xuXHQvLyB3ZSBhcmVudCB1c2luZyBhIGNvbnN0cnVjdG9yIGJlY2F1c2Ugd2Ugd2FudCB0byBiZSBhYmxlIHRvIHJ1blxuXHQvLyB0aGlzIGluc3RhbmNlb2YgT3V0cHV0VGVuc29yIGZyb20gd2l0aGluIHRoZSBUZW5zb3IgY29uc3RydWN0b3Jcblx0XG5cdF9pbml0KGdsLCBmb3JtYXQsIHNoYXBlLCBkYXRhKXtcblx0XHQvLyB2YWxpZGF0ZSBnbGNvbnRleHRcblx0XHRpZighZ2wuY3JlYXRlVGV4dHVyZSkgdGhyb3cgbmV3IEVycm9yKCdJbnZhbGlkIFdlYkdMUmVuZGVyaW5nQ29udGV4dCcpO1xuXHRcdHRoaXMuZ2wgPSBnbDtcblxuXHRcdC8vIHZhbGlkYXRlIHNoYXBlXG5cdFx0aWYoIUFycmF5LmlzQXJyYXkoc2hhcGUpKSB0aHJvdyBuZXcgRXJyb3IoXCJzaGFwZSBtdXN0IGJlIEFycmF5XCIpO1xuXHRcdGlmKHNoYXBlLmxlbmd0aCA+IDQpIHRocm93IG5ldyBFcnJvcihcIlRlbnNvciBtdXN0IGhhdmUgZGltZW5zaW9uIDw9IDRcIik7XG4gICAgICAgIGlmKHNoYXBlLnNvbWUoayA9PiAhaXNGaW5pdGUoaykgfHwgayA8IDEgfHwgIU51bWJlci5pc0ludGVnZXIoaykpKSBcbiAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcignSW52YWxpZCBzaGFwZTogJyArIHNoYXBlKTtcbiAgICAgICAgc2hhcGUgPSBzaGFwZS5jb25jYXQoWzEsIDEsIDEsIDFdKS5zbGljZSgwLCA0KVxuXHRcdHRoaXMuc2hhcGUgPSBzaGFwZTtcblx0XHRcblx0XHQvLyB2YWxpZGF0ZSBmb3JtYXRcblx0XHRpZighWydmbG9hdDMyJywgJ3VpbnQ4J10uaW5jbHVkZXMoZm9ybWF0LnR5cGUpKVxuXHRcdFx0dGhyb3cgbmV3IEVycm9yKCdmb3JtYXQudHlwZSBtdXN0IGJlIHVpbnQ4IG9yIGZsb2F0MzInKTtcblx0XHRpZihmb3JtYXQuZGVuc2l0eSBpbiBGb3JtYXRzKXtcblx0XHRcdGxldCBmZCA9IEZvcm1hdHNbZm9ybWF0LmRlbnNpdHldO1xuXHRcdFx0aWYoIShmb3JtYXQucGFjayBpbiBmZC5wYWNrKSkgXG5cdFx0XHRcdHRocm93IG5ldyBFcnJvcignZm9ybWF0LnBhY2sgbXVzdCBiZSAnICsgT2JqZWN0LmtleXMoZmQucGFjaykuam9pbignIG9yICcpKTtcblx0XHRcdGlmKCEoZm9ybWF0LmNvZGVjIGluIGZkLmNvZGVjKSkgXG5cdFx0XHRcdHRocm93IG5ldyBFcnJvcignZm9ybWF0LmNvZGVjIG11c3QgYmUgJyArIE9iamVjdC5rZXlzKGZkLmNvZGVjKS5qb2luKCcgb3IgJykpO1xuXHRcdH1lbHNlIHRocm93IG5ldyBFcnJvcignZm9ybWF0LmRlbnNpdHkgbXVzdCBiZSAnICsgT2JqZWN0LmtleXMoRm9ybWF0cykuam9pbignIG9yICcpKTtcblxuXHRcdHRoaXMuZm9ybWF0ID0gZm9ybWF0O1xuXG5cdFx0Ly8gY2FsY3VsYXRlIHRleHR1cmUgc2l6ZVxuXHRcdHRoaXMuaW5mbyA9IE9iamVjdC5hc3NpZ24oe30sXG5cdFx0XHR0aGlzLl9mb3JtYXQucGFjay5pbml0KHNoYXBlLCBmb3JtYXQpLFxuXHRcdFx0dGhpcy5fZm9ybWF0LmNvZGVjLmluaXQoc2hhcGUsIGZvcm1hdClcblx0XHQpO1xuXHRcdGlmKCF0aGlzLmluZm8udGV4U2l6ZSkgdGhyb3cgbmV3IEVycm9yKCdGb3JtYXQgZGlkIG5vdCB5aWVsZCB0ZXhTaXplJyk7XG5cblx0XHQvLyBpbml0aWFsaXplIHRleHR1cmVcblx0XHR0aGlzLnRleCA9IG1ha2VUZXh0dXJlKGdsKTtcblx0XHR0aGlzLnVwZGF0ZShkYXRhKVxuXHR9XG5cdF91cGRhdGUoZGF0YSl7XG5cdFx0aWYoZGF0YSAhPT0gbnVsbCl7XG5cdFx0XHRpZih0aGlzLmZvcm1hdC50eXBlID09PSAndWludDgnKXtcblx0XHRcdFx0aWYoQXJyYXkuaXNBcnJheShkYXRhKSB8fCBkYXRhIGluc3RhbmNlb2YgVWludDhDbGFtcGVkQXJyYXkpXG5cdFx0XHRcdFx0ZGF0YSA9IG5ldyBVaW50OEFycmF5KGRhdGEpO1xuXHRcdFx0XHRpZighKGRhdGEgaW5zdGFuY2VvZiBVaW50OEFycmF5KSlcblx0XHRcdFx0XHR0aHJvdyBuZXcgRXJyb3IoJ2RhdGEgbXVzdCBiZSBVaW50OEFycmF5Jyk7XG5cdFx0XHR9ZWxzZSBpZih0aGlzLmZvcm1hdC50eXBlID09PSAnZmxvYXQzMicpe1xuXHRcdFx0XHRpZihBcnJheS5pc0FycmF5KGRhdGEpIHx8IGRhdGEgaW5zdGFuY2VvZiBGbG9hdDY0QXJyYXkpXG5cdFx0XHRcdFx0ZGF0YSA9IG5ldyBGbG9hdDMyQXJyYXkoZGF0YSk7XG5cdFx0XHRcdGlmKCEoZGF0YSBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkpXG5cdFx0XHRcdFx0dGhyb3cgbmV3IEVycm9yKCdkYXRhIG11c3QgYmUgRmxvYXQzMkFycmF5Jyk7XG5cdFx0XHR9ZWxzZSB0aHJvdyBuZXcgRXJyb3IoJ1R5cGUgbXVzdCBiZSB1aW50OCBvciBmbG9hdDMyJyk7XG5cdFx0XHRpZihkYXRhLmxlbmd0aCAhPT0gdGhpcy5pbmZvLnRleFNpemVbMF0gKiB0aGlzLmluZm8udGV4U2l6ZVsxXSAqIDQpXG5cdFx0XHRcdHRocm93IG5ldyBFcnJvcignZGF0YSBpcyB0aGUgd3JvbmcgbGVuZ3RoJyk7XG5cdFx0fVxuXHRcdC8vIGlmKGRhdGEpIGNvbnNvbGUubG9nKCdfdXBkYXRlJywgZGF0YSk7XG5cdFx0dmFyIGdsID0gdGhpcy5nbDtcbiAgICAgICAgZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGhpcy50ZXgpO1xuICAgICAgICBnbC50ZXhJbWFnZTJEKGdsLlRFWFRVUkVfMkQsIDAsIGdsLlJHQkEsIFxuICAgICAgICBcdHRoaXMuaW5mby50ZXhTaXplWzBdLCB0aGlzLmluZm8udGV4U2l6ZVsxXSwgMCwgZ2wuUkdCQSwgXG4gICAgICAgIFx0dGhpcy5mb3JtYXQudHlwZSA9PSAndWludDgnID8gZ2wuVU5TSUdORURfQllURSA6IGdsLkZMT0FULCBkYXRhKTtcblx0fVxuXG5cdHVwZGF0ZShkYXRhKXtcblx0XHRpZighZGF0YSkgcmV0dXJuIHRoaXMuX3VwZGF0ZShudWxsKTtcblx0XHRpZihkYXRhLnNoYXBlKSByZXR1cm4gdGhpcy5fdXBkYXRlKFxuXHRcdFx0dGhpcy5fZm9ybWF0LnBhY2sucGFjayh0aGlzLmluZm8sIGRhdGEsIHRoaXMuX2Zvcm1hdC5jb2RlYy5lbmNvZGUsIHRoaXMuZm9ybWF0KSk7XG5cdFx0aWYodGhpcy50eXBlICE9ICd1aW50OCcpIGNvbnNvbGUud2FybignQ2FsbGluZyB1cGRhdGUgd2l0aCByYXcgVHlwZWRBcnJheSBtYXkgbm90IHdvcmsgYWNyb3NzIGFsbCBicm93c2Vycy4nKTtcblx0XHRyZXR1cm4gdGhpcy5fdXBkYXRlKGRhdGEpO1xuXHR9XG5cblx0Z2V0IF9mb3JtYXQoKXtcblx0XHRyZXR1cm4ge1xuXHRcdFx0cGFjazogRm9ybWF0c1t0aGlzLmZvcm1hdC5kZW5zaXR5XS5wYWNrW3RoaXMuZm9ybWF0LnBhY2tdLFxuXHRcdFx0Y29kZWM6IEZvcm1hdHNbdGhpcy5mb3JtYXQuZGVuc2l0eV0uY29kZWNbdGhpcy5mb3JtYXQuY29kZWNdLFxuXHRcdFx0YWN0aXZhdGlvbnM6IEZvcm1hdHNbdGhpcy5mb3JtYXQuZGVuc2l0eV0uYWN0aXZhdGlvbnMsXG5cdFx0XHRyZWFkX3NoaW06IEZvcm1hdHNbdGhpcy5mb3JtYXQuZGVuc2l0eV0ucmVhZF9zaGltLFxuXHRcdFx0d3JpdGVfc2hpbTogRm9ybWF0c1t0aGlzLmZvcm1hdC5kZW5zaXR5XS53cml0ZV9zaGltXG5cdFx0fVxuXHR9XG5cbiAgICBkZXN0cm95KCl7IHRoaXMuZ2wuZGVsZXRlVGV4dHVyZSh0aGlzLnRleCkgfVxufSIsImltcG9ydCB7IGJpbmRBdHRyaWJ1dGVCdWZmZXIsIGNyZWF0ZVNoYWRlclByb2dyYW0gfSBmcm9tICcuLi9ydW50aW1lL3Byb2dyYW0uanMnXG5pbXBvcnQgeyBtYWtlRnJhbWVCdWZmZXIsIG1ha2VUZXh0dXJlIH0gZnJvbSAnLi9oZWxwZXJzLmpzJ1xuXG5leHBvcnQgZGVmYXVsdCBmdW5jdGlvbiBydW5GZWF0dXJlVGVzdHMoZ2wpe1xuICAgIFxuICAgIGlmKCFnbC5GTE9BVF9URVhUVVJFU19URVNURUQgJiYgIWdsLk5PX0ZMT0FUX1RFWFRVUkVTKXtcbiAgICAgICAgaWYoIWdsLmdldEV4dGVuc2lvbignT0VTX3RleHR1cmVfZmxvYXQnKSl7XG4gICAgICAgICAgICBjb25zb2xlLmluZm8oXCJUaGlzIGJyb3dzZXIgZG9lcyBub3Qgc2VlbSB0byBzdXBwb3J0IE9FU190ZXh0dXJlX2Zsb2F0LiBcIlxuICAgICAgICAgICAgICAgICsgXCJVc2luZyBmbG9hdCBjb2RlYyB3b3JrYXJvdW5kIGZyb20gbm93IG9uLlwiKVxuICAgICAgICAgICAgZ2wuTk9fRkxPQVRfVEVYVFVSRVMgPSB0cnVlO1xuICAgICAgICB9XG4gICAgICAgIGdsLkZMT0FUX1RFWFRVUkVTX1RFU1RFRCA9IHRydWU7XG4gICAgfVxuXG4gICAgaWYoIWdsLk5PX0ZMT0FUX1RFWFRVUkVTKXtcbiAgICAgICAgaWYoIWdsLlJFTkRFUl9GTE9BVF9URVNURUQgJiYgIWdsLk5PX1JFTkRFUl9GTE9BVCl7XG4gICAgICAgICAgICBpZighdGVzdFJlbmRlckZsb2F0KGdsKSl7XG4gICAgICAgICAgICAgICAgY29uc29sZS5pbmZvKFwiVGhpcyBicm93c2VyIHN1cHBvcnRzIE9FU190ZXh0dXJlX2Zsb2F0LCBcIiArIFxuICAgICAgICAgICAgICAgICAgICBcImJ1dCBjYW4gbm90IHJlbmRlciB0byBmbG9hdGluZyB0ZXh0dXJlcy4gXCIgKyBcbiAgICAgICAgICAgICAgICAgICAgXCJVc2luZyBmbG9hdCBjb2RlYyB3b3JrYXJvdW5kIGZvciBvdXRwdXQgdGVuc29ycyBmcm9tIG5vdyBvbi5cIilcbiAgICAgICAgICAgICAgICBnbC5OT19SRU5ERVJfRkxPQVQgPSB0cnVlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgZ2wuUkVOREVSX0ZMT0FUX1RFU1RFRCA9IHRydWU7XG4gICAgICAgIH1cblxuICAgICAgICBpZighZ2wuUkVBRF9GTE9BVF9URVNURUQgJiYgIWdsLk5PX1JFQURfRkxPQVQgJiYgIWdsLk5PX1JFQURfRkxPQVQpe1xuICAgICAgICAgICAgaWYoIXRlc3RSZWFkRmxvYXQoZ2wpKXtcbiAgICAgICAgICAgICAgICBjb25zb2xlLmluZm8oXCJUaGlzIGJyb3dzZXIgc3VwcG9ydHMgT0VTX3RleHR1cmVfZmxvYXQsIFwiICsgXG4gICAgICAgICAgICAgICAgICAgIFwiY2FuIHJlbmRlciB0byBmbG9hdGluZyBwb2ludCB0ZXh0dXJlcywgYnV0IGNhbiBub3QgXCIgK1xuICAgICAgICAgICAgICAgICAgICBcInJlYWQgaW50byBhIEZsb2F0MzJBcnJheSBidWZmZXIuIFVzaW5nIGZsb2F0IGNvZGVjIFwiICtcbiAgICAgICAgICAgICAgICAgICAgXCJ3b3JrYXJvdW5kIGZvciByZWFkaW5nIGZyb20gb3V0cHV0IHRlbnNvcnMgZnJvbSBub3cgb24uXCIpXG4gICAgICAgICAgICAgICAgZ2wuTk9fUkVBRF9GTE9BVCA9IHRydWU7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBnbC5SRUFEX0ZMT0FUX1RFU1RFRCA9IHRydWU7XG4gICAgICAgIH1cbiAgICB9XG5cblxufVxuXG5cbmNvbnN0IENIRUNLX0ZMT0FUX1ZFUlRFWCA9IGBcbiAgICBhdHRyaWJ1dGUgdmVjMiBhX3Bvc2l0aW9uO1xuICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgZ2xfUG9zaXRpb24gPSB2ZWM0KGFfcG9zaXRpb24sIDAsIDEpO1xuICAgIH1cbmBcbmNvbnN0IENIRUNLX0ZMT0FUX0ZSQUdNRU5UID0gYFxuICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgZ2xfRnJhZ0NvbG9yID0gdmVjNCgzLjE0MTU5LCAtMi43MTgyOCwgMS42MTgyOCwgNDIpO1xuICAgIH1cbmA7XG5cbi8vIHNvbWUgYnJvd3NlcnMgKGUuZy4gbW9iaWxlIHNhZmFyaSkgYXJlIGNhcGFibGUgb2YgaW5pdGlhbGl6aW5nIGZsb2F0aW5nIFxuLy8gcG9pbnQgdGV4dHVyZXMgYnV0IHVuYWJsZSB0byB3cml0ZSB0byB0aGVtLiBUaGUgb25seSB3YXkgb2YgZmluZGluZyB0aGlzXG4vLyBvdXQgaXMgYnkgdHJ5aW5nIHRvIHJlbmRlciB0byBhIGZsb2F0aW5nIHBvaW50IHRleHR1cmUgYW5kIG5vdGljaW5nXG4vLyB0aGUgaW52YWxpZCBmcmFtZWJ1ZmZlciBzdGF0dXMuXG5cbmV4cG9ydCBmdW5jdGlvbiB0ZXN0UmVuZGVyRmxvYXQoZ2wpe1xuICAgIHZhciB0ZXggPSBtYWtlVGV4dHVyZShnbClcbiAgICBnbC50ZXhJbWFnZTJEKGdsLlRFWFRVUkVfMkQsIDAsIGdsLlJHQkEsIDEwLCAxMCwgMCwgZ2wuUkdCQSwgZ2wuRkxPQVQsIG51bGwpO1xuICAgIHZhciBmYm8gPSBtYWtlRnJhbWVCdWZmZXIoZ2wsIHRleCk7XG5cbiAgICB2YXIgcHJvZ3JhbSA9IGNyZWF0ZVNoYWRlclByb2dyYW0oZ2wsIENIRUNLX0ZMT0FUX1ZFUlRFWCwgQ0hFQ0tfRkxPQVRfRlJBR01FTlQpO1xuICAgIGdsLnVzZVByb2dyYW0ocHJvZ3JhbSk7XG4gICAgYmluZEF0dHJpYnV0ZUJ1ZmZlcihnbCwgcHJvZ3JhbSk7XG5cbiAgICBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIGZibyk7XG4gICAgZ2wudmlld3BvcnQoMCwgMCwgMTAsIDEwKTtcbiAgICBnbC5kcmF3QXJyYXlzKGdsLlRSSUFOR0xFX1NUUklQLCAwLCA0KTtcblxuICAgIHZhciBzdGF0dXMgPSBnbC5jaGVja0ZyYW1lYnVmZmVyU3RhdHVzKGdsLkZSQU1FQlVGRkVSKTtcbiAgICBnbC5kZWxldGVUZXh0dXJlKHRleClcbiAgICBnbC5kZWxldGVGcmFtZWJ1ZmZlcihmYm8pXG4gICAgZ2wuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKVxuXG4gICAgcmV0dXJuIHN0YXR1cyA9PSBnbC5GUkFNRUJVRkZFUl9DT01QTEVURTtcbn1cblxuXG5mdW5jdGlvbiB0ZXN0UmVhZEZsb2F0KGdsKXtcbiAgICB2YXIgdGV4ID0gbWFrZVRleHR1cmUoZ2wpXG4gICAgZ2wudGV4SW1hZ2UyRChnbC5URVhUVVJFXzJELCAwLCBnbC5SR0JBLCAxMCwgMTAsIDAsIGdsLlJHQkEsIGdsLkZMT0FULCBudWxsKTtcbiAgICB2YXIgZmJvID0gbWFrZUZyYW1lQnVmZmVyKGdsLCB0ZXgpO1xuXG4gICAgdmFyIHByb2dyYW0gPSBjcmVhdGVTaGFkZXJQcm9ncmFtKGdsLCBDSEVDS19GTE9BVF9WRVJURVgsIENIRUNLX0ZMT0FUX0ZSQUdNRU5UKTtcbiAgICBnbC51c2VQcm9ncmFtKHByb2dyYW0pO1xuICAgIGJpbmRBdHRyaWJ1dGVCdWZmZXIoZ2wsIHByb2dyYW0pO1xuXG4gICAgZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBmYm8pO1xuICAgIGdsLnZpZXdwb3J0KDAsIDAsIDEwLCAxMCk7XG4gICAgZ2wuZHJhd0FycmF5cyhnbC5UUklBTkdMRV9TVFJJUCwgMCwgNCk7XG5cbiAgICB2YXIgc2l6ZSA9IFszLCAzXTtcbiAgICB2YXIgcGl4ZWxzID0gcGl4ZWxzID0gbmV3IEZsb2F0MzJBcnJheShzaXplWzBdICogc2l6ZVsxXSAqIDQpXG4gICAgZ2wucmVhZFBpeGVscygwLCAwLCBzaXplWzBdLCBzaXplWzFdLCBnbC5SR0JBLCBnbC5GTE9BVCwgcGl4ZWxzKTtcblxuICAgIGdsLmRlbGV0ZVRleHR1cmUodGV4KVxuICAgIGdsLmRlbGV0ZUZyYW1lYnVmZmVyKGZibylcbiAgICBnbC5kZWxldGVQcm9ncmFtKHByb2dyYW0pXG5cblxuICAgIHZhciB0b3RhbF9lcnJvciA9IE1hdGguYWJzKHBpeGVsc1swXSAtIDMuMTQxNTkpICtcbiAgICAgICAgICAgIE1hdGguYWJzKHBpeGVsc1sxXSArIDIuNzE4MjgpICtcbiAgICAgICAgICAgIE1hdGguYWJzKHBpeGVsc1syXSAtIDEuNjE4MjgpICtcbiAgICAgICAgICAgIE1hdGguYWJzKHBpeGVsc1szXSAtIDQyKTtcblxuICAgIHJldHVybiB0b3RhbF9lcnJvciA8IDAuMDE7XG59XG4iLCJleHBvcnQgZnVuY3Rpb24gbWFrZUZyYW1lQnVmZmVyKGdsLCB0ZXh0dXJlKXtcbiAgICB2YXIgZnJhbWVidWZmZXIgPSBnbC5jcmVhdGVGcmFtZWJ1ZmZlcigpO1xuICAgIGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgZnJhbWVidWZmZXIpO1xuICAgIGdsLmZyYW1lYnVmZmVyVGV4dHVyZTJEKGdsLkZSQU1FQlVGRkVSLCBnbC5DT0xPUl9BVFRBQ0hNRU5UMCwgZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSwgMCk7XG4gICAgcmV0dXJuIGZyYW1lYnVmZmVyO1xufVxuXG5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlVGV4dHVyZShnbCl7XG4gICAgdmFyIHRleHR1cmUgPSBnbC5jcmVhdGVUZXh0dXJlKCk7XG4gICAgZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSk7XG4gICAgZ2wudGV4UGFyYW1ldGVyaShnbC5URVhUVVJFXzJELCBnbC5URVhUVVJFX1dSQVBfUywgZ2wuQ0xBTVBfVE9fRURHRSk7XG4gICAgZ2wudGV4UGFyYW1ldGVyaShnbC5URVhUVVJFXzJELCBnbC5URVhUVVJFX1dSQVBfVCwgZ2wuQ0xBTVBfVE9fRURHRSk7XG4gICAgZ2wudGV4UGFyYW1ldGVyaShnbC5URVhUVVJFXzJELCBnbC5URVhUVVJFX01JTl9GSUxURVIsIGdsLk5FQVJFU1QpO1xuICAgIGdsLnRleFBhcmFtZXRlcmkoZ2wuVEVYVFVSRV8yRCwgZ2wuVEVYVFVSRV9NQUdfRklMVEVSLCBnbC5ORUFSRVNUKTtcblxuICAgIHJldHVybiB0ZXh0dXJlO1xufVxuXG4iLCJpbXBvcnQgQmFzZVRlbnNvciBmcm9tICcuL2Jhc2UuanMnO1xuaW1wb3J0IHNob3dUZXh0dXJlIGZyb20gJy4vc2hvdy5qcydcbmltcG9ydCBydW5GZWF0dXJlVGVzdHMgZnJvbSAnLi9mZWF0dXJlLmpzJ1xuaW1wb3J0IHsgbWFrZVRleHR1cmUsIG1ha2VGcmFtZUJ1ZmZlciB9IGZyb20gJy4vaGVscGVycy5qcydcbmltcG9ydCB7IFJ1biwgQ29tcGlsZSB9IGZyb20gJy4uL3J1bnRpbWUvaW5kZXguanMnXG5pbXBvcnQgbmRzaG93IGZyb20gJ25kYXJyYXktc2hvdydcbmltcG9ydCBuZGFycmF5IGZyb20gJ25kYXJyYXknXG5cbmV4cG9ydCBjbGFzcyBUZW5zb3IgZXh0ZW5kcyBCYXNlVGVuc29yIHtcbiAgICAvLyBuZXcgVGVuc29yKGdsKVxuICAgIC8vIG5ldyBUZW5zb3IoZ2wsIFsxLCAxXSlcbiAgICAvLyBuZXcgVGVuc29yKGdsLCBbMSwgMV0sIG51bGwpXG4gICAgLy8gbmV3IFRlbnNvcihnbCwgWzEsIDFdLCBkYXRhKVxuICAgIC8vIG5ldyBUZW5zb3IoZ2wsIFsxLCAxXSwgZGF0YSwgeyB0eXBlLCBwYWNrLCBjb2RlYywgZGVuc2l0eSB9KVxuICAgIC8vIG5ldyBUZW5zb3IoZ2wsIFsxLCAxXSwgeyB0eXBlLCBwYWNrLCBjb2RlYywgZGVuc2l0eSB9KVxuICAgIC8vIG5ldyBUZW5zb3IoZ2wsIFsxLCAxXSwgJ3NvZnRmbG9hdCcpXG4gICAgLy8gbmV3IFRlbnNvcihnbCwgWzEsIDFdLCAnZmxvYXQzMicpXG4gICAgLy8gbmV3IFRlbnNvcihnbCwgWzEsIDFdLCAndWludDgnKVxuICAgIC8vIG5ldyBUZW5zb3IoZ2wsIHsgc2hhcGUsIGRhdGEgfSlcbiAgICAvLyBuZXcgVGVuc29yKGdsLCB7IHdpZHRoLCBoZWlnaHQsIGRhdGEgfSlcbiAgICAvLyBwaXggPSBuZXcgVGVuc29yKGdsLCBbMSwgMSwgNF0sIFsxLCAwLjQsIDMsIDRdLCAndWludDgnKVxuXG5cdGNvbnN0cnVjdG9yKGdsLCBzaGFwZSA9IFtdLCBkYXRhID0gbnVsbCwgZm9ybWF0ID0gbnVsbCl7XG4gICAgICAgIHN1cGVyKClcbiAgICAgICAgcnVuRmVhdHVyZVRlc3RzKGdsKTtcblxuICAgICAgICB2YXIgeGRhdGEgPSBkYXRhO1xuICAgICAgICBpZihzaGFwZS5zaGFwZSl7IC8vIG5kYXJyYXlzXG4gICAgICAgICAgICBmb3JtYXQgPSBkYXRhO1xuICAgICAgICAgICAgeGRhdGEgPSBzaGFwZS5kYXRhO1xuICAgICAgICAgICAgZGF0YSA9IHNoYXBlO1xuICAgICAgICAgICAgc2hhcGUgPSBzaGFwZS5zaGFwZTtcbiAgICAgICAgfVxuXG4gICAgICAgIGlmKHNoYXBlLndpZHRoICYmIHNoYXBlLmhlaWdodCAmJiBzaGFwZS5kYXRhKXsgLy8gaW1hZ2VkYXRhXG4gICAgICAgICAgICBkYXRhID0gc2hhcGUuZGF0YTtcbiAgICAgICAgICAgIHNoYXBlID0gW3NoYXBlLndpZHRoLCBzaGFwZS5oZWlnaHRdXG4gICAgICAgIH1cblxuICAgICAgICBpZih0eXBlb2YgZGF0YSA9PT0gJ3N0cmluZycpeyAvLyBkYXRhID0gdWludDggfCBmbG9hdDMyXG4gICAgICAgICAgICBpZihmb3JtYXQgIT09IG51bGwpXG4gICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdGb3JtYXQgbXVzdCBub3QgYmUgc3BlY2lmaWVkIGlmIGRhdGEgaXMgYSBzdHJpbmcuJyk7XG4gICAgICAgICAgICBmb3JtYXQgPSBkYXRhO1xuICAgICAgICAgICAgZGF0YSA9IG51bGw7XG4gICAgICAgIH1lbHNlIGlmKGRhdGEgJiYgdHlwZW9mIGRhdGEgPT09ICdvYmplY3QnICYmIGRhdGEudHlwZSAmJiBkYXRhLmNvZGVjICYmIGRhdGEucGFjayAmJiBkYXRhLmRlbnNpdHkpe1xuICAgICAgICAgICAgaWYoZm9ybWF0ICE9PSBudWxsKVxuICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcignRm9ybWF0IG11c3Qgbm90IGJlIHNwZWNpZmllZCBpZiBkYXRhIGlzIGFuIG9iamVjdC4nKTtcbiAgICAgICAgICAgIGZvcm1hdCA9IGRhdGE7XG4gICAgICAgICAgICBkYXRhID0gbnVsbDtcbiAgICAgICAgfVxuXG4gICAgICAgIGlmKGZvcm1hdCA9PT0gbnVsbCl7IC8vIGF1dG8taW5mZXIgZm9ybWF0IGJhc2VkIG9uIGRhdGFcbiAgICAgICAgICAgIGlmKGRhdGEgPT09IG51bGwpe1xuICAgICAgICAgICAgICAgIGZvcm1hdCA9ICdmbG9hdDMyJ1xuICAgICAgICAgICAgfWVsc2UgaWYoeGRhdGEgaW5zdGFuY2VvZiBVaW50OEFycmF5IHx8IHhkYXRhIGluc3RhbmNlb2YgVWludDhDbGFtcGVkQXJyYXkpe1xuICAgICAgICAgICAgICAgIGZvcm1hdCA9ICd1aW50OCdcbiAgICAgICAgICAgIH1lbHNlIGlmKHhkYXRhIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5IHx8IHhkYXRhIGluc3RhbmNlb2YgRmxvYXQ2NEFycmF5IHx8IEFycmF5LmlzQXJyYXkoeGRhdGEpKXtcbiAgICAgICAgICAgICAgICBmb3JtYXQgPSAnZmxvYXQzMidcbiAgICAgICAgICAgIH1lbHNlIHRocm93IG5ldyBFcnJvcihcIkludmFsaWQgZm9ybWF0IGZvciBkYXRhOiBtdXN0IGJlIFVpbnQ4QXJyYXkgb3IgRmxvYXQzMkFycmF5IG9yIG5kYXJyYXlcIik7XG4gICAgICAgIH1cblxuICAgICAgICB2YXIgdHlwZSA9IG51bGw7XG4gICAgICAgIGlmKChmb3JtYXQgPT09ICdmbG9hdDMyJyAmJiBcbiAgICAgICAgICAgIChnbC5OT19GTE9BVF9URVhUVVJFUyB8fCBcbiAgICAgICAgICAgIChnbC5OT19SRU5ERVJfRkxPQVQgJiYgdGhpcyBpbnN0YW5jZW9mIE91dHB1dFRlbnNvcikpKVxuICAgICAgICAgICAgfHwgZm9ybWF0ID09PSAnc29mdGZsb2F0Jyl7XG4gICAgICAgICAgICBmb3JtYXQgPSB7IHR5cGU6ICd1aW50OCcsIHBhY2s6ICdzdHJpZGUnLCBkZW5zaXR5OiAnMTo0JywgY29kZWM6ICdzb2Z0ZmxvYXQnIH1cbiAgICAgICAgICAgIHR5cGUgPSAnZmxvYXQzMidcbiAgICAgICAgfWVsc2UgaWYoZm9ybWF0ID09PSAndWludDgnIHx8IGZvcm1hdCA9PT0gJ2Zsb2F0MzInKXtcbiAgICAgICAgICAgIGZvcm1hdCA9IHsgdHlwZTogZm9ybWF0LCBwYWNrOiAnc3RyaWRlJywgZGVuc2l0eTogJzQ6NCcsIGNvZGVjOiAncmF3JyB9XG4gICAgICAgIH1cblxuICAgICAgICB0aGlzLnR5cGUgPSB0eXBlIHx8IGZvcm1hdC50eXBlO1xuICAgICAgICB0aGlzLl9pbml0KGdsLCBmb3JtYXQsIHNoYXBlLCBkYXRhKTtcblx0fVxuXG5cblx0Y29weShmb3JtYXQgPSB0aGlzLnR5cGUsIFQgPSBPdXRwdXRUZW5zb3Ipe1xuICAgICAgICBjb25zdCBURU5TT1JfSURFTlRJVFkgPSBgXG4gICAgICAgICAgICB1bmlmb3JtIFRlbnNvciBpbWFnZTtcbiAgICAgICAgICAgIHZlYzQgcHJvY2VzczQoaXZlYzQgcG9zKSB7IHJldHVybiBpbWFnZS5yZWFkNChwb3MpOyB9XG4gICAgICAgIGA7XG4gICAgICAgIHZhciBvdXQgPSBuZXcgVCh0aGlzLmdsLCB0aGlzLnNoYXBlLCBmb3JtYXQpO1xuICAgICAgICBvdXQucnVuKFRFTlNPUl9JREVOVElUWSwgeyBpbWFnZTogdGhpcyB9KVxuICAgICAgICByZXR1cm4gb3V0XG4gICAgfVxuXG4gICAgd2l0aENvcHkoZm4sIC4uLmFyZ3Mpe1xuICAgICAgICB2YXIgY29weSA9IHRoaXMuY29weSguLi5hcmdzKTtcbiAgICAgICAgdmFyIHJlc3VsdCA9IGZuKGNvcHkpXG4gICAgICAgIGNvcHkuZGVzdHJveSgpXG4gICAgICAgIHJldHVybiByZXN1bHQ7XG4gICAgfVxuXG5cdF9zaG93KG9wdCA9IHt9KXsgc2hvd1RleHR1cmUodGhpcy5nbCwgdGhpcy50ZXgsIG9wdCkgfVxuICAgIHNob3cob3B0ID0ge30pe1xuICAgICAgICB2YXIgZ2wgPSB0aGlzLmdsO1xuICAgICAgICBpZih0aGlzLmZvcm1hdC5wYWNrID09ICd0aWxlJyBcbiAgICAgICAgICAgICYmIHRoaXMuZm9ybWF0LmRlbnNpdHkgPT0gJzQ6NCcgXG4gICAgICAgICAgICAmJiB0aGlzLmZvcm1hdC5jb2RlYyA9PSAncmF3Jyl7XG4gICAgICAgICAgICB0aGlzLl9zaG93KG9wdClcbiAgICAgICAgfWVsc2V7XG4gICAgICAgICAgICAvLyBDLmluZm8ubWFpbl9pbnB1dC5vdXRwdXQuY29weSh7IHR5cGU6ICd1aW50OCcsIHBhY2s6ICd0aWxlJywgZGVuc2l0eTogJzQ6NCcsIGNvZGVjOiAnbGlucXVhbnQnLCBtaW46IDAsIG1heDogMjU1IH0pLl9zaG93KHsgfSlcbiAgICAgICAgICAgIHRoaXMud2l0aENvcHkoeCA9PiB4LnNob3cob3B0KSwgXG4gICAgICAgICAgICAgICAgeyB0eXBlOiBcbiAgICAgICAgICAgICAgICAgICAgKGdsLk5PX0ZMT0FUX1RFWFRVUkVTIHx8IGdsLk5PX1JFTkRFUl9GTE9BVCkgPyAndWludDgnIDogJ2Zsb2F0MzInLCBcbiAgICAgICAgICAgICAgICAgICAgcGFjazogJ3RpbGUnLCBkZW5zaXR5OiAnNDo0JywgY29kZWM6ICdyYXcnIH0pXG4gICAgICAgIH07XG4gICAgfVxuXG4gICAgcnVuKHNoYWRlciwgcGFyYW1zKXtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdPbmx5IE91dHB1dFRlbnNvciBjYW4gcnVuIHNoYWRlcnMuJylcbiAgICB9XG4gICAgY29tcGlsZShzaGFkZXIsIHBhcmFtcyl7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcignT25seSBPdXRwdXRUZW5zb3IgY2FuIGNvbXBpbGUgc2hhZGVycy4nKVxuICAgIH1cbiAgICByZWFkKCl7XG4gICAgICAgIGNvbnNvbGUud2FybihcIkNvcHlpbmcgYmVmb3JlIHJlYWQuLi5cIilcbiAgICAgICAgcmV0dXJuIHRoaXMud2l0aENvcHkoeCA9PiB4LnJlYWQoKSlcbiAgICB9XG4gICAgcHJpbnQoKXtcbiAgICAgICAgcmV0dXJuIG5kc2hvdyh0aGlzLnJlYWQoKSlcbiAgICB9XG4gICAgc3dhcCgpe1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXCJPbmx5IEluUGxhY2VUZW5zb3IgY2FuIGJlIGJvdGggYSBwYXJhbWV0ZXIgYW5kIGRlc3RpbmF0aW9uLlwiKTtcbiAgICB9XG59XG5cbmV4cG9ydCBjbGFzcyBPdXRwdXRUZW5zb3IgZXh0ZW5kcyBUZW5zb3Ige1xuXHRjb25zdHJ1Y3RvciguLi5hcmdzKXtcbiAgICAgICAgc3VwZXIoLi4uYXJncyk7XG5cdFx0dGhpcy5mYm8gPSBtYWtlRnJhbWVCdWZmZXIodGhpcy5nbCwgdGhpcy50ZXgpO1xuXHR9XG5cbiAgICBkZXN0cm95KCl7XG4gICAgICAgIHN1cGVyLmRlc3Ryb3koKVxuICAgICAgICB0aGlzLmdsLmRlbGV0ZUZyYW1lYnVmZmVyKHRoaXMuZmJvKVxuICAgIH1cblxuICAgIF9yZWFkKCl7XG4gICAgICAgIHZhciBnbCA9IHRoaXMuZ2wsXG4gICAgICAgICAgICBzaXplID0gdGhpcy5pbmZvLnRleFNpemU7XG5cbiAgICAgICAgaWYodGhpcy5mb3JtYXQudHlwZSA9PSAndWludDgnKXtcbiAgICAgICAgICAgIHZhciBnbFR5cGUgPSBnbC5VTlNJR05FRF9CWVRFLFxuICAgICAgICAgICAgICAgIHBpeGVscyA9IG5ldyBVaW50OEFycmF5KHNpemVbMF0gKiBzaXplWzFdICogNClcbiAgICAgICAgfWVsc2UgaWYodGhpcy5mb3JtYXQudHlwZSA9PT0gJ2Zsb2F0MzInKXtcbiAgICAgICAgICAgIHZhciBnbFR5cGUgPSBnbC5GTE9BVCxcbiAgICAgICAgICAgICAgICBwaXhlbHMgPSBuZXcgRmxvYXQzMkFycmF5KHNpemVbMF0gKiBzaXplWzFdICogNClcbiAgICAgICAgfVxuXG4gICAgICAgIGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgdGhpcy5mYm8pO1xuICAgICAgICBnbC5yZWFkUGl4ZWxzKDAsIDAsIHNpemVbMF0sIHNpemVbMV0sIGdsLlJHQkEsIGdsVHlwZSwgcGl4ZWxzKTtcblxuICAgICAgICAvLyBjb25zb2xlLmxvZygnX19fcmVhZCcsIHBpeGVscylcbiAgICAgICAgcmV0dXJuIHBpeGVscztcbiAgICB9XG5cbiAgICBydW4oc2hhZGVyLCBwYXJhbXMsIGNhbGxiYWNrKXtcbiAgICAgICAgcmV0dXJuIFJ1bihzaGFkZXIsIHRoaXMsIHBhcmFtcywgY2FsbGJhY2spO1xuICAgIH1cbiAgICBjb21waWxlKHNoYWRlciwgcGFyYW1zKXtcbiAgICAgICAgcmV0dXJuIENvbXBpbGUoc2hhZGVyLCB0aGlzLCBwYXJhbXMpO1xuICAgIH1cblxuXHRyZWFkKCl7XG4gICAgICAgIGlmKHRoaXMuZm9ybWF0LnR5cGUgPT09ICdmbG9hdDMyJyAmJiB0aGlzLmdsLk5PX1JFQURfRkxPQVQpe1xuICAgICAgICAgICAgcmV0dXJuIHRoaXMud2l0aENvcHkoeCA9PiB4LnJlYWQoKSwgJ3NvZnRmbG9hdCcpXG4gICAgICAgIH1cblxuXHRcdHZhciBhcnJheSA9IHRoaXMuX2Zvcm1hdC5wYWNrLnVucGFjayh0aGlzLmluZm8sIHRoaXMuX3JlYWQoKSwgdGhpcy5fZm9ybWF0LmNvZGVjLmRlY29kZSwgdGhpcy50eXBlKTtcbiAgICAgICAgXG4gICAgICAgIC8vIHN0cmlwIHRyYWlsaW5nIHNpbmdsZXRvbiBkaW1lbnNpb25zXG4gICAgICAgIHZhciBzaGFwZSA9IGFycmF5LnNoYXBlLnNsaWNlKDApLFxuICAgICAgICAgICAgc3RyaWRlID0gYXJyYXkuc3RyaWRlLnNsaWNlKDApO1xuICAgICAgICB3aGlsZShzaGFwZVtzaGFwZS5sZW5ndGggLSAxXSA9PSAxICYmIHNoYXBlLmxlbmd0aCA+IDEpe1xuICAgICAgICAgICAgc2hhcGUucG9wKClcbiAgICAgICAgICAgIHN0cmlkZS5wb3AoKVxuICAgICAgICB9XG4gICAgICAgIHJldHVybiBuZGFycmF5KGFycmF5LmRhdGEsIHNoYXBlLCBzdHJpZGUsIGFycmF5Lm9mZnNldCk7XG5cdH1cbn1cblxuZXhwb3J0IGNsYXNzIEluUGxhY2VUZW5zb3IgZXh0ZW5kcyBPdXRwdXRUZW5zb3Ige1xuXHRjb25zdHJ1Y3RvciguLi5hcmdzKXtcblx0XHRzdXBlciguLi5hcmdzKVxuXG4gICAgICAgIHRoaXMudGV4MiA9IHRoaXMudGV4O1xuICAgICAgICB0aGlzLnRleCA9IG1ha2VUZXh0dXJlKHRoaXMuZ2wpO1xuXHRcdHRoaXMudXBkYXRlKG51bGwpO1xuICAgICAgICB0aGlzLnN3YXAoKVxuXHR9XG4gICAgZGVzdHJveSgpe1xuICAgICAgICBzdXBlci5kZXN0cm95KClcbiAgICAgICAgdGhpcy5nbC5kZWxldGVUZXh0dXJlKHRoaXMudGV4MilcbiAgICB9XG4gICAgc3dhcCgpe1xuICAgICAgICB2YXIgdG1wID0gdGhpcy50ZXg7XG4gICAgICAgIHRoaXMudGV4ID0gdGhpcy50ZXgyO1xuICAgICAgICB0aGlzLnRleDIgPSB0bXA7XG5cbiAgICAgICAgLy8gVE9ETzogaW52ZXN0aWdhdGUgcGVyZm9ybWFuY2Ugb2YgdXNpbmcgbXVsdGlwbGUgRkJPcyBpbnN0ZWFkXG4gICAgICAgIC8vIG9mIHJlYmluZGluZyB0aGUgZnJhbWVidWZmZXJcbiAgICAgICAgdmFyIGdsID0gdGhpcy5nbDtcbiAgICAgICAgZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCB0aGlzLmZibyk7XG4gICAgICAgIGdsLmZyYW1lYnVmZmVyVGV4dHVyZTJEKGdsLkZSQU1FQlVGRkVSLCBnbC5DT0xPUl9BVFRBQ0hNRU5UMCwgZ2wuVEVYVFVSRV8yRCwgdGhpcy50ZXgsIDApO1xuICAgIH1cbn0iLCJpbXBvcnQgeyBiaW5kQXR0cmlidXRlQnVmZmVyLCBjcmVhdGVTaGFkZXJQcm9ncmFtIH0gZnJvbSAnLi4vcnVudGltZS9wcm9ncmFtLmpzJ1xuXG5jb25zdCBTSE9XX1RFWFRVUkVfVkVSVEVYID0gYFxuICAgIGF0dHJpYnV0ZSB2ZWMyIGFfcG9zaXRpb247XG4gICAgdmFyeWluZyBtZWRpdW1wIHZlYzIgcG9zO1xuICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgcG9zID0gKGFfcG9zaXRpb24gKyB2ZWMyKDEsIDEpKSAvIDIuMDtcbiAgICAgICAgZ2xfUG9zaXRpb24gPSB2ZWM0KGFfcG9zaXRpb24sIDAsIDEpO1xuICAgIH1cbmBcblxuY29uc3QgU0hPV19URVhUVVJFX0ZSQUdNRU5UID0gYFxuICAgIHByZWNpc2lvbiBtZWRpdW1wIGZsb2F0O1xuXG4gICAgdW5pZm9ybSBzYW1wbGVyMkQgdGV4O1xuICAgIHVuaWZvcm0gZmxvYXQgc2NhbGU7XG4gICAgdW5pZm9ybSBmbG9hdCBvZmZzZXQ7XG4gICAgdW5pZm9ybSBib29sIHRyYW5zcG9zZTtcbiAgICB1bmlmb3JtIGJvb2wgZmxpcFg7XG4gICAgdW5pZm9ybSBib29sIGZsaXBZO1xuICAgIHVuaWZvcm0gaW50IGNoYW5uZWxzO1xuXG4gICAgdmFyeWluZyB2ZWMyIHBvcztcblxuICAgIHZlYzQgY29sb3JtYXAoZmxvYXQgeCkge1xuICAgICAgICBmbG9hdCByID0gY2xhbXAoOC4wIC8gMy4wICogeCwgMC4wLCAxLjApO1xuICAgICAgICBmbG9hdCBnID0gY2xhbXAoOC4wIC8gMy4wICogeCAtIDEuMCwgMC4wLCAxLjApO1xuICAgICAgICBmbG9hdCBiID0gY2xhbXAoNC4wICogeCAtIDMuMCwgMC4wLCAxLjApO1xuICAgICAgICByZXR1cm4gdmVjNChyLCBnLCBiLCAxLjApO1xuICAgIH1cblxuICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgdmVjMiBwID0gcG9zO1xuICAgICAgICBpZihmbGlwWCkgcC54ID0gMS4wIC0gcC54O1xuICAgICAgICBpZihmbGlwWSkgcC55ID0gMS4wIC0gcC55O1xuICAgICAgICBpZih0cmFuc3Bvc2UpIHAgPSBwLnl4O1xuICAgICAgICBpZihjaGFubmVscyA9PSA0KXtcbiAgICAgICAgICAgIGdsX0ZyYWdDb2xvciA9IHZlYzQodmVjNChvZmZzZXQsIG9mZnNldCwgb2Zmc2V0LCBvZmZzZXQpIFxuICAgICAgICAgICAgICAgICsgc2NhbGUgKiB0ZXh0dXJlMkQodGV4LCBwKSk7XG4gICAgICAgIH1lbHNlIGlmKGNoYW5uZWxzID09IDMpe1xuICAgICAgICAgICAgZ2xfRnJhZ0NvbG9yID0gdmVjNCh2ZWMzKG9mZnNldCwgb2Zmc2V0LCBvZmZzZXQpIFxuICAgICAgICAgICAgICAgICsgc2NhbGUgKiB0ZXh0dXJlMkQodGV4LCBwKS5yZ2IsIDEpO1xuICAgICAgICB9ZWxzZSBpZihjaGFubmVscyA9PSAyKXtcbiAgICAgICAgICAgIGdsX0ZyYWdDb2xvciA9IHZlYzQodmVjMihvZmZzZXQsIG9mZnNldCkgXG4gICAgICAgICAgICAgICAgKyBzY2FsZSAqIHRleHR1cmUyRCh0ZXgsIHApLnJnLCAwLCAxKTtcbiAgICAgICAgfWVsc2UgaWYoY2hhbm5lbHMgPT0gMSl7XG4gICAgICAgICAgICBnbF9GcmFnQ29sb3IgPSBjb2xvcm1hcChvZmZzZXQgKyBzY2FsZSAqIHRleHR1cmUyRCh0ZXgsIHApLnIpO1xuICAgICAgICB9XG4gICAgfVxuYFxuXG5leHBvcnQgZGVmYXVsdCBmdW5jdGlvbiBzaG93VGV4dHVyZShnbCwgdGV4LCBvcHQgPSB7fSl7XG4gICAgaWYoIWdsLl9zaG93UHJvZ3JhbSl7XG4gICAgICAgIGdsLl9zaG93UHJvZ3JhbSA9IGNyZWF0ZVNoYWRlclByb2dyYW0oZ2wsIFNIT1dfVEVYVFVSRV9WRVJURVgsIFNIT1dfVEVYVFVSRV9GUkFHTUVOVCk7XG4gICAgICAgIGdsLnVzZVByb2dyYW0oZ2wuX3Nob3dQcm9ncmFtKTtcbiAgICAgICAgYmluZEF0dHJpYnV0ZUJ1ZmZlcihnbCwgZ2wuX3Nob3dQcm9ncmFtKTtcbiAgICAgICAgZ2wudW5pZm9ybTFpKGdsLmdldFVuaWZvcm1Mb2NhdGlvbihnbC5fc2hvd1Byb2dyYW0sICd0ZXgnKSwgMCk7XG4gICAgfVxuICAgIFxuXG4gICAgaWYoZ2wuY2FudmFzICYmIGdsLmNhbnZhcy5fdGZBdXRvKXtcbiAgICAgICAgZ2wuY2FudmFzLnN0eWxlLmRpc3BsYXkgPSAnYmxvY2snXG4gICAgICAgIGdsLmNhbnZhcy5zdHlsZS5wb3NpdGlvbiA9ICdhYnNvbHV0ZSdcbiAgICAgICAgZ2wuY2FudmFzLnN0eWxlLnRvcCA9IDA7XG4gICAgICAgIGdsLmNhbnZhcy5zdHlsZS5sZWZ0ID0gMDtcbiAgICAgICAgZ2wuY2FudmFzLnN0eWxlLndpZHRoID0gTWF0aC5taW4oaW5uZXJIZWlnaHQsIGlubmVyV2lkdGgpICsgJ3B4J1xuICAgICAgICBnbC5jYW52YXMuc3R5bGUuaGVpZ2h0ID0gTWF0aC5taW4oaW5uZXJIZWlnaHQsIGlubmVyV2lkdGgpICsgJ3B4J1xuICAgIH1cblxuICAgIGdsLnVzZVByb2dyYW0oZ2wuX3Nob3dQcm9ncmFtKTtcbiAgICBnbC5hY3RpdmVUZXh0dXJlKGdsLlRFWFRVUkUwKTtcbiAgICBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0ZXgpO1xuICAgIGdsLnVuaWZvcm0xZihnbC5nZXRVbmlmb3JtTG9jYXRpb24oZ2wuX3Nob3dQcm9ncmFtLCAnc2NhbGUnKSwgb3B0LnNjYWxlIHx8IDEpXG4gICAgZ2wudW5pZm9ybTFmKGdsLmdldFVuaWZvcm1Mb2NhdGlvbihnbC5fc2hvd1Byb2dyYW0sICdvZmZzZXQnKSwgb3B0Lm9mZnNldCB8fCAwKVxuICAgIGdsLnVuaWZvcm0xaShnbC5nZXRVbmlmb3JtTG9jYXRpb24oZ2wuX3Nob3dQcm9ncmFtLCAndHJhbnNwb3NlJyksIG9wdC50cmFuc3Bvc2UgPyAxIDogMClcbiAgICBnbC51bmlmb3JtMWkoZ2wuZ2V0VW5pZm9ybUxvY2F0aW9uKGdsLl9zaG93UHJvZ3JhbSwgJ2ZsaXBYJyksIG9wdC5mbGlwWCA/IDEgOiAwKVxuICAgIGdsLnVuaWZvcm0xaShnbC5nZXRVbmlmb3JtTG9jYXRpb24oZ2wuX3Nob3dQcm9ncmFtLCAnZmxpcFknKSwgb3B0LmZsaXBZID8gMSA6IDApXG4gICAgZ2wudW5pZm9ybTFpKGdsLmdldFVuaWZvcm1Mb2NhdGlvbihnbC5fc2hvd1Byb2dyYW0sICdjaGFubmVscycpLCBvcHQuY2hhbm5lbHMgfHwgMyk7XG5cbiAgICBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIG51bGwpO1xuICAgIGdsLnZpZXdwb3J0KDAsIDAsIGdsLmRyYXdpbmdCdWZmZXJXaWR0aCwgZ2wuZHJhd2luZ0J1ZmZlckhlaWdodCk7XG4gICAgZ2wuZHJhd0FycmF5cyhnbC5UUklBTkdMRV9TVFJJUCwgMCwgNCk7XG5cbn1cbiIsImV4cG9ydCBmdW5jdGlvbiBjcmVhdGVHTChjYW52YXMpe1xuICAgIGlmKCFjYW52YXMpe1xuICAgICAgICBjYW52YXMgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdjYW52YXMnKTtcbiAgICAgICAgY2FudmFzLndpZHRoID0gNTEyXG4gICAgICAgIGNhbnZhcy5oZWlnaHQgPSA1MTJcbiAgICAgICAgY2FudmFzLnN0eWxlLmRpc3BsYXkgPSAnbm9uZSc7XG4gICAgICAgIGNhbnZhcy5fdGZBdXRvID0gdHJ1ZTtcbiAgICAgICAgZG9jdW1lbnQuYm9keS5hcHBlbmRDaGlsZChjYW52YXMpXG4gICAgfVxuICAgIHZhciBnbCA9IGNhbnZhcy5nZXRDb250ZXh0KFwid2ViZ2xcIiwgeyBhbnRpYWxpYXM6IGZhbHNlIH0pIFxuICAgICAgICAgIHx8IGNhbnZhcy5nZXRDb250ZXh0KFwiZXhwZXJpbWVudGFsLXdlYmdsXCIsIHsgYW50aWFsaWFzOiBmYWxzZSB9KTtcbiAgICBpZiAoIWdsKSBhbGVydCgnQ291bGQgbm90IGluaXRpYWxpemUgV2ViR0wsIHRyeSBhbm90aGVyIGJyb3dzZXInKTtcbiAgICByZXR1cm4gZ2w7XG59XG4iXX0=
