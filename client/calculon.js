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
	//r.setRequestHeader("Accept-Encoding", "gzip,deflate");
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
	//r.setRequestHeader("Accept-Encoding", "gzip, deflate");
	r.setRequestHeader("Content-Type", contentType);
	r.send(body);
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
		received: null, // model data & weights received from server
		loaded: null, // model weights loaded
		trained: null, // model finished training
		updated: null // model updates sent
	};

	Model = Model(TF, GL);

	function update(arraybuffer) {

		var iteration = new Float32Array(arraybuffer, 0, 1)[0],
		    testView = new Float32Array(arraybuffer),
		    view,
		    weights,
		    data,
		    len,
		    i,
		    batch;

		//console.log(testView);
		times.received = window.performance.now();

		view = new Float32Array(arraybuffer, 4);

		if (iteration >= 0) {
			// includes new weights and data
			iterations = iteration;
			i = model.size;
			weights = view.subarray(0, i);
			len = view[i] * net.layers[0].shape[1]; // first float is number of samples in this batch
			len += ++i;
			batch = {
				x: view.subarray(i, len),
				y: view.subarray(len)
			};

			model.load(weights);
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
		model.train(net.learning_rate, net.iterations, batch.x, batch.y, function (weights, accuracy) {
			var r = 0,
			    log = "",
			    w = new Float32Array(weights);
			times.trained = window.performance.now();
			//console.log("Time to train: " + (delta / 1000) + " seconds");
			// post results to server
			PUT("./weights/" + net.id, "arraybuffer", weights, update);
			r = window.performance.now();
			log += iterations + ",";
			log += accuracy + ",";
			log += times.requested + ",";
			log += times.received + ",";
			log += times.loaded + ",";
			log += times.trained;
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
		//console.log(output.read().data);
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
	if (typeof callback === "function") callback(this.save(), this.loss);
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
	if (layers != null && !(layers instanceof Float32Array)) {
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
		"linear": "\n\t\t\to = n;\n\t\t",
		// "binary": `
		// 	if (n > 0.0) { o = 0.0; } else { o = 1.0; }
		// `,
		"relu": "\n\t\t\to = max(0.0, n);\n\t\t",
		"lrelu": "\n\t\t\tif (n >= 0.0) { o = n; } else { o = 0.01 * n; }\n\t\t",
		"sigmoid": "\n\t\t\to = 1.0 / (1.0 + exp(0.0 - n));\n\t\t",
		"tanh": "\n\t\t\to = (2.0 / (1.0 + exp(-2.0 * n))) - 1.0;\n\t\t",
		"softplus": "\n\t\t\to = log(1.0 + exp(n));\n\t\t",
		"softmax": "\n\t\t\tfloat k = 0.0;\n\t\t\tfor(int i = 0; i < #(O.shape).x; i++){\n\t\t\t\tk += exp(O.read(i, pos.y));\n\t\t\t}\n\t\t\to = exp(n) / k;\n\t\t"
	},
	Derivative: {
		"linear": "\n\t\t\td = 1.0;\n\t\t",
		// "binary": `
		// 	if (o == 0.0) {
		// 		d = 0.0;
		// 	} else {
		// 		d = 0.0;
		// 	}
		// `,
		"relu": "\n\t\t\tif (o >= 0.0) {\n\t\t\t\td = 1.0;\n\t\t\t} else {\n\t\t\t\td = 0.0;\n\t\t\t}\n\t\t",
		"lrelu": "\n\t\t\tif (o >= 0.0) {\n\t\t\t\td = 1.0;\n\t\t\t} else {\n\t\t\t\td = 0.01;\n\t\t\t}\n\t\t",
		"sigmoid": "\n\t\t\td = o * ( 1.0 - o );\n\t\t",
		"tanh": "\n\t\t\td = ( 4.0 / pow(( exp(-o) + exp(o)), 2.0) );\n\t\t",
		"softplus": "\n\t\t\td = 1.0 - ( 1.0 / exp(o) );\n\t\t",
		"softmax": "\n\t\t\td = o * ( 1.0 - o );\n\t\t"
	}
};

},{}],5:[function(require,module,exports){
"use strict";

var ndarray = require("ndarray"),
    TF,
    GL,
    Grad = "\n\t\tuniform Tensor O;\n\t\tuniform Tensor E;\n\t\tfloat process(ivec4 pos) {\n\t\t\treturn 0.0 - E.read(pos) / O.read(pos);\n\t\t}\n\t",
    Loss = "\n\t\tuniform Tensor O;\n\t\tuniform Tensor E;\n\t\tfloat process(ivec4 pos) {\n\t\t\tfloat loss = 0.0;\n\t\t\tfor(int i = 0; i < #(O.shape).y; i++){ // iterate over each sample\n\t\t\t\tfloat l = 0.0;\n\t\t\t\tfor(int j = 0; j < #(O.shape).x; j++){ // iterate over every output and calculate average\n\t\t\t\t\tl -= E.read(j, i) * log(O.read(j, i));\n\t\t\t\t}\n\t\t\t\tloss = l / float(#(O.shape).y);\n\t\t\t}\n\t\t\treturn loss;\n\t\t}\n\t";

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
    ForwardBiased = '\n\t\tuniform Tensor W; // layer weights\n\t\tuniform Tensor I; // layer inputs\n\t\tfloat process(ivec4 pos) { // for each unit in output (x: unit, y: sample)\n\t\t\t\tfloat n = 0.0;\n\t\t\t\tfor(int i = 0; i < #(W.shape).y; i++){ // for each weight\n\t\t\t\t\tif (i == #(W.shape).y - 1) {\n\t\t\t\t\t\tn += W.read(pos.x, i);\n\t\t\t\t\t} else {\n\t\t\t\t\t\tn += I.read(i, pos.y) * W.read(pos.x, i);\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t\treturn n;\n\t\t}\n\t',
    ForwardUnbiased = ' // for each output node\n\t\tuniform Tensor W; // layer weights\n\t\tuniform Tensor I; // layer inputs\n\t\tfloat process(ivec4 pos) {\n\t\t\tfloat n = 0.0;\n\t\t\tfor(int i = 0; i < #(W.shape).y; i++){\n\t\t\t\tn += I.read(i, pos.y) * W.read(pos.x, i);\n\t\t\t}\n\t\t\treturn n;\n\t\t}\n\t',
    BackwardBiased = ' // for each input node\n\t\tuniform Tensor E; // local error (from activation)\n\t\tuniform Tensor W; // weights\n\t\tfloat process(ivec4 pos) { // position in input gradient Tensor\n\t\t\tfloat e = 0.0; // sum output error\n\t\t\tfor(int i = 0; i < #(E.shape).x; i++){\n\t\t\t\tif (pos.y != #(E.shape).x) {\n\t\t\t\t\te += W.read(i, pos.x) * E.read(i, pos.y);\n\t\t\t\t}\n\t\t\t}\n\t\t\treturn e;\n\t\t}\n\t',
    BackwardUnbiased = ' // for each input node\n\t\tuniform Tensor E; // local error (from activation)\n\t\tuniform Tensor W; // weights\n\t\tfloat process(ivec4 pos) { // position in input gradient Tensor\n\t\t\tfloat e = 0.0; // sum output error\n\t\t\tfor(int i = 0; i < #(E.shape).x; i++){\n\t\t\t\te += W.read(i, pos.x) * E.read(i, pos.y);\n\t\t\t}\n\t\t\treturn e;\n\t\t}\n\t',
    Weights = '\n\t\tuniform Tensor E; // local error (from activation)\n\t\tuniform Tensor W; // weights\n\t\tuniform Tensor I; // input\n\t\tuniform float l; // learning rate\n\t\tfloat process(ivec4 pos) { // pos in weights Tensor\n\t\t\tfloat e = 0.0; // avg node batch error\n\t\t\tfor(int i = 0; i < #(E.shape).y; i++){\n\t\t\t\tif (pos.y == #(I.shape).x) { // handle bias layer ?\n\t\t\t\t\te += E.read(pos.x, i);\n\t\t\t\t} else {\n\t\t\t\t\te += E.read(pos.x, i) * I.read(pos.y, i);\n\t\t\t\t}\n\t\t\t}\n\t\t\treturn W.read(pos) - (l * e);\n\t\t}\n\t',
    Activation = function Activation(activationFunction) {
	return '\n\t\tuniform Tensor O; // weighted input\n\t\tfloat process(ivec4 pos) {\n\t\t\tfloat n = O.read(pos);\n\t\t\tfloat o;\n\t\t\t' + activationFunction + '\n\t\t\treturn o;\n\t\t}\n\t';
},
    Gradient = function Gradient(derivativeFunction) {
	return '\n\t\tuniform Tensor E;\t// downstream error\n\t\tuniform Tensor O;\t// layer output\n\t\tuniform Tensor H;\t// weighted input\n\t\tfloat process(ivec4 pos) {\n\t\t\tfloat d;\n\t\t\tfloat o = O.read(pos);\n\t\t\t' + derivativeFunction + '\n\t\t\td *= E.read(pos);\n\t\t\treturn d;\n\t\t}\n\t';
};

function Dense(layer, index) {
	this.l = index;
	// produce Output Tensor given input, weights, and bias Tensors
	this.forward = layer.bias ? ForwardBiased : ForwardUnbiased;

	this.activation = Activation(Funcs.Activation[layer.activation]);

	// produce upstream error Tensor given downstream error, input, weights, bias
	this.backward = layer.bias ? BackwardBiased : BackwardUnbiased;
	this.gradient = Gradient(Funcs.Derivative[layer.activation]);
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
	this.weights = new TF.InPlaceTensor(GL, ndarray(generateWeights(this.shape, this.bias ? this.shape[0] : 0), // values
	[this.shape[0], this.shape[1] + (this.bias ? 1 : 0)] // shape
	));
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

	//console.log("output " + this.l: " + this.output.read().data);
	return this.output;
};
Dense.prototype.train = function (error, learning_rate) {
	this.partial = new TF.OutputTensor(GL, this.input.shape);
	this.local = new TF.OutputTensor(GL, this.output.shape);

	//console.log("Calculon- error: " + error.read().data);
	//console.log("Calculon- weights " + this.l: " + this.weights.read().data);

	// calculate local error from weightedOutput (strips out error from activation function)
	this.local.run(this.gradient, { E: error, O: this.output, H: this.weightedOutput });
	//console.log("Calculon- localE: " + local.read().data);

	// calculate upstream errors from input
	this.partial.run(this.backward, { E: this.local, W: this.weights });

	// train weights based on local error
	this.weights.run(this.update, { W: this.weights, E: this.local, I: this.input, l: learning_rate });
	//console.log("Calculon- updated " + this.l: " + this.weights.read().data);

	return this.partial;
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
    Grad = "\n\t\tuniform Tensor O;\n\t\tuniform Tensor E;\n\t\tfloat process(ivec4 pos) {\n\t\t\treturn O.read(pos) - E.read(pos);\n\t\t}\n\t",
    Accuracy = "\n\t\tuniform Tensor O;\n\t\tuniform Tensor E;\n\t\tfloat process(ivec4 pos) {\n\t\t\tfloat s = 0.0;\n\t\t\tfor (int i = 0; i < #(O.shape).x; i++) { // iterate over every output\n\t\t\t\ts += pow((E.read(i, pos.x) - O.read(i, pos.x)), 2.0);\n\t\t\t}\n\t\t\treturn 1.0 - clamp(s / float(#(O.shape).x), 0.0, 1.0);\n\t\t}\n\t",
    Loss = "\n\t\tuniform Tensor G;\n\t\tfloat process(ivec4 pos) {\n\t\t\tfloat loss = 0.0;\n\t\t\tfor(int i = 0; i < #(G.shape).y; i++){ // iterate over each sample\n\t\t\t\tfloat l = 0.0;\n\t\t\t\tfor(int j = 0; j < #(G.shape).x; j++){ // iterate over every output and calculate average\n\t\t\t\t\tl += pow(float(G.read(j, i)), 2.0);\n\t\t\t\t}\n\t\t\t\tloss += l / float(#(G.shape).y);\n\t\t\t}\n\t\t\treturn loss;\n\t\t}\n\t";

function MSE() {
	// calculate loss gradients
	this.grad = Grad;

	// calculate batch average loss
	this.lossF = Loss;

	this.Accuracy = Accuracy;

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
    AccSum = '\n\t\tuniform Tensor A;\n\t\tfloat process(ivec4 pos) {\n\t\t\tfloat acc = 0.0;\n\t\t\tfor (int i = 0; i < #(A.shape).y; i++){ // iterate over each sample\n\t\t\t\tacc += A.read(pos.x, i);\n\t\t\t}\n\t\t\treturn acc / float(#(A.shape).y);\n\t\t}\n\t';

function Output(layer, index) {
	var _this = this;

	this.output = null;
	if (layer.activation === "softmax" && layer.loss === "xentropy") {
		this.output = new Softmax(layer, index);
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
	//console.log("Expected: " + expected);
	if (expected instanceof Float32Array) expected = new TF.Tensor(GL, ndarray(expected, this._output.shape));

	//console.log("  Output: " + this._output.read().data);

	this.batchAccuracy = new TF.OutputTensor(GL, [1, this._output.shape[1]]);
	this._accuracy = new TF.OutputTensor(GL, [1]);
	this.batchAccuracy.run(this.output.Accuracy, { O: this._output, E: expected });
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
    Act = "\n\t\tuniform Tensor I; // input\n\t\tfloat process(ivec4 pos) {\n\t\t\tfloat k = 0.0;\n\t\t\tfor(int i = 0; i < #(I.shape).x; i++){\n\t\t\t\tk += exp(I.read(i, pos.y));\n\t\t\t}\n\t\t\treturn exp(I.read(pos)) / k;\n\t\t}\n\t",
    Grad = "\n\t\tuniform Tensor O; // Softmax output\n\t\tuniform Tensor E; // expected output\n\t\tfloat process(ivec4 pos) {\n\t\t\treturn O.read(pos) - E.read(pos);\n\t\t}\n\t",
    Accuracy = "\n\t\tuniform Tensor O;\n\t\tuniform Tensor E;\n\t\tfloat process(ivec4 pos) {\n\t\t\tfloat s = 0.0;\n\t\t\tfor (int i = 0; i < #(O.shape).x; i++) { // iterate over every output\n\t\t\t\ts += pow((E.read(i, pos.x) - O.read(i, pos.x)), 2.0);\n\t\t\t}\n\t\t\treturn 1.0 - clamp(s / float(#(O.shape).x), 0.0, 1.0);\n\t\t}\n\t",


// Accuracy = `
// 	uniform Tensor O;
// 	uniform Tensor E;
// 	float process(ivec4 pos) {
// 		int l = #(O.shape).x;
// 		float v = -100000.0;
// 		for (int i = 0; i < #(O.shape).x; i++) { // iterate over every output
// 			if (O.read(i, pos.x) > v) {
// 				v = O.read(i, pos.y); /* get largest category in output */
// 				l = i; /* save index of largest value */
// 			}
// 		}
// 		if (E.read(l, pos.y) > 0.9) {
// 			return 1.0;
// 		} else {
// 			return 0.0;
// 		}
// 	}
// `,


Loss = "\n\t\tuniform Tensor G;\n\t\tfloat process(ivec4 pos) {\n\t\t\tfloat loss = 0.0;\n\t\t\tfor(int i = 0; i < #(G.shape).y; i++){ /* iterate over each sample */\n\t\t\t\tfloat l = 0.0;\n\t\t\t\tfor(int j = 0; j < #(G.shape).x; j++){ /* iterate over every output and calculate average */\n\t\t\t\t\tl += pow(float(G.read(j, i)), 2.0) / float(#(G.shape).x);\n\t\t\t\t}\n\t\t\t\tloss += l / float(#(G.shape).y);\n\t\t\t}\n\t\t\treturn loss;\n\t\t}\n\t";

function Softmax(layer, index) {
	this.l = index;

	this.activation = Act;

	this.gradient = Grad;
	this.lossF = Loss;

	this.Accuracy = Accuracy;

	this.input = null;
	this.output = null;
	this.loss = new TF.OutputTensor(GL, [1]);
}
Softmax.prototype.run = function (input) {

	this.input = input;
	this.output = new TF.OutputTensor(GL, this.input.shape);

	this.output.run(this.activation, { I: this.input });

	//console.log("output " + this.l + ": " + this.output.read().data);
	return this.output;
};
Softmax.prototype.deltas = function (output, expected) {
	this.partial = new TF.OutputTensor(GL, this.input.shape);

	if (expected instanceof Float32Array) expected = new TF.Tensor(GL, ndarray(expected, this.input.shape));

	// calculate upstream errors
	this.partial.run(this.gradient, { O: output, E: expected });

	// calculate batch training loss
	this.loss.run(this.lossF, { G: this.partial });
	//console.log(this.loss.read());

	// console.log(output.read().data);

	return this.partial;
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
	console.log("Layer weights + bias: " + result.length);
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJjbGllbnQvY2xpZW50LmpzIiwibGliL0xheWVycy5qcyIsImxpYi9Nb2RlbC5qcyIsImxpYi9sYXllcnMvQWN0aXZhdGlvbnMuanMiLCJsaWIvbGF5ZXJzL0Nyb3NzRW50cm9weS5qcyIsImxpYi9sYXllcnMvRGVuc2UuanMiLCJsaWIvbGF5ZXJzL01TRS5qcyIsImxpYi9sYXllcnMvT3V0cHV0LmpzIiwibGliL2xheWVycy9Tb2Z0bWF4LmpzIiwibGliL3V0aWwvZ2VuZXJhdGVXZWlnaHRzLmpzIiwibm9kZV9tb2R1bGVzL2ZpeGVkLXdpZHRoLWZsb2F0L2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL2lvdGEtYXJyYXkvaW90YS5qcyIsIm5vZGVfbW9kdWxlcy9pcy1idWZmZXIvaW5kZXguanMiLCJub2RlX21vZHVsZXMvbmRhcnJheS1zaG93L2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL25kYXJyYXkvbmRhcnJheS5qcyIsIm5vZGVfbW9kdWxlcy9zcHJpbnRmL2xpYi9zcHJpbnRmLmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC8xLTQvYWN0aXZhdGlvbi9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9mb3JtYXQvMS00L2NvZGVjL2ZpeG51bS9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9mb3JtYXQvMS00L2NvZGVjL3NvZnRmbG9hdC9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9mb3JtYXQvMS00L2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC8xLTQvcGFjay9zdHJpZGUvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvZm9ybWF0LzEtNC9wYWNrL3RpbGUvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvZm9ybWF0LzQtNC9hY3RpdmF0aW9uL2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC80LTQvY29kZWMvbGlucXVhbnQvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvZm9ybWF0LzQtNC9jb2RlYy9yYXcvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvZm9ybWF0LzQtNC9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9mb3JtYXQvNC00L3BhY2svc3RyaWRlL2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC80LTQvcGFjay90aWxlL2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9ydW50aW1lL2NoZWNrLmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL3J1bnRpbWUvZnJhZy5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9ydW50aW1lL2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL3J1bnRpbWUvcHJvZ3JhbS5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9ydW50aW1lL3RpbWVyLmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL3J1bnRpbWUvdG5zbC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy90ZW5zb3IvYmFzZS5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy90ZW5zb3IvZmVhdHVyZS5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy90ZW5zb3IvaGVscGVycy5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy90ZW5zb3IvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvdGVuc29yL3Nob3cuanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvdXRpbC5qcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7O0FDQUEsSUFBSSxRQUFRLFFBQVosQUFBWSxBQUFRO0lBQ25CLEtBQUssUUFETixBQUNNLEFBQVE7SUFDYixLQUFLLEdBRk4sQUFFTSxBQUFHOztBQUVULFNBQUEsQUFBUyxJQUFULEFBQWEsTUFBYixBQUFtQixjQUFuQixBQUFpQyxVQUFVLEFBQzFDO0tBQUksSUFBSSxJQUFSLEFBQVEsQUFBSSxBQUNaO0dBQUEsQUFBRSxxQkFBcUIsWUFBWSxBQUNsQztNQUFJLEVBQUEsQUFBRSxlQUFlLGVBQWpCLEFBQWdDLFFBQVEsRUFBQSxBQUFFLFdBQTlDLEFBQXlELEtBQUssQUFDN0Q7WUFBUyxFQUFULEFBQVcsQUFDWDtBQUNEO0FBSkQsQUFLQTtHQUFBLEFBQUUsS0FBRixBQUFPLE9BQVAsQUFBYyxBQUNkO0FBQ0E7R0FBQSxBQUFFLGVBQUYsQUFBaUIsQUFDakI7R0FBQSxBQUFFLEFBQ0Y7OztBQUVELFNBQUEsQUFBUyxJQUFULEFBQWEsTUFBYixBQUFtQixhQUFuQixBQUFnQyxNQUFoQyxBQUFzQyxVQUFVLEFBQy9DO0tBQUksSUFBSSxJQUFSLEFBQVEsQUFBSSxBQUNaO0dBQUEsQUFBRSxxQkFBcUIsWUFBWSxBQUNsQztNQUFJLEVBQUEsQUFBRSxlQUFlLGVBQWpCLEFBQWdDLFFBQVEsRUFBQSxBQUFFLFdBQTlDLEFBQXlELEtBQUssQUFDN0Q7T0FBQSxBQUFJLFVBQVUsU0FBUyxFQUFULEFBQVcsQUFDekI7QUFDRDtBQUpELEFBS0E7R0FBQSxBQUFFLEtBQUYsQUFBTyxPQUFQLEFBQWMsQUFDZDtLQUFBLEFBQUksVUFBVSxFQUFBLEFBQUUsZUFBRixBQUFpQixBQUMvQjtBQUNBO0dBQUEsQUFBRSxpQkFBRixBQUFtQixnQkFBbkIsQUFBbUMsQUFDbkM7R0FBQSxBQUFFLEtBQUYsQUFBTyxBQUNQOzs7QUFFRDtBQUNBO0FBQ0E7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QUFLLEtBQUssSUFBSSxNQUFwQyxBQUFZLEFBQVUsQUFBYyxBQUFNLEFBQzFDO0FBQ0Q7QUFDQTtRQUFBLEFBQU8sQUFDUDtBQVREOzs7QUNYQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQ3JGQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQ1ZBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQ3JCQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUN0REE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUN2VkE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFD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QUM1QixRQUFHLENBQUMsTUFBSixFQUFXO0FBQ1AsaUJBQVMsU0FBUyxhQUFULENBQXVCLFFBQXZCLENBQVQ7QUFDQSxlQUFPLEtBQVAsR0FBZSxHQUFmO0FBQ0EsZUFBTyxNQUFQLEdBQWdCLEdBQWhCO0FBQ0EsZUFBTyxLQUFQLENBQWEsT0FBYixHQUF1QixNQUF2QjtBQUNBLGVBQU8sT0FBUCxHQUFpQixJQUFqQjtBQUNBLGlCQUFTLElBQVQsQ0FBYyxXQUFkLENBQTBCLE1BQTFCO0FBQ0g7QUFDRCxRQUFJLEtBQUssT0FBTyxVQUFQLENBQWtCLE9BQWxCLEVBQTJCLEVBQUUsV0FBVyxLQUFiLEVBQTNCLEtBQ0EsT0FBTyxVQUFQLENBQWtCLG9CQUFsQixFQUF3QyxFQUFFLFdBQVcsS0FBYixFQUF4QyxDQURUO0FBRUEsUUFBSSxDQUFDLEVBQUwsRUFBUyxNQUFNLGlEQUFOO0FBQ1QsV0FBTyxFQUFQO0FBQ0giLCJmaWxlIjoiZ2VuZXJhdGVkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXNDb250ZW50IjpbIihmdW5jdGlvbiBlKHQsbixyKXtmdW5jdGlvbiBzKG8sdSl7aWYoIW5bb10pe2lmKCF0W29dKXt2YXIgYT10eXBlb2YgcmVxdWlyZT09XCJmdW5jdGlvblwiJiZyZXF1aXJlO2lmKCF1JiZhKXJldHVybiBhKG8sITApO2lmKGkpcmV0dXJuIGkobywhMCk7dmFyIGY9bmV3IEVycm9yKFwiQ2Fubm90IGZpbmQgbW9kdWxlICdcIitvK1wiJ1wiKTt0aHJvdyBmLmNvZGU9XCJNT0RVTEVfTk9UX0ZPVU5EXCIsZn12YXIgbD1uW29dPXtleHBvcnRzOnt9fTt0W29dWzBdLmNhbGwobC5leHBvcnRzLGZ1bmN0aW9uKGUpe3ZhciBuPXRbb11bMV1bZV07cmV0dXJuIHMobj9uOmUpfSxsLGwuZXhwb3J0cyxlLHQsbixyKX1yZXR1cm4gbltvXS5leHBvcnRzfXZhciBpPXR5cGVvZiByZXF1aXJlPT1cImZ1bmN0aW9uXCImJnJlcXVpcmU7Zm9yKHZhciBvPTA7bzxyLmxlbmd0aDtvKyspcyhyW29dKTtyZXR1cm4gc30pIiwidmFyIE1vZGVsID0gcmVxdWlyZShcIi4uL2xpYi9Nb2RlbFwiKSxcblx0VEYgPSByZXF1aXJlKFwiLi4vbm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2luZGV4XCIpLFxuXHRHTCA9IFRGLmNyZWF0ZUdMKCk7XG5cbmZ1bmN0aW9uIEdFVChwYXRoLCByZXNwb25zZVR5cGUsIGNhbGxiYWNrKSB7XG5cdHZhciByID0gbmV3IFhNTEh0dHBSZXF1ZXN0KCk7XG5cdHIub25yZWFkeXN0YXRlY2hhbmdlID0gZnVuY3Rpb24gKCkge1xuXHRcdGlmIChyLnJlYWR5U3RhdGUgPT09IFhNTEh0dHBSZXF1ZXN0LkRPTkUgJiYgci5zdGF0dXMgPT09IDIwMCkge1xuXHRcdFx0Y2FsbGJhY2soci5yZXNwb25zZSk7XG5cdFx0fVxuXHR9O1xuXHRyLm9wZW4oXCJHRVRcIiwgcGF0aCk7XG5cdC8vci5zZXRSZXF1ZXN0SGVhZGVyKFwiQWNjZXB0LUVuY29kaW5nXCIsIFwiZ3ppcCxkZWZsYXRlXCIpO1xuXHRyLnJlc3BvbnNlVHlwZSA9IHJlc3BvbnNlVHlwZTtcblx0ci5zZW5kKCk7XG59XG5cbmZ1bmN0aW9uIFBVVChwYXRoLCBjb250ZW50VHlwZSwgYm9keSwgY2FsbGJhY2spIHtcblx0dmFyIHIgPSBuZXcgWE1MSHR0cFJlcXVlc3QoKTtcblx0ci5vbnJlYWR5c3RhdGVjaGFuZ2UgPSBmdW5jdGlvbiAoKSB7XG5cdFx0aWYgKHIucmVhZHlTdGF0ZSA9PT0gWE1MSHR0cFJlcXVlc3QuRE9ORSAmJiByLnN0YXR1cyA9PT0gMjAwKSB7XG5cdFx0XHRpZiAoY2FsbGJhY2spIGNhbGxiYWNrKHIucmVzcG9uc2UpO1xuXHRcdH1cblx0fVxuXHRyLm9wZW4oXCJQVVRcIiwgcGF0aCk7XG5cdGlmIChjYWxsYmFjaykgci5yZXNwb25zZVR5cGUgPSBjb250ZW50VHlwZTtcblx0Ly9yLnNldFJlcXVlc3RIZWFkZXIoXCJBY2NlcHQtRW5jb2RpbmdcIiwgXCJnemlwLCBkZWZsYXRlXCIpO1xuXHRyLnNldFJlcXVlc3RIZWFkZXIoXCJDb250ZW50LVR5cGVcIiwgY29udGVudFR5cGUpO1xuXHRyLnNlbmQoYm9keSk7XG59XG5cbi8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vXG4vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vL1xuLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy9cbi8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vXG5cblxuLypcblxuXHQxLiBHZXQgbW9kZWwgZnJvbSBzZXJ2ZXJcblx0Mi4gR2V0IHdlaWdodHMgZnJvbSBzZXJ2ZXJcblx0My4gR2V0IGRhdGEgZnJvbSBzZXJ2ZXJcblx0NC4gVHJhaW4gYW5kIHJldHVybiB1cGRhdGVzXG5cblxuKi9cblxuKGZ1bmN0aW9uIG1haW4oKSB7XG5cdHZhciBydW4gPSB0cnVlLFxuXHRcdG5ldCxcblx0XHRtb2RlbCxcblx0XHRpdGVyYXRpb25zLFxuXHRcdHRpbWVzID0ge1xuXHRcdFx0cmVxdWVzdGVkOiBudWxsLFx0Ly8gbW9kZWwgd2VpZ2h0cyBhbmQgZGF0YSByZXF1ZXN0IHNlbnRcblx0XHRcdHJlY2VpdmVkOiBudWxsLFx0XHQvLyBtb2RlbCBkYXRhICYgd2VpZ2h0cyByZWNlaXZlZCBmcm9tIHNlcnZlclxuXHRcdFx0bG9hZGVkOiBudWxsLCBcdFx0Ly8gbW9kZWwgd2VpZ2h0cyBsb2FkZWRcblx0XHRcdHRyYWluZWQ6IG51bGwsIFx0XHQvLyBtb2RlbCBmaW5pc2hlZCB0cmFpbmluZ1xuXHRcdFx0dXBkYXRlZDogbnVsbCBcdFx0Ly8gbW9kZWwgdXBkYXRlcyBzZW50XG5cdFx0fTtcblxuXHRNb2RlbCA9IE1vZGVsKFRGLCBHTCk7XG5cblx0ZnVuY3Rpb24gdXBkYXRlKGFycmF5YnVmZmVyKSB7XG5cblx0XHR2YXIgaXRlcmF0aW9uID0gbmV3IEZsb2F0MzJBcnJheShhcnJheWJ1ZmZlciwgMCwgMSlbMF0sXG5cdFx0XHR0ZXN0VmlldyA9IG5ldyBGbG9hdDMyQXJyYXkoYXJyYXlidWZmZXIpLFxuXHRcdFx0dmlldyxcblx0XHRcdHdlaWdodHMsXG5cdFx0XHRkYXRhLFxuXHRcdFx0bGVuLFxuXHRcdFx0aSxcblx0XHRcdGJhdGNoO1xuXG5cdFx0Ly9jb25zb2xlLmxvZyh0ZXN0Vmlldyk7XG5cdFx0dGltZXMucmVjZWl2ZWQgPSB3aW5kb3cucGVyZm9ybWFuY2Uubm93KCk7XG5cblx0XHR2aWV3ID0gbmV3IEZsb2F0MzJBcnJheShhcnJheWJ1ZmZlciwgNCk7XG5cblxuXHRcdGlmIChpdGVyYXRpb24gPj0gMCkgeyAvLyBpbmNsdWRlcyBuZXcgd2VpZ2h0cyBhbmQgZGF0YVxuXHRcdFx0aXRlcmF0aW9ucyA9IGl0ZXJhdGlvbjtcblx0XHRcdGkgPSBtb2RlbC5zaXplO1xuXHRcdFx0d2VpZ2h0cyA9IHZpZXcuc3ViYXJyYXkoMCwgaSk7XG5cdFx0XHRsZW4gPSB2aWV3W2ldICogbmV0LmxheWVyc1swXS5zaGFwZVsxXTsgLy8gZmlyc3QgZmxvYXQgaXMgbnVtYmVyIG9mIHNhbXBsZXMgaW4gdGhpcyBiYXRjaFxuXHRcdFx0bGVuICs9ICsraTtcblx0XHRcdGJhdGNoID0ge1xuXHRcdFx0XHR4OiB2aWV3LnN1YmFycmF5KGksIGxlbiksXG5cdFx0XHRcdHk6IHZpZXcuc3ViYXJyYXkobGVuKVxuXHRcdFx0fTtcblxuXHRcdFx0bW9kZWwubG9hZCh3ZWlnaHRzKTtcblxuXHRcdH0gZWxzZSB7IC8vIHdlaWdodHMgYXJlIGZyZXNoLCBzbyBkYXRhIG9ubHlcblx0XHRcdGl0ZXJhdGlvbnMrKztcblx0XHRcdGxlbiA9IHZpZXdbMF0gKiBuZXQubGF5ZXJzWzBdLnNoYXBlWzFdOyAvLyBmaXJzdCBmbG9hdCBpcyBudW1iZXIgb2Ygc2FtcGxlcyBpbiB0aGlzIGJhdGNoXG5cdFx0XHRiYXRjaCA9IHtcblx0XHRcdFx0eDogdmlldy5zdWJhcnJheSgxLCArK2xlbiksXG5cdFx0XHRcdHk6IHZpZXcuc3ViYXJyYXkobGVuKVxuXHRcdFx0fTtcblx0XHR9XG5cblx0XHQvLyBUUkFJTlxuXHRcdHRpbWVzLmxvYWRlZCA9IHdpbmRvdy5wZXJmb3JtYW5jZS5ub3coKTtcblx0XHRtb2RlbC50cmFpbihuZXQubGVhcm5pbmdfcmF0ZSwgbmV0Lml0ZXJhdGlvbnMsIGJhdGNoLngsIGJhdGNoLnksIGZ1bmN0aW9uKHdlaWdodHMsIGFjY3VyYWN5KSB7XG5cdFx0XHR2YXIgciA9IDAsIGxvZyA9IFwiXCIsIHcgPSBuZXcgRmxvYXQzMkFycmF5KHdlaWdodHMpO1xuXHRcdFx0dGltZXMudHJhaW5lZCA9IHdpbmRvdy5wZXJmb3JtYW5jZS5ub3coKTtcblx0XHRcdC8vY29uc29sZS5sb2coXCJUaW1lIHRvIHRyYWluOiBcIiArIChkZWx0YSAvIDEwMDApICsgXCIgc2Vjb25kc1wiKTtcblx0XHRcdC8vIHBvc3QgcmVzdWx0cyB0byBzZXJ2ZXJcblx0XHRcdFBVVChcIi4vd2VpZ2h0cy9cIiArIG5ldC5pZCwgXCJhcnJheWJ1ZmZlclwiLCB3ZWlnaHRzLCB1cGRhdGUpO1xuXHRcdFx0ciA9IHdpbmRvdy5wZXJmb3JtYW5jZS5ub3coKTtcblx0XHRcdGxvZyArPSBpdGVyYXRpb25zICsgXCIsXCI7XG5cdFx0XHRsb2cgKz0gYWNjdXJhY3kgKyBcIixcIjtcblx0XHRcdGxvZyArPSB0aW1lcy5yZXF1ZXN0ZWQgKyBcIixcIjtcblx0XHRcdGxvZyArPSB0aW1lcy5yZWNlaXZlZCArIFwiLFwiO1xuXHRcdFx0bG9nICs9IHRpbWVzLmxvYWRlZCArIFwiLFwiO1xuXHRcdFx0bG9nICs9IHRpbWVzLnRyYWluZWQ7XG5cdFx0XHQvLyBzZW5kIHRpbWUgYW5kIHRyYWluaW5nIGxvZyB0byBzZXJ2ZXJcblx0XHRcdFBVVChcIi4vbG9nL1wiICsgbmV0LmlkLCBcInRleHRcIiwgbG9nKTtcblx0XHRcdHRpbWVzLnJlcXVlc3RlZCA9IHI7XG5cdFx0XHRuZXQud2VpZ2h0c192ZXJzaW9uKys7XG5cdFx0fSk7XG5cdH1cblxuXHQvL3ZhciBzZXJ2ZXIgPSBpbygpO1xuXG5cdC8vIHJlcXVlc3QgbW9kZWwgdG8gdHJhaW5cblx0R0VUKFwiLi9tb2RlbFwiLCBcImFwcGxpY2F0aW9uL2pzb25cIiwgZnVuY3Rpb24oanNvbk1vZGVsKSB7XG5cdFx0bmV0ID0gSlNPTi5wYXJzZShqc29uTW9kZWwpO1xuXG5cdFx0bW9kZWwgPSBuZXcgTW9kZWwobmV0LCBudWxsKTtcblx0XHR3aW5kb3cub25iZWZvcmV1bmxvYWQgPSBmdW5jdGlvbigpIHtcblx0XHRcdFBPU1QoXCIuL2Nsb3NlL1wiICsgbmV0LmlkLCBcInN0cmluZ1wiKTtcblx0XHR9O1xuXHRcdHRpbWVzLnJlcXVlc3RlZCA9IHdpbmRvdy5wZXJmb3JtYW5jZS5ub3coKTtcblx0XHRHRVQoXCIuL3dlaWdodHMvXCIgKyBuZXQuaWQsIFwiYXJyYXlidWZmZXJcIiwgdXBkYXRlKTtcblx0fSk7XG59KSgpOyIsInZhciBPdXRwdXQgPSByZXF1aXJlKCcuL2xheWVycy9PdXRwdXQnKSxcblx0RGVuc2UgPSByZXF1aXJlKCcuL2xheWVycy9EZW5zZScpO1xuXG5tb2R1bGUuZXhwb3J0cyA9IGZ1bmN0aW9uKHRlbnNvcmZpcmUsIGdsQ29udGV4dCkge1xuXHRyZXR1cm4ge1xuXHRcdFwiZGVuc2VcIjogRGVuc2UodGVuc29yZmlyZSwgZ2xDb250ZXh0KSxcblx0XHRcIm91dHB1dFwiOiBPdXRwdXQodGVuc29yZmlyZSwgZ2xDb250ZXh0KVxuXHR9O1xufTsiLCJ2YXIgTGF5ZXJzID0gcmVxdWlyZShcIi4vTGF5ZXJzXCIpO1xuXG52YXIgTW9kZWwgPSBmdW5jdGlvbihtb2RlbCwgbGF5ZXJzKSB7XG5cdHRoaXMubGF5ZXJzID0gbmV3IEFycmF5KG1vZGVsLmxheWVycy5sZW5ndGgpO1xuXHR0aGlzLmxvc3MgPSAwLjA7XG5cdHRoaXMuc2l6ZSA9IDAuMDtcblx0dGhpcy5tb2RlbCA9IG1vZGVsO1xuXHR0aGlzLmxvYWQobGF5ZXJzKTtcblxuXHQvL2NvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KHRoaXMubGF5ZXJzWzBdLnNhdmUoKSkpO1xufTtcbk1vZGVsLnByb3RvdHlwZS5ydW4gPSBmdW5jdGlvbihpbnB1dCkge1xuXHR2YXIgb3V0cHV0ID0gaW5wdXQsXG5cdFx0bCA9IC0xO1xuXHR3aGlsZSAoKytsIDwgdGhpcy5sYXllcnMubGVuZ3RoKVxuXHRcdG91dHB1dCA9IHRoaXMubGF5ZXJzW2xdLnJ1bihvdXRwdXQpO1xufTtcbk1vZGVsLnByb3RvdHlwZS5mb3J3YXJkID0gZnVuY3Rpb24ob3V0cHV0KSB7XG5cdC8vY29uc29sZS53YXJuKFwiQ2FsY3Vsb24tIEZvcndhcmQgcGFzc1xcblwiKTtcblx0Ly8gZm9yd2FyZCBwcm9wb2dhdGlvblxuXHR2YXIgbCA9IC0xO1xuXHR3aGlsZSAoKytsIDwgdGhpcy5sYXllcnMubGVuZ3RoKSB7XG5cdFx0b3V0cHV0ID0gdGhpcy5sYXllcnNbbF0ucnVuKG91dHB1dCk7XG5cdFx0Ly9jb25zb2xlLmxvZyhcIkNhbGN1bG9uLSBvdXRwdXQgXCIgKyBsICsgXCI6IFwiICsgb3V0cHV0LnJlYWQoKS5kYXRhKTtcblx0fVxuXHRyZXR1cm4gb3V0cHV0O1xufTtcbk1vZGVsLnByb3RvdHlwZS5iYWNrd2FyZCA9IGZ1bmN0aW9uKG91dHB1dCwgbGVhcm4pIHtcblx0Ly9jb25zb2xlLndhcm4oXCJDYWxjdWxvbi0gQmFja3dhcmQgcGFzc1wiKTtcblx0Ly8gYmFja3dhcmQgcHJvcG9nYXRpb25cblx0dmFyIGwgPSB0aGlzLmxheWVycy5sZW5ndGggLSAxO1xuXHR3aGlsZSAobC0tID4gMCkge1xuXHRcdG91dHB1dCA9IHRoaXMubGF5ZXJzW2xdLnRyYWluKG91dHB1dCwgbGVhcm4pO1xuXHRcdC8vY29uc29sZS5sb2cob3V0cHV0LnJlYWQoKS5kYXRhKTtcblx0fVxufTtcblxuTW9kZWwucHJvdG90eXBlLnZhbGlkYXRlID0gZnVuY3Rpb24oaW5wdXQsIGV4cGVjdCwgY2FsbGJhY2spIHtcblx0dmFyIG91dHB1dCA9IGlucHV0LFxuXHRcdGxvc3NMYXllciA9IHRoaXMubGF5ZXJzW3RoaXMubGF5ZXJzLmxlbmd0aCAtIDFdO1xuXHRvdXRwdXQgPSB0aGlzLmZvcndhcmQob3V0cHV0KTtcblxuXHQvLyBjYWxjdWxhdGUgbG9zc1xuXHRvdXRwdXQgPSBsb3NzTGF5ZXIudHJhaW4oZXhwZWN0KTtcblx0aWYgKHR5cGVvZiBjYWxsYmFjayA9PT0gXCJmdW5jdGlvblwiKSBjYWxsYmFjayhsb3NzTGF5ZXIuYWNjdXJhY3kpXG5cbn1cblxuTW9kZWwucHJvdG90eXBlLnRyYWluID0gZnVuY3Rpb24obGVhcm4sIGl0ZXJhdGlvbnMsIGlucHV0LCBleHBlY3QsIGNhbGxiYWNrKSB7XG5cdHZhciBvdXRwdXQsXG5cdFx0ZSA9IDAsXG5cdFx0bG9zc0xheWVyID0gdGhpcy5sYXllcnNbdGhpcy5sYXllcnMubGVuZ3RoIC0gMV07XG5cdHdoaWxlIChlKysgPCBpdGVyYXRpb25zKSB7XG5cdFx0b3V0cHV0ID0gaW5wdXQ7XG5cdFx0b3V0cHV0ID0gdGhpcy5mb3J3YXJkKG91dHB1dCk7XG5cblx0XHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIG91dHB1dDogXCIgKyBvdXRwdXQucmVhZCgpLmRhdGEpO1xuXHRcdC8vIGNhbGN1bGF0ZSBsb3NzXG5cdFx0b3V0cHV0ID0gbG9zc0xheWVyLnRyYWluKGV4cGVjdCk7XG5cdFx0dGhpcy5sb3NzID0gbG9zc0xheWVyLmFjY3VyYWN5O1xuXHRcdGNvbnNvbGUubG9nKFwiQWNjdXJhY3k6IFwiICsgbG9zc0xheWVyLmFjY3VyYWN5KTtcblxuXHRcdHRoaXMuYmFja3dhcmQob3V0cHV0LCBsZWFybik7XG5cblx0XHQvLyBjaGFuY2UgdG8gc2VuZCBvdXQgZGF0YSBmcm9tIG1vZGVsIChtZXRhZGF0YSBhbmQgbG9nIGRhdGEpXG5cdFx0aWYgKHR5cGVvZiB0aGlzLmFmdGVySXRlcmF0aW9uID09PSBcImZ1bmN0aW9uXCIpIHRoaXMuYWZ0ZXJJdGVyYXRpb24odGhpcywgZSk7XG5cblx0XHQvL2NvbnNvbGUud2FybihcIkNhbGN1bG9uLSBJdGVyYXRpb246IFwiICsgZSArIFwiLCBMb3NzOiBcIiArIHRoaXMubG9zcyk7XG5cdH1cblx0aWYgKHR5cGVvZiBjYWxsYmFjayA9PT0gXCJmdW5jdGlvblwiKSBjYWxsYmFjayh0aGlzLnNhdmUoKSwgdGhpcy5sb3NzKTtcbn1cbk1vZGVsLnByb3RvdHlwZS5zYXZlID0gZnVuY3Rpb24oKSB7XG5cdC8vIFR5cGVkQXJyYXkgdG8gaG9sZCB3ZWlnaHRzLCBiaWFzLCBldGMuIGZyb20gZXZlcnkgbGF5ZXIgb2YgbW9kZWxcblx0dmFyIHdlaWdodHMgPSBuZXcgRmxvYXQzMkFycmF5KHRoaXMuc2l6ZSk7XG5cdFxuXHR2YXIgbCA9IC0xLFxuXHRcdG8gPSAwO1xuXHQvLyBwdWxsIG91dCB0cmFpbmVkIHdlaWdodHMgZm9yIGVhY2ggbGF5ZXJcblx0d2hpbGUgKCsrbCA8ICh0aGlzLmxheWVycy5sZW5ndGggLSAxKSkge1xuXHRcdHdlaWdodHMuc2V0KCB0aGlzLmxheWVyc1tsXS5zYXZlKCksIG8pO1xuXHRcdG8gKz0gdGhpcy5sYXllcnNbbF0uc2l6ZTtcblx0fVxuXHQvL2NvbnNvbGUubG9nKFwid2VpZ2h0czogXCIgKyB3ZWlnaHRzKTtcblx0cmV0dXJuIHdlaWdodHMuYnVmZmVyO1xufTtcbk1vZGVsLnByb3RvdHlwZS5sb2FkID0gZnVuY3Rpb24obGF5ZXJzKSB7XG5cdC8vIGNvbnN0cnVjdCBsYXllcnNcblx0dmFyIG9mZnNldCA9IDAsXG5cdFx0bGF5ZXIsXG5cdFx0bCA9IC0xO1xuXG5cblx0dGhpcy5zaXplID0gMDtcblx0aWYgKGxheWVycyAhPSBudWxsICYmICEobGF5ZXJzIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KSkge1xuXHRcdGxheWVycyA9IG5ldyBGbG9hdDMyQXJyYXkobGF5ZXJzKTtcblx0fVxuXHR3aGlsZSAoKytsIDwgKHRoaXMubGF5ZXJzLmxlbmd0aCAtIDEpKSB7XG5cdFx0bGF5ZXIgPSB0aGlzLm1vZGVsLmxheWVyc1tsXTtcblx0XHRsYXllciA9IG5ldyBMYXllcnNbbGF5ZXIudHlwZV0obGF5ZXIsIGwpO1xuXHRcdHRoaXMuc2l6ZSArPSBsYXllci5zaXplO1xuXHRcdGlmIChsYXllcnMgIT0gbnVsbClcblx0XHRcdG9mZnNldCA9IGxheWVyLmxvYWQobGF5ZXJzLCBvZmZzZXQpO1xuXHRcdGVsc2UgbGF5ZXIucmFuZG9tV2VpZ2h0cygpO1xuXHRcdHRoaXMubGF5ZXJzW2xdID0gbGF5ZXI7XG5cdH1cblx0Ly8gaW5pdGlhbGl6ZSBvdXRwdXQgbGF5ZXJcblx0bGF5ZXIgPSB0aGlzLm1vZGVsLmxheWVyc1tsXTtcblx0bGF5ZXIgPSBuZXcgTGF5ZXJzW2xheWVyLnR5cGVdKGxheWVyLCBsKTtcblx0dGhpcy5sYXllcnNbbF0gPSBsYXllcjtcblxufTtcblxubW9kdWxlLmV4cG9ydHMgPSBmdW5jdGlvbih0ZW5zb3JmaXJlLCBnbENvbnRleHQpIHtcblx0TGF5ZXJzID0gTGF5ZXJzKHRlbnNvcmZpcmUsIGdsQ29udGV4dCk7XG5cdHJldHVybiBNb2RlbDtcbn07IiwibW9kdWxlLmV4cG9ydHMgPSB7XG5cdEFjdGl2YXRpb246IHtcblx0XHRcImxpbmVhclwiOiBgXG5cdFx0XHRvID0gbjtcblx0XHRgLFxuXHRcdC8vIFwiYmluYXJ5XCI6IGBcblx0XHQvLyBcdGlmIChuID4gMC4wKSB7IG8gPSAwLjA7IH0gZWxzZSB7IG8gPSAxLjA7IH1cblx0XHQvLyBgLFxuXHRcdFwicmVsdVwiOiBgXG5cdFx0XHRvID0gbWF4KDAuMCwgbik7XG5cdFx0YCxcblx0XHRcImxyZWx1XCI6IGBcblx0XHRcdGlmIChuID49IDAuMCkgeyBvID0gbjsgfSBlbHNlIHsgbyA9IDAuMDEgKiBuOyB9XG5cdFx0YCxcblx0XHRcInNpZ21vaWRcIjogYFxuXHRcdFx0byA9IDEuMCAvICgxLjAgKyBleHAoMC4wIC0gbikpO1xuXHRcdGAsXG5cdFx0XCJ0YW5oXCI6IGBcblx0XHRcdG8gPSAoMi4wIC8gKDEuMCArIGV4cCgtMi4wICogbikpKSAtIDEuMDtcblx0XHRgLFxuXHRcdFwic29mdHBsdXNcIjogYFxuXHRcdFx0byA9IGxvZygxLjAgKyBleHAobikpO1xuXHRcdGAsXG5cdFx0XCJzb2Z0bWF4XCI6IGBcblx0XHRcdGZsb2F0IGsgPSAwLjA7XG5cdFx0XHRmb3IoaW50IGkgPSAwOyBpIDwgIyhPLnNoYXBlKS54OyBpKyspe1xuXHRcdFx0XHRrICs9IGV4cChPLnJlYWQoaSwgcG9zLnkpKTtcblx0XHRcdH1cblx0XHRcdG8gPSBleHAobikgLyBrO1xuXHRcdGBcblx0fSxcblx0RGVyaXZhdGl2ZToge1xuXHRcdFwibGluZWFyXCI6IGBcblx0XHRcdGQgPSAxLjA7XG5cdFx0YCxcblx0XHQvLyBcImJpbmFyeVwiOiBgXG5cdFx0Ly8gXHRpZiAobyA9PSAwLjApIHtcblx0XHQvLyBcdFx0ZCA9IDAuMDtcblx0XHQvLyBcdH0gZWxzZSB7XG5cdFx0Ly8gXHRcdGQgPSAwLjA7XG5cdFx0Ly8gXHR9XG5cdFx0Ly8gYCxcblx0XHRcInJlbHVcIjogYFxuXHRcdFx0aWYgKG8gPj0gMC4wKSB7XG5cdFx0XHRcdGQgPSAxLjA7XG5cdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHRkID0gMC4wO1xuXHRcdFx0fVxuXHRcdGAsXG5cdFx0XCJscmVsdVwiOiBgXG5cdFx0XHRpZiAobyA+PSAwLjApIHtcblx0XHRcdFx0ZCA9IDEuMDtcblx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdGQgPSAwLjAxO1xuXHRcdFx0fVxuXHRcdGAsXG5cdFx0XCJzaWdtb2lkXCI6IGBcblx0XHRcdGQgPSBvICogKCAxLjAgLSBvICk7XG5cdFx0YCxcblx0XHRcInRhbmhcIjogYFxuXHRcdFx0ZCA9ICggNC4wIC8gcG93KCggZXhwKC1vKSArIGV4cChvKSksIDIuMCkgKTtcblx0XHRgLFxuXHRcdFwic29mdHBsdXNcIjogYFxuXHRcdFx0ZCA9IDEuMCAtICggMS4wIC8gZXhwKG8pICk7XG5cdFx0YCxcblx0XHRcInNvZnRtYXhcIjogYFxuXHRcdFx0ZCA9IG8gKiAoIDEuMCAtIG8gKTtcblx0XHRgIFxuXHR9XG59OyIsInZhciBuZGFycmF5ID0gcmVxdWlyZShcIm5kYXJyYXlcIiksXG5cdFRGLFxuXHRHTCxcblxuXHRHcmFkID0gYFxuXHRcdHVuaWZvcm0gVGVuc29yIE87XG5cdFx0dW5pZm9ybSBUZW5zb3IgRTtcblx0XHRmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykge1xuXHRcdFx0cmV0dXJuIDAuMCAtIEUucmVhZChwb3MpIC8gTy5yZWFkKHBvcyk7XG5cdFx0fVxuXHRgLFxuXHRMb3NzID0gYFxuXHRcdHVuaWZvcm0gVGVuc29yIE87XG5cdFx0dW5pZm9ybSBUZW5zb3IgRTtcblx0XHRmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykge1xuXHRcdFx0ZmxvYXQgbG9zcyA9IDAuMDtcblx0XHRcdGZvcihpbnQgaSA9IDA7IGkgPCAjKE8uc2hhcGUpLnk7IGkrKyl7IC8vIGl0ZXJhdGUgb3ZlciBlYWNoIHNhbXBsZVxuXHRcdFx0XHRmbG9hdCBsID0gMC4wO1xuXHRcdFx0XHRmb3IoaW50IGogPSAwOyBqIDwgIyhPLnNoYXBlKS54OyBqKyspeyAvLyBpdGVyYXRlIG92ZXIgZXZlcnkgb3V0cHV0IGFuZCBjYWxjdWxhdGUgYXZlcmFnZVxuXHRcdFx0XHRcdGwgLT0gRS5yZWFkKGosIGkpICogbG9nKE8ucmVhZChqLCBpKSk7XG5cdFx0XHRcdH1cblx0XHRcdFx0bG9zcyA9IGwgLyBmbG9hdCgjKE8uc2hhcGUpLnkpO1xuXHRcdFx0fVxuXHRcdFx0cmV0dXJuIGxvc3M7XG5cdFx0fVxuXHRgO1xuXG5mdW5jdGlvbiBDcm9zc0VudHJvcHkoKSB7XG5cdC8vIGNhbGN1bGF0ZSBsb3NzIGdyYWRpZW50c1xuXHR0aGlzLmdyYWQgPSBHcmFkO1xuXG5cdC8vIGNhbGN1bGF0ZSBiYXRjaCBhdmVyYWdlIGxvc3Ncblx0dGhpcy5sb3NzRiA9IExvc3M7XG5cblx0dGhpcy5sb3NzID0gbmV3IFRGLk91dHB1dFRlbnNvcihHTCwgWzFdKTtcblx0dGhpcy5vdXRwdXQgPSBudWxsO1xufVxuQ3Jvc3NFbnRyb3B5LnByb3RvdHlwZS5kZWx0YXMgPSBmdW5jdGlvbihvdXRwdXQsIGV4cGVjdCkge1xuXHRpZiAoZXhwZWN0IGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KVxuXHRcdGV4cGVjdCA9IG5ldyBURi5UZW5zb3IoR0wsIG5kYXJyYXkoIGV4cGVjdCwgb3V0cHV0LnNoYXBlKSk7XG5cblx0Ly9jb25zb2xlLmxvZyhcIkNhbGN1bG9uLSBleHBlY3RlZDogXCIgKyBleHBlY3QucmVhZCgpLmRhdGEpO1xuXG5cdHRoaXMub3V0cHV0ID0gbmV3IFRGLk91dHB1dFRlbnNvcihHTCwgb3V0cHV0LnNoYXBlKTtcblx0dGhpcy5vdXRwdXQucnVuKHRoaXMuZ3JhZCwgeyBPOiBvdXRwdXQsIEU6IGV4cGVjdCB9KTtcblx0Ly9jb25zb2xlLmxvZyhcIkNhbGN1bG9uLSBncmFkaWVudDogXCIgKyB0aGlzLm91dHB1dC5yZWFkKCkuZGF0YSk7XG5cblx0dGhpcy5sb3NzLnJ1bih0aGlzLmxvc3NGLCB7IE86IG91dHB1dCwgRTogZXhwZWN0IH0pO1xuXHQvL2NvbnNvbGUubG9nKHRoaXMubG9zcy5yZWFkKCkpO1xuXG5cdHJldHVybiB0aGlzLm91dHB1dDtcbn1cblxubW9kdWxlLmV4cG9ydHMgPSBmdW5jdGlvbih0ZW5zb3JmaXJlLCBnbENvbnRleHQpIHtcblx0VEYgPSB0ZW5zb3JmaXJlO1xuXHRHTCA9IGdsQ29udGV4dDtcblx0cmV0dXJuIENyb3NzRW50cm9weTtcbn07IiwidmFyIG5kYXJyYXkgPSByZXF1aXJlKFwibmRhcnJheVwiKSxcblx0VEYsXG5cdEdMLFxuXG5cdEZ1bmNzID0gcmVxdWlyZSgnLi9BY3RpdmF0aW9ucycpLFxuXHRnZW5lcmF0ZVdlaWdodHMgPSByZXF1aXJlKCcuLi91dGlsL2dlbmVyYXRlV2VpZ2h0cycpLFxuXG5cdEZvcndhcmRCaWFzZWQgPSBgXG5cdFx0dW5pZm9ybSBUZW5zb3IgVzsgLy8gbGF5ZXIgd2VpZ2h0c1xuXHRcdHVuaWZvcm0gVGVuc29yIEk7IC8vIGxheWVyIGlucHV0c1xuXHRcdGZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7IC8vIGZvciBlYWNoIHVuaXQgaW4gb3V0cHV0ICh4OiB1bml0LCB5OiBzYW1wbGUpXG5cdFx0XHRcdGZsb2F0IG4gPSAwLjA7XG5cdFx0XHRcdGZvcihpbnQgaSA9IDA7IGkgPCAjKFcuc2hhcGUpLnk7IGkrKyl7IC8vIGZvciBlYWNoIHdlaWdodFxuXHRcdFx0XHRcdGlmIChpID09ICMoVy5zaGFwZSkueSAtIDEpIHtcblx0XHRcdFx0XHRcdG4gKz0gVy5yZWFkKHBvcy54LCBpKTtcblx0XHRcdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHRcdFx0biArPSBJLnJlYWQoaSwgcG9zLnkpICogVy5yZWFkKHBvcy54LCBpKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblx0XHRcdFx0cmV0dXJuIG47XG5cdFx0fVxuXHRgLFxuXHRGb3J3YXJkVW5iaWFzZWQgPSBgIC8vIGZvciBlYWNoIG91dHB1dCBub2RlXG5cdFx0dW5pZm9ybSBUZW5zb3IgVzsgLy8gbGF5ZXIgd2VpZ2h0c1xuXHRcdHVuaWZvcm0gVGVuc29yIEk7IC8vIGxheWVyIGlucHV0c1xuXHRcdGZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7XG5cdFx0XHRmbG9hdCBuID0gMC4wO1xuXHRcdFx0Zm9yKGludCBpID0gMDsgaSA8ICMoVy5zaGFwZSkueTsgaSsrKXtcblx0XHRcdFx0biArPSBJLnJlYWQoaSwgcG9zLnkpICogVy5yZWFkKHBvcy54LCBpKTtcblx0XHRcdH1cblx0XHRcdHJldHVybiBuO1xuXHRcdH1cblx0YCxcblx0QmFja3dhcmRCaWFzZWQgPSBgIC8vIGZvciBlYWNoIGlucHV0IG5vZGVcblx0XHR1bmlmb3JtIFRlbnNvciBFOyAvLyBsb2NhbCBlcnJvciAoZnJvbSBhY3RpdmF0aW9uKVxuXHRcdHVuaWZvcm0gVGVuc29yIFc7IC8vIHdlaWdodHNcblx0XHRmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykgeyAvLyBwb3NpdGlvbiBpbiBpbnB1dCBncmFkaWVudCBUZW5zb3Jcblx0XHRcdGZsb2F0IGUgPSAwLjA7IC8vIHN1bSBvdXRwdXQgZXJyb3Jcblx0XHRcdGZvcihpbnQgaSA9IDA7IGkgPCAjKEUuc2hhcGUpLng7IGkrKyl7XG5cdFx0XHRcdGlmIChwb3MueSAhPSAjKEUuc2hhcGUpLngpIHtcblx0XHRcdFx0XHRlICs9IFcucmVhZChpLCBwb3MueCkgKiBFLnJlYWQoaSwgcG9zLnkpO1xuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gZTtcblx0XHR9XG5cdGAsXG5cdEJhY2t3YXJkVW5iaWFzZWQgPSBgIC8vIGZvciBlYWNoIGlucHV0IG5vZGVcblx0XHR1bmlmb3JtIFRlbnNvciBFOyAvLyBsb2NhbCBlcnJvciAoZnJvbSBhY3RpdmF0aW9uKVxuXHRcdHVuaWZvcm0gVGVuc29yIFc7IC8vIHdlaWdodHNcblx0XHRmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykgeyAvLyBwb3NpdGlvbiBpbiBpbnB1dCBncmFkaWVudCBUZW5zb3Jcblx0XHRcdGZsb2F0IGUgPSAwLjA7IC8vIHN1bSBvdXRwdXQgZXJyb3Jcblx0XHRcdGZvcihpbnQgaSA9IDA7IGkgPCAjKEUuc2hhcGUpLng7IGkrKyl7XG5cdFx0XHRcdGUgKz0gVy5yZWFkKGksIHBvcy54KSAqIEUucmVhZChpLCBwb3MueSk7XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gZTtcblx0XHR9XG5cdGAsXG5cdFdlaWdodHMgPSBgXG5cdFx0dW5pZm9ybSBUZW5zb3IgRTsgLy8gbG9jYWwgZXJyb3IgKGZyb20gYWN0aXZhdGlvbilcblx0XHR1bmlmb3JtIFRlbnNvciBXOyAvLyB3ZWlnaHRzXG5cdFx0dW5pZm9ybSBUZW5zb3IgSTsgLy8gaW5wdXRcblx0XHR1bmlmb3JtIGZsb2F0IGw7IC8vIGxlYXJuaW5nIHJhdGVcblx0XHRmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykgeyAvLyBwb3MgaW4gd2VpZ2h0cyBUZW5zb3Jcblx0XHRcdGZsb2F0IGUgPSAwLjA7IC8vIGF2ZyBub2RlIGJhdGNoIGVycm9yXG5cdFx0XHRmb3IoaW50IGkgPSAwOyBpIDwgIyhFLnNoYXBlKS55OyBpKyspe1xuXHRcdFx0XHRpZiAocG9zLnkgPT0gIyhJLnNoYXBlKS54KSB7IC8vIGhhbmRsZSBiaWFzIGxheWVyID9cblx0XHRcdFx0XHRlICs9IEUucmVhZChwb3MueCwgaSk7XG5cdFx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdFx0ZSArPSBFLnJlYWQocG9zLngsIGkpICogSS5yZWFkKHBvcy55LCBpKTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdFx0cmV0dXJuIFcucmVhZChwb3MpIC0gKGwgKiBlKTtcblx0XHR9XG5cdGAsXG5cdEFjdGl2YXRpb24gPSAoYWN0aXZhdGlvbkZ1bmN0aW9uKSA9PiBgXG5cdFx0dW5pZm9ybSBUZW5zb3IgTzsgLy8gd2VpZ2h0ZWQgaW5wdXRcblx0XHRmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykge1xuXHRcdFx0ZmxvYXQgbiA9IE8ucmVhZChwb3MpO1xuXHRcdFx0ZmxvYXQgbztcblx0XHRcdCR7IGFjdGl2YXRpb25GdW5jdGlvbiB9XG5cdFx0XHRyZXR1cm4gbztcblx0XHR9XG5cdGAsXG5cdEdyYWRpZW50ID0gKGRlcml2YXRpdmVGdW5jdGlvbikgPT4gYFxuXHRcdHVuaWZvcm0gVGVuc29yIEU7XHQvLyBkb3duc3RyZWFtIGVycm9yXG5cdFx0dW5pZm9ybSBUZW5zb3IgTztcdC8vIGxheWVyIG91dHB1dFxuXHRcdHVuaWZvcm0gVGVuc29yIEg7XHQvLyB3ZWlnaHRlZCBpbnB1dFxuXHRcdGZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7XG5cdFx0XHRmbG9hdCBkO1xuXHRcdFx0ZmxvYXQgbyA9IE8ucmVhZChwb3MpO1xuXHRcdFx0JHsgZGVyaXZhdGl2ZUZ1bmN0aW9uIH1cblx0XHRcdGQgKj0gRS5yZWFkKHBvcyk7XG5cdFx0XHRyZXR1cm4gZDtcblx0XHR9XG5cdGA7XG5cblxuXG5mdW5jdGlvbiBEZW5zZShsYXllciwgaW5kZXgpIHtcblx0dGhpcy5sID0gaW5kZXg7XG5cdC8vIHByb2R1Y2UgT3V0cHV0IFRlbnNvciBnaXZlbiBpbnB1dCwgd2VpZ2h0cywgYW5kIGJpYXMgVGVuc29yc1xuXHR0aGlzLmZvcndhcmQgPSBsYXllci5iaWFzID8gRm9yd2FyZEJpYXNlZCA6IEZvcndhcmRVbmJpYXNlZDtcblxuXHR0aGlzLmFjdGl2YXRpb24gPSBBY3RpdmF0aW9uKEZ1bmNzLkFjdGl2YXRpb25bbGF5ZXIuYWN0aXZhdGlvbl0pO1xuXHRcdFx0XHRcdFx0XG5cdC8vIHByb2R1Y2UgdXBzdHJlYW0gZXJyb3IgVGVuc29yIGdpdmVuIGRvd25zdHJlYW0gZXJyb3IsIGlucHV0LCB3ZWlnaHRzLCBiaWFzXG5cdHRoaXMuYmFja3dhcmQgPSBsYXllci5iaWFzID8gQmFja3dhcmRCaWFzZWQgOiBCYWNrd2FyZFVuYmlhc2VkO1xuXHR0aGlzLmdyYWRpZW50ID0gR3JhZGllbnQoRnVuY3MuRGVyaXZhdGl2ZVtsYXllci5hY3RpdmF0aW9uXSk7XG5cdC8vIGFkanVzdCB3ZWlnaHRzIFRlbnNvciBnaXZlbiBlcnJvciBhbmQgaW5wdXQgVGVuc29yXG5cdHRoaXMudXBkYXRlID0gV2VpZ2h0cztcblxuXHR0aGlzLnNoYXBlID0gbGF5ZXIuc2hhcGU7XG5cdHRoaXMuaW5wdXQgPSBudWxsO1xuXHR0aGlzLm91dHB1dCA9IG51bGw7XG5cdHRoaXMud2VpZ2h0ZWRPdXRwdXQgPSBudWxsO1xuXHR0aGlzLndlaWdodHMgPSBudWxsO1xuXHR0aGlzLmJpYXMgPSBsYXllci5iaWFzO1xuXHR0aGlzLnNpemUgPSB0aGlzLnNoYXBlWzBdICogdGhpcy5zaGFwZVsxXSArICh0aGlzLmJpYXMgPyB0aGlzLnNoYXBlWzBdIDogMCk7XG5cbn1cbkRlbnNlLnByb3RvdHlwZS5sb2FkID0gZnVuY3Rpb24oYXJyYXksIG9mZnNldCkge1xuXHR2YXIgbGVuZ3RoID0gdGhpcy5zaXplO1xuXHQvLyByZWFkIGluIHdlaWdodHMgKGFuZCBiaWFzKVxuXHR0aGlzLndlaWdodHMgPSBuZXcgVEYuSW5QbGFjZVRlbnNvcihHTCwgbmRhcnJheSggYXJyYXkuc3ViYXJyYXkob2Zmc2V0LCBvZmZzZXQgKyBsZW5ndGgpLCBbdGhpcy5zaGFwZVswXSwgdGhpcy5zaGFwZVsxXSArICh0aGlzLmJpYXMgPyAxIDogMCldICkgKTtcblx0b2Zmc2V0ICs9IGxlbmd0aDtcblx0cmV0dXJuIG9mZnNldDtcbn1cbkRlbnNlLnByb3RvdHlwZS5yYW5kb21XZWlnaHRzID0gZnVuY3Rpb24oKSB7XG5cdHRoaXMud2VpZ2h0cyA9IG5ldyBURi5JblBsYWNlVGVuc29yKEdMLCBcblx0XHRuZGFycmF5KFxuXHRcdFx0Z2VuZXJhdGVXZWlnaHRzKHRoaXMuc2hhcGUsICh0aGlzLmJpYXMgPyB0aGlzLnNoYXBlWzBdIDogMCkpLCAvLyB2YWx1ZXNcblx0XHRcdFt0aGlzLnNoYXBlWzBdLCB0aGlzLnNoYXBlWzFdICsgKHRoaXMuYmlhcyA/IDEgOiAwKV0gLy8gc2hhcGVcblx0XHQpXG5cdCk7XG59XG5EZW5zZS5wcm90b3R5cGUuc2F2ZSA9IGZ1bmN0aW9uKCkge1xuXHRyZXR1cm4gdGhpcy53ZWlnaHRzLnJlYWQoKS5kYXRhO1xufVxuRGVuc2UucHJvdG90eXBlLnJ1biA9IGZ1bmN0aW9uKGlucHV0KSB7XG5cdGlmIChpbnB1dCBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkge1xuXHRcdHRoaXMuaW5wdXQgPSBuZXcgVEYuVGVuc29yKEdMLCBuZGFycmF5KCBpbnB1dCwgWyB0aGlzLnNoYXBlWzFdLCAoaW5wdXQubGVuZ3RoIC8gdGhpcy5zaGFwZVsxXSkgPj4gMCBdKSk7XG5cdH0gZWxzZSB0aGlzLmlucHV0ID0gaW5wdXQ7XG5cdC8vY29uc29sZS5sb2codGhpcy5pbnB1dC5zaGFwZSk7XG5cdC8vY29uc29sZS5sb2coXCJDYWxjdWxvbi0gaW5wdXQgXCIgKyB0aGlzLmwgKyBcIjogXCIgKyB0aGlzLmlucHV0LnJlYWQoKS5kYXRhKTtcblx0Ly9jb25zb2xlLmxvZyhcIkNhbGN1bG9uLSB3ZWlnaHRzIFwiICsgdGhpcy5sICsgXCI6IFwiICsgdGhpcy53ZWlnaHRzLnJlYWQoKS5kYXRhKTtcblxuXHR0aGlzLndlaWdodGVkT3V0cHV0ID0gbmV3IFRGLk91dHB1dFRlbnNvcihHTCwgWyB0aGlzLnNoYXBlWzBdLCB0aGlzLmlucHV0LnNoYXBlWzFdIF0pO1xuXHR0aGlzLndlaWdodGVkT3V0cHV0LnJ1bih0aGlzLmZvcndhcmQsIHtXOiB0aGlzLndlaWdodHMsIEk6IHRoaXMuaW5wdXR9KTtcblxuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIHdlaWdodGVkT3V0cHV0IFwiICsgdGhpcy5sICsgXCI6IFwiICsgdGhpcy53ZWlnaHRlZE91dHB1dC5yZWFkKCkuZGF0YSk7XG5cblx0dGhpcy5vdXRwdXQgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBbIHRoaXMuc2hhcGVbMF0sIHRoaXMuaW5wdXQuc2hhcGVbMV0gXSk7XG5cdHRoaXMub3V0cHV0LnJ1bih0aGlzLmFjdGl2YXRpb24sIHtPOiB0aGlzLndlaWdodGVkT3V0cHV0fSk7XG5cblx0Ly9jb25zb2xlLmxvZyhcIm91dHB1dCBcIiArIHRoaXMubDogXCIgKyB0aGlzLm91dHB1dC5yZWFkKCkuZGF0YSk7XG5cdHJldHVybiB0aGlzLm91dHB1dDtcbn07XG5EZW5zZS5wcm90b3R5cGUudHJhaW4gPSBmdW5jdGlvbihlcnJvciwgbGVhcm5pbmdfcmF0ZSkge1xuXHR0aGlzLnBhcnRpYWwgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCB0aGlzLmlucHV0LnNoYXBlKTtcblx0dGhpcy5sb2NhbCA9IG5ldyBURi5PdXRwdXRUZW5zb3IoR0wsIHRoaXMub3V0cHV0LnNoYXBlKTtcblxuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIGVycm9yOiBcIiArIGVycm9yLnJlYWQoKS5kYXRhKTtcblx0Ly9jb25zb2xlLmxvZyhcIkNhbGN1bG9uLSB3ZWlnaHRzIFwiICsgdGhpcy5sOiBcIiArIHRoaXMud2VpZ2h0cy5yZWFkKCkuZGF0YSk7XG5cblx0Ly8gY2FsY3VsYXRlIGxvY2FsIGVycm9yIGZyb20gd2VpZ2h0ZWRPdXRwdXQgKHN0cmlwcyBvdXQgZXJyb3IgZnJvbSBhY3RpdmF0aW9uIGZ1bmN0aW9uKVxuXHR0aGlzLmxvY2FsLnJ1bih0aGlzLmdyYWRpZW50LCB7RTogZXJyb3IsIE86IHRoaXMub3V0cHV0LCBIOiB0aGlzLndlaWdodGVkT3V0cHV0fSk7XG5cdC8vY29uc29sZS5sb2coXCJDYWxjdWxvbi0gbG9jYWxFOiBcIiArIGxvY2FsLnJlYWQoKS5kYXRhKTtcblxuXHQvLyBjYWxjdWxhdGUgdXBzdHJlYW0gZXJyb3JzIGZyb20gaW5wdXRcblx0dGhpcy5wYXJ0aWFsLnJ1bih0aGlzLmJhY2t3YXJkLCB7RTogdGhpcy5sb2NhbCwgVzogdGhpcy53ZWlnaHRzfSk7XG5cblx0Ly8gdHJhaW4gd2VpZ2h0cyBiYXNlZCBvbiBsb2NhbCBlcnJvclxuXHR0aGlzLndlaWdodHMucnVuKHRoaXMudXBkYXRlLCB7VzogdGhpcy53ZWlnaHRzLCBFOiB0aGlzLmxvY2FsLCBJOiB0aGlzLmlucHV0LCBsOiBsZWFybmluZ19yYXRlfSk7XG5cdC8vY29uc29sZS5sb2coXCJDYWxjdWxvbi0gdXBkYXRlZCBcIiArIHRoaXMubDogXCIgKyB0aGlzLndlaWdodHMucmVhZCgpLmRhdGEpO1xuXG5cdHJldHVybiB0aGlzLnBhcnRpYWw7XG59O1xuXG5tb2R1bGUuZXhwb3J0cyA9IGZ1bmN0aW9uKHRlbnNvcmZpcmUsIGdsQ29udGV4dCkge1xuXHRURiA9IHRlbnNvcmZpcmU7XG5cdEdMID0gZ2xDb250ZXh0O1xuXHRyZXR1cm4gRGVuc2U7XG59OyIsInZhciBuZGFycmF5ID0gcmVxdWlyZShcIm5kYXJyYXlcIiksXG5cdFRGLFxuXHRHTCxcblxuXHRHcmFkIFx0PSBgXG5cdFx0dW5pZm9ybSBUZW5zb3IgTztcblx0XHR1bmlmb3JtIFRlbnNvciBFO1xuXHRcdGZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7XG5cdFx0XHRyZXR1cm4gTy5yZWFkKHBvcykgLSBFLnJlYWQocG9zKTtcblx0XHR9XG5cdGAsXG5cblx0QWNjdXJhY3kgPSBgXG5cdFx0dW5pZm9ybSBUZW5zb3IgTztcblx0XHR1bmlmb3JtIFRlbnNvciBFO1xuXHRcdGZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7XG5cdFx0XHRmbG9hdCBzID0gMC4wO1xuXHRcdFx0Zm9yIChpbnQgaSA9IDA7IGkgPCAjKE8uc2hhcGUpLng7IGkrKykgeyAvLyBpdGVyYXRlIG92ZXIgZXZlcnkgb3V0cHV0XG5cdFx0XHRcdHMgKz0gcG93KChFLnJlYWQoaSwgcG9zLngpIC0gTy5yZWFkKGksIHBvcy54KSksIDIuMCk7XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gMS4wIC0gY2xhbXAocyAvIGZsb2F0KCMoTy5zaGFwZSkueCksIDAuMCwgMS4wKTtcblx0XHR9XG5cdGAsXG5cblx0TG9zcyBcdD0gYFxuXHRcdHVuaWZvcm0gVGVuc29yIEc7XG5cdFx0ZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHtcblx0XHRcdGZsb2F0IGxvc3MgPSAwLjA7XG5cdFx0XHRmb3IoaW50IGkgPSAwOyBpIDwgIyhHLnNoYXBlKS55OyBpKyspeyAvLyBpdGVyYXRlIG92ZXIgZWFjaCBzYW1wbGVcblx0XHRcdFx0ZmxvYXQgbCA9IDAuMDtcblx0XHRcdFx0Zm9yKGludCBqID0gMDsgaiA8ICMoRy5zaGFwZSkueDsgaisrKXsgLy8gaXRlcmF0ZSBvdmVyIGV2ZXJ5IG91dHB1dCBhbmQgY2FsY3VsYXRlIGF2ZXJhZ2Vcblx0XHRcdFx0XHRsICs9IHBvdyhmbG9hdChHLnJlYWQoaiwgaSkpLCAyLjApO1xuXHRcdFx0XHR9XG5cdFx0XHRcdGxvc3MgKz0gbCAvIGZsb2F0KCMoRy5zaGFwZSkueSk7XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gbG9zcztcblx0XHR9XG5cdGA7XG5cbmZ1bmN0aW9uIE1TRSgpIHtcblx0Ly8gY2FsY3VsYXRlIGxvc3MgZ3JhZGllbnRzXG5cdHRoaXMuZ3JhZCA9IEdyYWQ7XG5cblx0Ly8gY2FsY3VsYXRlIGJhdGNoIGF2ZXJhZ2UgbG9zc1xuXHR0aGlzLmxvc3NGID0gTG9zcztcblxuXHR0aGlzLkFjY3VyYWN5ID0gQWNjdXJhY3k7XG5cblx0dGhpcy5sb3NzID0gbmV3IFRGLk91dHB1dFRlbnNvcihHTCwgWzFdKTtcblx0dGhpcy5vdXRwdXQgPSBudWxsO1xufVxuTVNFLnByb3RvdHlwZS5kZWx0YXMgPSBmdW5jdGlvbihvdXRwdXQsIGV4cGVjdCkge1xuXHRpZiAoZXhwZWN0IGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KVxuXHRcdGV4cGVjdCA9IG5ldyBURi5UZW5zb3IoR0wsIG5kYXJyYXkoIGV4cGVjdCwgb3V0cHV0LnNoYXBlKSk7XG5cblx0Ly9jb25zb2xlLmxvZyhcIkNhbGN1bG9uLSBleHBlY3RlZDogXCIgKyBleHBlY3QucmVhZCgpLmRhdGEpO1xuXG5cdHRoaXMub3V0cHV0ID0gbmV3IFRGLk91dHB1dFRlbnNvcihHTCwgb3V0cHV0LnNoYXBlKTtcblx0dGhpcy5vdXRwdXQucnVuKHRoaXMuZ3JhZCwgeyBPOiBvdXRwdXQsIEU6IGV4cGVjdCB9KTtcblx0Ly9jb25zb2xlLmxvZyhcIkNhbGN1bG9uLSBncmFkaWVudDogXCIgKyB0aGlzLm91dHB1dC5yZWFkKCkuZGF0YSk7XG5cblx0dGhpcy5sb3NzLnJ1bih0aGlzLmxvc3NGLCB7IEc6IHRoaXMub3V0cHV0IH0pO1xuXG5cdHJldHVybiB0aGlzLm91dHB1dDtcbn1cblxubW9kdWxlLmV4cG9ydHMgPSBmdW5jdGlvbih0ZW5zb3JmaXJlLCBnbENvbnRleHQpIHtcblx0VEYgPSB0ZW5zb3JmaXJlO1xuXHRHTCA9IGdsQ29udGV4dDtcblx0cmV0dXJuIE1TRTtcbn07IiwidmFyIG5kYXJyYXkgPSByZXF1aXJlKFwibmRhcnJheVwiKSxcblx0VEYsXG5cdEdMLFxuXHRTb2Z0bWF4ID0gcmVxdWlyZSgnLi9Tb2Z0bWF4JyksXG5cdE1TRSA9IHJlcXVpcmUoJy4vTVNFJyksXG5cdENyb3NzRW50cm9weSA9IHJlcXVpcmUoJy4vQ3Jvc3NFbnRyb3B5JyksXG5cblx0QWNjU3VtXHQ9IGBcblx0XHR1bmlmb3JtIFRlbnNvciBBO1xuXHRcdGZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7XG5cdFx0XHRmbG9hdCBhY2MgPSAwLjA7XG5cdFx0XHRmb3IgKGludCBpID0gMDsgaSA8ICMoQS5zaGFwZSkueTsgaSsrKXsgLy8gaXRlcmF0ZSBvdmVyIGVhY2ggc2FtcGxlXG5cdFx0XHRcdGFjYyArPSBBLnJlYWQocG9zLngsIGkpO1xuXHRcdFx0fVxuXHRcdFx0cmV0dXJuIGFjYyAvIGZsb2F0KCMoQS5zaGFwZSkueSk7XG5cdFx0fVxuXHRgO1xuXG5mdW5jdGlvbiBPdXRwdXQobGF5ZXIsIGluZGV4KSB7XG5cdHRoaXMub3V0cHV0ID0gbnVsbDtcblx0aWYgKGxheWVyLmFjdGl2YXRpb24gPT09IFwic29mdG1heFwiICYmIGxheWVyLmxvc3MgPT09IFwieGVudHJvcHlcIikge1xuXHRcdHRoaXMub3V0cHV0ID0gbmV3IFNvZnRtYXgobGF5ZXIsIGluZGV4KTtcblx0XHR0aGlzLnJ1biA9IChpbnB1dCkgPT4ge1xuXHRcdFx0dGhpcy5fb3V0cHV0ID0gdGhpcy5vdXRwdXQucnVuKGlucHV0KTtcblx0XHRcdHJldHVybiB0aGlzLl9vdXRwdXQ7XG5cdFx0fTtcblx0fSBlbHNlIHtcblx0XHRzd2l0Y2ggKGxheWVyLmxvc3MpIHtcblx0XHRcdGNhc2UgXCJ4ZW50cm9weVwiOlxuXHRcdFx0XHR0aGlzLm91dHB1dCA9IG5ldyBDcm9zc0VudHJvcHkoKTtcblx0XHRcdFx0YnJlYWs7XG5cdFx0XHRjYXNlIFwibXNlXCI6XG5cdFx0XHRcdHRoaXMub3V0cHV0ID0gbmV3IE1TRSgpO1xuXHRcdFx0XHRicmVhaztcblx0XHR9XG5cdFx0dGhpcy5ydW4gPSB0aGlzLnJ1bi5iaW5kKHRoaXMpO1xuXHRcdHRoaXMudHJhaW4gPSB0aGlzLnRyYWluLmJpbmQodGhpcyk7XG5cdH1cblxuXHR0aGlzLl9vdXRwdXQgPSBudWxsO1xuXHR0aGlzLmFjY3VyYWN5ID0gMDtcbn1cbk91dHB1dC5wcm90b3R5cGUucnVuID0gZnVuY3Rpb24oaW5wdXQpIHtcblx0dGhpcy5fb3V0cHV0ID0gaW5wdXQ7XG5cdHJldHVybiBpbnB1dDtcbn07XG5PdXRwdXQucHJvdG90eXBlLnRyYWluID0gZnVuY3Rpb24oZXhwZWN0ZWQpIHtcblx0Ly9jb25zb2xlLmxvZyhcIkV4cGVjdGVkOiBcIiArIGV4cGVjdGVkKTtcblx0aWYgKGV4cGVjdGVkIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KVxuXHRcdGV4cGVjdGVkID0gbmV3IFRGLlRlbnNvcihHTCwgbmRhcnJheSggZXhwZWN0ZWQsIHRoaXMuX291dHB1dC5zaGFwZSkpO1xuXG5cdFxuXHQvL2NvbnNvbGUubG9nKFwiICBPdXRwdXQ6IFwiICsgdGhpcy5fb3V0cHV0LnJlYWQoKS5kYXRhKTtcblxuXHR0aGlzLmJhdGNoQWNjdXJhY3kgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBbMSwgdGhpcy5fb3V0cHV0LnNoYXBlWzFdXSk7XG5cdHRoaXMuX2FjY3VyYWN5ID0gbmV3IFRGLk91dHB1dFRlbnNvcihHTCwgWzFdKTtcblx0dGhpcy5iYXRjaEFjY3VyYWN5LnJ1bih0aGlzLm91dHB1dC5BY2N1cmFjeSwgeyBPOiB0aGlzLl9vdXRwdXQsIEU6IGV4cGVjdGVkIH0pO1xuXHR0aGlzLl9hY2N1cmFjeS5ydW4oQWNjU3VtLCB7IEE6IHRoaXMuYmF0Y2hBY2N1cmFjeSB9KTtcblx0dGhpcy5hY2N1cmFjeSA9IHRoaXMuX2FjY3VyYWN5LnJlYWQoKS5kYXRhWzBdO1xuXG5cdHJldHVybiB0aGlzLm91dHB1dC5kZWx0YXModGhpcy5fb3V0cHV0LCBleHBlY3RlZCk7XG59O1xuXG5cbm1vZHVsZS5leHBvcnRzID0gZnVuY3Rpb24odGVuc29yZmlyZSwgZ2xDb250ZXh0KSB7XG5cblx0VEYgPSB0ZW5zb3JmaXJlO1xuXHRHTCA9IGdsQ29udGV4dDtcblxuXHRTb2Z0bWF4ID0gU29mdG1heCh0ZW5zb3JmaXJlLCBnbENvbnRleHQpO1xuXHRNU0UgPSBNU0UodGVuc29yZmlyZSwgZ2xDb250ZXh0KTtcblx0Q3Jvc3NFbnRyb3B5ID0gQ3Jvc3NFbnRyb3B5KHRlbnNvcmZpcmUsIGdsQ29udGV4dCk7XG5cblx0cmV0dXJuIE91dHB1dDtcbn07IiwidmFyIG5kYXJyYXkgPSByZXF1aXJlKFwibmRhcnJheVwiKSxcblx0VEYsXG5cdEdMLFxuXG5cdEFjdCA9IGBcblx0XHR1bmlmb3JtIFRlbnNvciBJOyAvLyBpbnB1dFxuXHRcdGZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7XG5cdFx0XHRmbG9hdCBrID0gMC4wO1xuXHRcdFx0Zm9yKGludCBpID0gMDsgaSA8ICMoSS5zaGFwZSkueDsgaSsrKXtcblx0XHRcdFx0ayArPSBleHAoSS5yZWFkKGksIHBvcy55KSk7XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gZXhwKEkucmVhZChwb3MpKSAvIGs7XG5cdFx0fVxuXHRgLFxuXHRHcmFkID0gYFxuXHRcdHVuaWZvcm0gVGVuc29yIE87IC8vIFNvZnRtYXggb3V0cHV0XG5cdFx0dW5pZm9ybSBUZW5zb3IgRTsgLy8gZXhwZWN0ZWQgb3V0cHV0XG5cdFx0ZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHtcblx0XHRcdHJldHVybiBPLnJlYWQocG9zKSAtIEUucmVhZChwb3MpO1xuXHRcdH1cblx0YCxcblxuXHRBY2N1cmFjeSA9IGBcblx0XHR1bmlmb3JtIFRlbnNvciBPO1xuXHRcdHVuaWZvcm0gVGVuc29yIEU7XG5cdFx0ZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHtcblx0XHRcdGZsb2F0IHMgPSAwLjA7XG5cdFx0XHRmb3IgKGludCBpID0gMDsgaSA8ICMoTy5zaGFwZSkueDsgaSsrKSB7IC8vIGl0ZXJhdGUgb3ZlciBldmVyeSBvdXRwdXRcblx0XHRcdFx0cyArPSBwb3coKEUucmVhZChpLCBwb3MueCkgLSBPLnJlYWQoaSwgcG9zLngpKSwgMi4wKTtcblx0XHRcdH1cblx0XHRcdHJldHVybiAxLjAgLSBjbGFtcChzIC8gZmxvYXQoIyhPLnNoYXBlKS54KSwgMC4wLCAxLjApO1xuXHRcdH1cblx0YCxcblxuXHQvLyBBY2N1cmFjeSA9IGBcblx0Ly8gXHR1bmlmb3JtIFRlbnNvciBPO1xuXHQvLyBcdHVuaWZvcm0gVGVuc29yIEU7XG5cdC8vIFx0ZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHtcblx0Ly8gXHRcdGludCBsID0gIyhPLnNoYXBlKS54O1xuXHQvLyBcdFx0ZmxvYXQgdiA9IC0xMDAwMDAuMDtcblx0Ly8gXHRcdGZvciAoaW50IGkgPSAwOyBpIDwgIyhPLnNoYXBlKS54OyBpKyspIHsgLy8gaXRlcmF0ZSBvdmVyIGV2ZXJ5IG91dHB1dFxuXHQvLyBcdFx0XHRpZiAoTy5yZWFkKGksIHBvcy54KSA+IHYpIHtcblx0Ly8gXHRcdFx0XHR2ID0gTy5yZWFkKGksIHBvcy55KTsgLyogZ2V0IGxhcmdlc3QgY2F0ZWdvcnkgaW4gb3V0cHV0ICovXG5cdC8vIFx0XHRcdFx0bCA9IGk7IC8qIHNhdmUgaW5kZXggb2YgbGFyZ2VzdCB2YWx1ZSAqL1xuXHQvLyBcdFx0XHR9XG5cdC8vIFx0XHR9XG5cdC8vIFx0XHRpZiAoRS5yZWFkKGwsIHBvcy55KSA+IDAuOSkge1xuXHQvLyBcdFx0XHRyZXR1cm4gMS4wO1xuXHQvLyBcdFx0fSBlbHNlIHtcblx0Ly8gXHRcdFx0cmV0dXJuIDAuMDtcblx0Ly8gXHRcdH1cblx0Ly8gXHR9XG5cdC8vIGAsXG5cblxuXHRMb3NzID0gYFxuXHRcdHVuaWZvcm0gVGVuc29yIEc7XG5cdFx0ZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHtcblx0XHRcdGZsb2F0IGxvc3MgPSAwLjA7XG5cdFx0XHRmb3IoaW50IGkgPSAwOyBpIDwgIyhHLnNoYXBlKS55OyBpKyspeyAvKiBpdGVyYXRlIG92ZXIgZWFjaCBzYW1wbGUgKi9cblx0XHRcdFx0ZmxvYXQgbCA9IDAuMDtcblx0XHRcdFx0Zm9yKGludCBqID0gMDsgaiA8ICMoRy5zaGFwZSkueDsgaisrKXsgLyogaXRlcmF0ZSBvdmVyIGV2ZXJ5IG91dHB1dCBhbmQgY2FsY3VsYXRlIGF2ZXJhZ2UgKi9cblx0XHRcdFx0XHRsICs9IHBvdyhmbG9hdChHLnJlYWQoaiwgaSkpLCAyLjApIC8gZmxvYXQoIyhHLnNoYXBlKS54KTtcblx0XHRcdFx0fVxuXHRcdFx0XHRsb3NzICs9IGwgLyBmbG9hdCgjKEcuc2hhcGUpLnkpO1xuXHRcdFx0fVxuXHRcdFx0cmV0dXJuIGxvc3M7XG5cdFx0fVxuXHRgO1xuXG5cbmZ1bmN0aW9uIFNvZnRtYXgobGF5ZXIsIGluZGV4KSB7XG5cdHRoaXMubCA9IGluZGV4O1xuXG5cdHRoaXMuYWN0aXZhdGlvbiA9IEFjdDtcblxuXHR0aGlzLmdyYWRpZW50ID0gR3JhZDtcblx0dGhpcy5sb3NzRiA9IExvc3M7XHRcblxuXHR0aGlzLkFjY3VyYWN5ID0gQWNjdXJhY3k7XG5cblxuXHR0aGlzLmlucHV0ID0gbnVsbDtcblx0dGhpcy5vdXRwdXQgPSBudWxsO1xuXHR0aGlzLmxvc3MgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBbMV0pO1xufVxuU29mdG1heC5wcm90b3R5cGUucnVuID0gZnVuY3Rpb24oaW5wdXQpIHtcblxuXHR0aGlzLmlucHV0ID0gaW5wdXQ7XG5cdHRoaXMub3V0cHV0ID0gbmV3IFRGLk91dHB1dFRlbnNvcihHTCwgdGhpcy5pbnB1dC5zaGFwZSk7XG5cblx0dGhpcy5vdXRwdXQucnVuKHRoaXMuYWN0aXZhdGlvbiwge0k6IHRoaXMuaW5wdXR9KTtcblxuXHQvL2NvbnNvbGUubG9nKFwib3V0cHV0IFwiICsgdGhpcy5sICsgXCI6IFwiICsgdGhpcy5vdXRwdXQucmVhZCgpLmRhdGEpO1xuXHRyZXR1cm4gdGhpcy5vdXRwdXQ7XG59XG5Tb2Z0bWF4LnByb3RvdHlwZS5kZWx0YXMgPSBmdW5jdGlvbihvdXRwdXQsIGV4cGVjdGVkKSB7XG5cdHRoaXMucGFydGlhbCA9IG5ldyBURi5PdXRwdXRUZW5zb3IoR0wsIHRoaXMuaW5wdXQuc2hhcGUpO1xuXG5cdGlmIChleHBlY3RlZCBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSlcblx0XHRleHBlY3RlZCA9IG5ldyBURi5UZW5zb3IoR0wsIG5kYXJyYXkoZXhwZWN0ZWQsIHRoaXMuaW5wdXQuc2hhcGUpKTtcblxuXHQvLyBjYWxjdWxhdGUgdXBzdHJlYW0gZXJyb3JzXG5cdHRoaXMucGFydGlhbC5ydW4odGhpcy5ncmFkaWVudCwge086IG91dHB1dCwgRTogZXhwZWN0ZWR9KTtcblxuXHQvLyBjYWxjdWxhdGUgYmF0Y2ggdHJhaW5pbmcgbG9zc1xuXHR0aGlzLmxvc3MucnVuKHRoaXMubG9zc0YsIHsgRzogdGhpcy5wYXJ0aWFsIH0pO1xuXHQvL2NvbnNvbGUubG9nKHRoaXMubG9zcy5yZWFkKCkpO1xuXG5cdC8vIGNvbnNvbGUubG9nKG91dHB1dC5yZWFkKCkuZGF0YSk7XG5cblx0cmV0dXJuIHRoaXMucGFydGlhbDtcbn1cblxubW9kdWxlLmV4cG9ydHMgPSBmdW5jdGlvbih0ZW5zb3JmaXJlLCBnbENvbnRleHQpIHtcblx0VEYgPSB0ZW5zb3JmaXJlO1xuXHRHTCA9IGdsQ29udGV4dDtcblx0cmV0dXJuIFNvZnRtYXg7XG59IiwiLy8gU3RhbmRhcmQgTm9ybWFsIHZhcmlhdGUgdXNpbmcgQm94LU11bGxlciB0cmFuc2Zvcm0uXG5mdW5jdGlvbiByYW5kb20obWVhbiwgc3RkRGV2KSB7XG5cdG1lYW4gPSBtZWFuIHx8IDA7XG5cdHN0ZERldiA9IHN0ZERldiB8fCAxO1xuICAgIHZhciB1ID0gMCwgdiA9IDA7XG4gICAgd2hpbGUodSA9PT0gMCkgdSA9IE1hdGgucmFuZG9tKCk7IC8vQ29udmVydGluZyBbMCwxKSB0byAoMCwxKVxuICAgIHdoaWxlKHYgPT09IDApIHYgPSBNYXRoLnJhbmRvbSgpO1xuICAgIC8vcmV0dXJuIDAuNDtcbiAgICByZXR1cm4gKE1hdGguc3FydCggLTIuMCAqIE1hdGgubG9nKCB1ICkgKSAqIE1hdGguY29zKCAyLjAgKiBNYXRoLlBJICogdiApKSAqIHN0ZERldiArIG1lYW47XG59XG5cbm1vZHVsZS5leHBvcnRzID0gZnVuY3Rpb24gZ2VuZXJhdGVXZWlnaHRzKHNoYXBlLCBiaWFzKSB7XG5cdHZhciByZXN1bHQgPSBuZXcgRmxvYXQzMkFycmF5KHNoYXBlWzBdICogc2hhcGVbMV0gKyBiaWFzKTtcblx0Y29uc29sZS5sb2coXCJMYXllciB3ZWlnaHRzICsgYmlhczogXCIgKyByZXN1bHQubGVuZ3RoKTtcblx0dmFyIGwgPSAtMTtcblx0d2hpbGUgKCsrbCA8IHJlc3VsdC5sZW5ndGgpIHtcblx0XHRyZXN1bHRbbF0gPSByYW5kb20oMCwgTWF0aC5zcXJ0KDIgLyBzaGFwZVsxXSkpO1xuXHR9XG5cdC8vY29uc29sZS5sb2cocmVzdWx0WzBdKTtcblx0cmV0dXJuIHJlc3VsdDtcbn0iLCJ2YXIgc3ByaW50ZiA9IHJlcXVpcmUoJ3NwcmludGYnKTtcbm1vZHVsZS5leHBvcnRzID0gZm9ybWF0O1xuXG5mdW5jdGlvbiBmb3JtYXQgKHgsIGJ5dGVzKSB7XG4gICAgaWYgKGJ5dGVzID09PSB1bmRlZmluZWQpIGJ5dGVzID0gODtcbiAgICB2YXIgcmZtdCA9ICclJyArIGJ5dGVzICsgJy4nICsgYnl0ZXMgKyAncyc7XG4gICAgXG4gICAgaWYgKGJ5dGVzIDw9IDApIHJldHVybiB1bmRlZmluZWQ7XG4gICAgaWYgKGlzTmFOKHgpKSByZXR1cm4gc3ByaW50ZihyZm10LCAnTmFOJyk7XG4gICAgaWYgKHggPT09IEluZmluaXR5KSB7XG4gICAgICAgIGlmIChieXRlcyA9PT0gMSkgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICAgICAgcmV0dXJuIHNwcmludGYocmZtdCwgYnl0ZXMgPj0gOSA/ICdJbmZpbml0eScgOiAnIEluZicpLnNsaWNlKDAsIGJ5dGVzKTtcbiAgICB9XG4gICAgaWYgKHggPT09IC1JbmZpbml0eSkge1xuICAgICAgICBpZiAoYnl0ZXMgPT09IDEpIHJldHVybiB1bmRlZmluZWQ7XG4gICAgICAgIHJldHVybiBzcHJpbnRmKHJmbXQsIGJ5dGVzID49IDkgPyAnLUluZmluaXR5JyA6ICctSW5mJykuc2xpY2UoMCwgYnl0ZXMpO1xuICAgIH1cbiAgICByZXR1cm4gcGFja2YoeCwgYnl0ZXMpO1xufTtcblxuZnVuY3Rpb24gc2NpICh4LCBieXRlcykge1xuICAgIHZhciBuID0gTWF0aC5tYXgoMSwgbG9nMTBmKE1hdGguYWJzKHgpKSk7XG4gICAgdmFyIHN6ID0gbG9nMTBmKE1hdGguYWJzKG4pKTtcbiAgICBcbiAgICB2YXIgYiA9IE1hdGgucG93KDEwLGJ5dGVzKzEpO1xuICAgIGlmIChNYXRoLmFicyh4KSA8IDEpIHtcbiAgICAgICAgeCA9IE1hdGgucm91bmQoeCAqIGIpIC8gYjtcbiAgICB9XG4gICAgZWxzZSB7XG4gICAgICAgIHZhciB0biA9IE1hdGgucG93KDEwLCBuICsgMSk7XG4gICAgICAgIHggPSBNYXRoLnJvdW5kKHggLyB0biAqIGIpIC8gYiAqIHRuO1xuICAgIH1cbiAgICBcbiAgICB2YXIgcztcbiAgICBpZiAoYnl0ZXMgLSBzeiAtIDYgPT09IC0xKSB7XG4gICAgICAgIHggPSBNYXRoLnJvdW5kKHggLyBNYXRoLnBvdygxMCwgbikpO1xuICAgICAgICB4ID0geCAqIE1hdGgucG93KDEwLCBuKTtcbiAgICAgICAgcyA9IHNwcmludGYoJyUxZScsIHgpLnJlcGxhY2UoL1xcLlteZV0rLywgJycpO1xuICAgIH1cbiAgICBlbHNlIGlmIChieXRlcyAtIHN6IC0gNiA8IDApIHJldHVybiB1bmRlZmluZWQ7XG4gICAgZWxzZSB7XG4gICAgICAgIHMgPSBzcHJpbnRmKCclLicgKyAoYnl0ZXMgLSBzeiAtIDYpICsgJ2UnLCB4KTtcbiAgICB9XG4gICAgaWYgKHggPiAwKSBzID0gJyAnICsgcztcbiAgICByZXR1cm4gcGFkKHMsIGJ5dGVzKTtcbn1cblxuZnVuY3Rpb24gcGFkIChzLCBieXRlcykge1xuICAgIHJldHVybiBBcnJheShNYXRoLm1heCgwLCBieXRlcyAtIHMubGVuZ3RoICsgMSkpLmpvaW4oJyAnKSArIHM7XG59XG5cbmZ1bmN0aW9uIGxvZzEwZiAobikge1xuICAgIHJldHVybiBNYXRoLmZsb29yKE1hdGgubG9nKG4pIC8gTWF0aC5sb2coMTApKTtcbn1cblxuZnVuY3Rpb24gcGFja2YgKHgsIGJ5dGVzKSB7XG4gICAgdmFyIGxieXRlcyA9IE1hdGgubWF4KDEsIE1hdGguZmxvb3IoKGJ5dGVzIC0gMikgLyAyKSk7XG4gICAgdmFyIHJieXRlcyA9IGJ5dGVzIC0gbGJ5dGVzIC0gMjtcbiAgICBcbiAgICBpZiAoeCA9PT0gMCAmJiBieXRlcyA8IDQpIHtcbiAgICAgICAgcmV0dXJuIHBhZCgnMCcsIGJ5dGVzKTtcbiAgICB9XG4gICAgZWxzZSBpZiAoeCA9PT0gMCkge1xuICAgICAgICByZXR1cm4gcGFkKCcwLicgKyBBcnJheShyYnl0ZXMrMSkuam9pbignMCcpLCBieXRlcyk7XG4gICAgfVxuICAgIFxuICAgIGlmIChyYnl0ZXMgPD0gMCkge1xuICAgICAgICB2YXIgcyA9IHNwcmludGYoJyUnICsgbGJ5dGVzICsgJ2YnLCB4KTtcbiAgICAgICAgaWYgKHggPj0gMCkgcyA9ICcgJyArIHM7XG4gICAgICAgIGlmIChzLmxlbmd0aCA+IGJ5dGVzKSByZXR1cm4gdW5kZWZpbmVkO1xuICAgICAgICByZXR1cm4gcGFkKHMsIGJ5dGVzKTtcbiAgICB9XG4gICAgaWYgKE1hdGguYWJzKHgpIDwgTWF0aC5wb3coMTAsMS1yYnl0ZXMpKSByZXR1cm4gc2NpKHgsIGJ5dGVzKTtcbiAgICBcbiAgICB2YXIgYiA9IE1hdGgucG93KDEwLGJ5dGVzLTMpO1xuICAgIHZhciB0biA9IE1hdGgucG93KDEwLCBsb2cxMGYoTWF0aC5hYnMoeCkpKTtcbiAgICB2YXIgeHIgPSBNYXRoLnJvdW5kKHggLyB0biAqIGIpIC8gYiAqIHRuO1xuICAgIFxuICAgIHZhciBzID0gc3ByaW50ZignJScgKyBsYnl0ZXMgKyAnLicgKyByYnl0ZXMgKyAnZicsIHhyKTtcbiAgICBpZiAoeHIgPiAwKSBzID0gJyAnICsgcztcbiAgICBzID0gcy5zbGljZSgwLCBieXRlcyk7XG4gICAgdmFyIHIgPSBzLnNwbGl0KCcuJylbMV07XG4gICAgaWYgKCFyIHx8IHIubGVuZ3RoIDwgMSkgcmV0dXJuIHNjaSh4ciwgYnl0ZXMpO1xuICAgIHJldHVybiBwYWQocywgYnl0ZXMpLnNsaWNlKDAsIGJ5dGVzKTtcbn1cbiIsIlwidXNlIHN0cmljdFwiXG5cbmZ1bmN0aW9uIGlvdGEobikge1xuICB2YXIgcmVzdWx0ID0gbmV3IEFycmF5KG4pXG4gIGZvcih2YXIgaT0wOyBpPG47ICsraSkge1xuICAgIHJlc3VsdFtpXSA9IGlcbiAgfVxuICByZXR1cm4gcmVzdWx0XG59XG5cbm1vZHVsZS5leHBvcnRzID0gaW90YSIsIi8qIVxuICogRGV0ZXJtaW5lIGlmIGFuIG9iamVjdCBpcyBhIEJ1ZmZlclxuICpcbiAqIEBhdXRob3IgICBGZXJvc3MgQWJvdWtoYWRpamVoIDxmZXJvc3NAZmVyb3NzLm9yZz4gPGh0dHA6Ly9mZXJvc3Mub3JnPlxuICogQGxpY2Vuc2UgIE1JVFxuICovXG5cbi8vIFRoZSBfaXNCdWZmZXIgY2hlY2sgaXMgZm9yIFNhZmFyaSA1LTcgc3VwcG9ydCwgYmVjYXVzZSBpdCdzIG1pc3Npbmdcbi8vIE9iamVjdC5wcm90b3R5cGUuY29uc3RydWN0b3IuIFJlbW92ZSB0aGlzIGV2ZW50dWFsbHlcbm1vZHVsZS5leHBvcnRzID0gZnVuY3Rpb24gKG9iaikge1xuICByZXR1cm4gb2JqICE9IG51bGwgJiYgKGlzQnVmZmVyKG9iaikgfHwgaXNTbG93QnVmZmVyKG9iaikgfHwgISFvYmouX2lzQnVmZmVyKVxufVxuXG5mdW5jdGlvbiBpc0J1ZmZlciAob2JqKSB7XG4gIHJldHVybiAhIW9iai5jb25zdHJ1Y3RvciAmJiB0eXBlb2Ygb2JqLmNvbnN0cnVjdG9yLmlzQnVmZmVyID09PSAnZnVuY3Rpb24nICYmIG9iai5jb25zdHJ1Y3Rvci5pc0J1ZmZlcihvYmopXG59XG5cbi8vIEZvciBOb2RlIHYwLjEwIHN1cHBvcnQuIFJlbW92ZSB0aGlzIGV2ZW50dWFsbHkuXG5mdW5jdGlvbiBpc1Nsb3dCdWZmZXIgKG9iaikge1xuICByZXR1cm4gdHlwZW9mIG9iai5yZWFkRmxvYXRMRSA9PT0gJ2Z1bmN0aW9uJyAmJiB0eXBlb2Ygb2JqLnNsaWNlID09PSAnZnVuY3Rpb24nICYmIGlzQnVmZmVyKG9iai5zbGljZSgwLCAwKSlcbn1cbiIsInZhciBzaG93ZiA9IHJlcXVpcmUoJ2ZpeGVkLXdpZHRoLWZsb2F0Jyk7XG52YXIgbmRhcnJheSA9IHJlcXVpcmUoJ25kYXJyYXknKTtcblxubW9kdWxlLmV4cG9ydHMgPSBmdW5jdGlvbiAobSwgb3B0cykge1xuICAgIGlmICghb3B0cykgb3B0cyA9IHt9O1xuICAgIGlmICh0eXBlb2Ygb3B0cyA9PT0gJ251bWJlcicpIG9wdHMgPSB7IHdpZHRoOiBvcHRzIH07XG4gICAgaWYgKCFvcHRzLndpZHRoKSBvcHRzLndpZHRoID0gODtcblxuICAgIGlmIChtLmRpbWVuc2lvbiA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICAgIG0gPSBuZGFycmF5KG0pO1xuICAgIH1cblxuICAgIGlmIChtLmRpbWVuc2lvbiA9PT0gMSkgcmV0dXJuIGQxKG0sIG9wdHMpO1xuICAgIGlmIChtLmRpbWVuc2lvbiA9PT0gMikgcmV0dXJuIGQyKG0sIG9wdHMpO1xuICAgIGlmIChtLmRpbWVuc2lvbiA9PT0gMykgcmV0dXJuIGQzKG0sIG9wdHMpO1xuICAgIGlmIChtLmRpbWVuc2lvbiA9PT0gNCkgcmV0dXJuIGQ0KG0sIG9wdHMpO1xufTtcblxuZnVuY3Rpb24gZDEgKG0sIG9wdHMpIHtcbiAgICB2YXIgdGVybXMgPSBbXTtcbiAgICBmb3IgKHZhciBpID0gMDsgaSA8IG0uc2hhcGVbMF07IGkrKykge1xuICAgICAgICB0ZXJtcy5wdXNoKHNob3dmKG0uZ2V0KGkpLCBvcHRzLndpZHRoKSk7XG4gICAgfVxuICAgIHJldHVybiB0ZXJtcy5qb2luKCcgJyk7XG59XG5cbmZ1bmN0aW9uIGQyIChtLCBvcHRzKSB7XG4gICAgdmFyIHJvd3MgPSBbXTtcbiAgICBmb3IgKHZhciB5ID0gMDsgeSA8IG0uc2hhcGVbMF07IHkrKykge1xuICAgICAgICByb3dzLnB1c2goZDEobS5waWNrKHksIG51bGwpLCBvcHRzKSk7XG4gICAgfVxuICAgIHJldHVybiByb3dzLmpvaW4oJ1xcbicpO1xufVxuXG5mdW5jdGlvbiBkMyAobSwgb3B0cykge1xuICAgIHZhciByb3dzID0gW107XG4gICAgZm9yICh2YXIgeiA9IDA7IHogPCBtLnNoYXBlWzBdOyB6KyspIHtcbiAgICAgICAgcm93cy5wdXNoKGQyKG0ucGljayh6LCBudWxsLCBudWxsKSwgb3B0cyksICcnKTtcbiAgICB9XG4gICAgcmV0dXJuIHJvd3Muam9pbignXFxuJyk7XG59XG5cbmZ1bmN0aW9uIGQ0IChtLCBvcHRzKSB7XG4gICAgdmFyIHJvd3MgPSBbXSwgbGVuID0gM1xuICAgIGZvciAodmFyIHcgPSAwOyB3IDwgbS5zaGFwZVswXTsgdysrKSB7XG4gICAgICAgIHZhciByID0gZDMobS5waWNrKHcsIG51bGwsIG51bGwsIG51bGwpLCBvcHRzKVxuICAgICAgICByb3dzLnB1c2gocik7XG4gICAgICAgIHZhciBsaW5lcyA9IHIuc3BsaXQoJ1xcbicpO1xuICAgICAgICBmb3IgKHZhciBpID0gMDsgaSA8IGxpbmVzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgICBsZW4gPSBNYXRoLm1heChsZW4sIGxpbmVzW2ldLmxlbmd0aCk7XG4gICAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIHJvd3Muam9pbignXFxuJyArIEFycmF5KGxlbisxKS5qb2luKCctJykgKyAnXFxuXFxuJyk7XG59XG4iLCJ2YXIgaW90YSA9IHJlcXVpcmUoXCJpb3RhLWFycmF5XCIpXG52YXIgaXNCdWZmZXIgPSByZXF1aXJlKFwiaXMtYnVmZmVyXCIpXG5cbnZhciBoYXNUeXBlZEFycmF5cyAgPSAoKHR5cGVvZiBGbG9hdDY0QXJyYXkpICE9PSBcInVuZGVmaW5lZFwiKVxuXG5mdW5jdGlvbiBjb21wYXJlMXN0KGEsIGIpIHtcbiAgcmV0dXJuIGFbMF0gLSBiWzBdXG59XG5cbmZ1bmN0aW9uIG9yZGVyKCkge1xuICB2YXIgc3RyaWRlID0gdGhpcy5zdHJpZGVcbiAgdmFyIHRlcm1zID0gbmV3IEFycmF5KHN0cmlkZS5sZW5ndGgpXG4gIHZhciBpXG4gIGZvcihpPTA7IGk8dGVybXMubGVuZ3RoOyArK2kpIHtcbiAgICB0ZXJtc1tpXSA9IFtNYXRoLmFicyhzdHJpZGVbaV0pLCBpXVxuICB9XG4gIHRlcm1zLnNvcnQoY29tcGFyZTFzdClcbiAgdmFyIHJlc3VsdCA9IG5ldyBBcnJheSh0ZXJtcy5sZW5ndGgpXG4gIGZvcihpPTA7IGk8cmVzdWx0Lmxlbmd0aDsgKytpKSB7XG4gICAgcmVzdWx0W2ldID0gdGVybXNbaV1bMV1cbiAgfVxuICByZXR1cm4gcmVzdWx0XG59XG5cbmZ1bmN0aW9uIGNvbXBpbGVDb25zdHJ1Y3RvcihkdHlwZSwgZGltZW5zaW9uKSB7XG4gIHZhciBjbGFzc05hbWUgPSBbXCJWaWV3XCIsIGRpbWVuc2lvbiwgXCJkXCIsIGR0eXBlXS5qb2luKFwiXCIpXG4gIGlmKGRpbWVuc2lvbiA8IDApIHtcbiAgICBjbGFzc05hbWUgPSBcIlZpZXdfTmlsXCIgKyBkdHlwZVxuICB9XG4gIHZhciB1c2VHZXR0ZXJzID0gKGR0eXBlID09PSBcImdlbmVyaWNcIilcblxuICBpZihkaW1lbnNpb24gPT09IC0xKSB7XG4gICAgLy9TcGVjaWFsIGNhc2UgZm9yIHRyaXZpYWwgYXJyYXlzXG4gICAgdmFyIGNvZGUgPVxuICAgICAgXCJmdW5jdGlvbiBcIitjbGFzc05hbWUrXCIoYSl7dGhpcy5kYXRhPWE7fTtcXFxudmFyIHByb3RvPVwiK2NsYXNzTmFtZStcIi5wcm90b3R5cGU7XFxcbnByb3RvLmR0eXBlPSdcIitkdHlwZStcIic7XFxcbnByb3RvLmluZGV4PWZ1bmN0aW9uKCl7cmV0dXJuIC0xfTtcXFxucHJvdG8uc2l6ZT0wO1xcXG5wcm90by5kaW1lbnNpb249LTE7XFxcbnByb3RvLnNoYXBlPXByb3RvLnN0cmlkZT1wcm90by5vcmRlcj1bXTtcXFxucHJvdG8ubG89cHJvdG8uaGk9cHJvdG8udHJhbnNwb3NlPXByb3RvLnN0ZXA9XFxcbmZ1bmN0aW9uKCl7cmV0dXJuIG5ldyBcIitjbGFzc05hbWUrXCIodGhpcy5kYXRhKTt9O1xcXG5wcm90by5nZXQ9cHJvdG8uc2V0PWZ1bmN0aW9uKCl7fTtcXFxucHJvdG8ucGljaz1mdW5jdGlvbigpe3JldHVybiBudWxsfTtcXFxucmV0dXJuIGZ1bmN0aW9uIGNvbnN0cnVjdF9cIitjbGFzc05hbWUrXCIoYSl7cmV0dXJuIG5ldyBcIitjbGFzc05hbWUrXCIoYSk7fVwiXG4gICAgdmFyIHByb2NlZHVyZSA9IG5ldyBGdW5jdGlvbihjb2RlKVxuICAgIHJldHVybiBwcm9jZWR1cmUoKVxuICB9IGVsc2UgaWYoZGltZW5zaW9uID09PSAwKSB7XG4gICAgLy9TcGVjaWFsIGNhc2UgZm9yIDBkIGFycmF5c1xuICAgIHZhciBjb2RlID1cbiAgICAgIFwiZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiKGEsZCkge1xcXG50aGlzLmRhdGEgPSBhO1xcXG50aGlzLm9mZnNldCA9IGRcXFxufTtcXFxudmFyIHByb3RvPVwiK2NsYXNzTmFtZStcIi5wcm90b3R5cGU7XFxcbnByb3RvLmR0eXBlPSdcIitkdHlwZStcIic7XFxcbnByb3RvLmluZGV4PWZ1bmN0aW9uKCl7cmV0dXJuIHRoaXMub2Zmc2V0fTtcXFxucHJvdG8uZGltZW5zaW9uPTA7XFxcbnByb3RvLnNpemU9MTtcXFxucHJvdG8uc2hhcGU9XFxcbnByb3RvLnN0cmlkZT1cXFxucHJvdG8ub3JkZXI9W107XFxcbnByb3RvLmxvPVxcXG5wcm90by5oaT1cXFxucHJvdG8udHJhbnNwb3NlPVxcXG5wcm90by5zdGVwPWZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl9jb3B5KCkge1xcXG5yZXR1cm4gbmV3IFwiK2NsYXNzTmFtZStcIih0aGlzLmRhdGEsdGhpcy5vZmZzZXQpXFxcbn07XFxcbnByb3RvLnBpY2s9ZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX3BpY2soKXtcXFxucmV0dXJuIFRyaXZpYWxBcnJheSh0aGlzLmRhdGEpO1xcXG59O1xcXG5wcm90by52YWx1ZU9mPXByb3RvLmdldD1mdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfZ2V0KCl7XFxcbnJldHVybiBcIisodXNlR2V0dGVycyA/IFwidGhpcy5kYXRhLmdldCh0aGlzLm9mZnNldClcIiA6IFwidGhpcy5kYXRhW3RoaXMub2Zmc2V0XVwiKStcblwifTtcXFxucHJvdG8uc2V0PWZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl9zZXQodil7XFxcbnJldHVybiBcIisodXNlR2V0dGVycyA/IFwidGhpcy5kYXRhLnNldCh0aGlzLm9mZnNldCx2KVwiIDogXCJ0aGlzLmRhdGFbdGhpcy5vZmZzZXRdPXZcIikrXCJcXFxufTtcXFxucmV0dXJuIGZ1bmN0aW9uIGNvbnN0cnVjdF9cIitjbGFzc05hbWUrXCIoYSxiLGMsZCl7cmV0dXJuIG5ldyBcIitjbGFzc05hbWUrXCIoYSxkKX1cIlxuICAgIHZhciBwcm9jZWR1cmUgPSBuZXcgRnVuY3Rpb24oXCJUcml2aWFsQXJyYXlcIiwgY29kZSlcbiAgICByZXR1cm4gcHJvY2VkdXJlKENBQ0hFRF9DT05TVFJVQ1RPUlNbZHR5cGVdWzBdKVxuICB9XG5cbiAgdmFyIGNvZGUgPSBbXCIndXNlIHN0cmljdCdcIl1cblxuICAvL0NyZWF0ZSBjb25zdHJ1Y3RvciBmb3Igdmlld1xuICB2YXIgaW5kaWNlcyA9IGlvdGEoZGltZW5zaW9uKVxuICB2YXIgYXJncyA9IGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHsgcmV0dXJuIFwiaVwiK2kgfSlcbiAgdmFyIGluZGV4X3N0ciA9IFwidGhpcy5vZmZzZXQrXCIgKyBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7XG4gICAgICAgIHJldHVybiBcInRoaXMuc3RyaWRlW1wiICsgaSArIFwiXSppXCIgKyBpXG4gICAgICB9KS5qb2luKFwiK1wiKVxuICB2YXIgc2hhcGVBcmcgPSBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7XG4gICAgICByZXR1cm4gXCJiXCIraVxuICAgIH0pLmpvaW4oXCIsXCIpXG4gIHZhciBzdHJpZGVBcmcgPSBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7XG4gICAgICByZXR1cm4gXCJjXCIraVxuICAgIH0pLmpvaW4oXCIsXCIpXG4gIGNvZGUucHVzaChcbiAgICBcImZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIihhLFwiICsgc2hhcGVBcmcgKyBcIixcIiArIHN0cmlkZUFyZyArIFwiLGQpe3RoaXMuZGF0YT1hXCIsXG4gICAgICBcInRoaXMuc2hhcGU9W1wiICsgc2hhcGVBcmcgKyBcIl1cIixcbiAgICAgIFwidGhpcy5zdHJpZGU9W1wiICsgc3RyaWRlQXJnICsgXCJdXCIsXG4gICAgICBcInRoaXMub2Zmc2V0PWR8MH1cIixcbiAgICBcInZhciBwcm90bz1cIitjbGFzc05hbWUrXCIucHJvdG90eXBlXCIsXG4gICAgXCJwcm90by5kdHlwZT0nXCIrZHR5cGUrXCInXCIsXG4gICAgXCJwcm90by5kaW1lbnNpb249XCIrZGltZW5zaW9uKVxuXG4gIC8vdmlldy5zaXplOlxuICBjb2RlLnB1c2goXCJPYmplY3QuZGVmaW5lUHJvcGVydHkocHJvdG8sJ3NpemUnLHtnZXQ6ZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX3NpemUoKXtcXFxucmV0dXJuIFwiK2luZGljZXMubWFwKGZ1bmN0aW9uKGkpIHsgcmV0dXJuIFwidGhpcy5zaGFwZVtcIitpK1wiXVwiIH0pLmpvaW4oXCIqXCIpLFxuXCJ9fSlcIilcblxuICAvL3ZpZXcub3JkZXI6XG4gIGlmKGRpbWVuc2lvbiA9PT0gMSkge1xuICAgIGNvZGUucHVzaChcInByb3RvLm9yZGVyPVswXVwiKVxuICB9IGVsc2Uge1xuICAgIGNvZGUucHVzaChcIk9iamVjdC5kZWZpbmVQcm9wZXJ0eShwcm90bywnb3JkZXInLHtnZXQ6XCIpXG4gICAgaWYoZGltZW5zaW9uIDwgNCkge1xuICAgICAgY29kZS5wdXNoKFwiZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX29yZGVyKCl7XCIpXG4gICAgICBpZihkaW1lbnNpb24gPT09IDIpIHtcbiAgICAgICAgY29kZS5wdXNoKFwicmV0dXJuIChNYXRoLmFicyh0aGlzLnN0cmlkZVswXSk+TWF0aC5hYnModGhpcy5zdHJpZGVbMV0pKT9bMSwwXTpbMCwxXX19KVwiKVxuICAgICAgfSBlbHNlIGlmKGRpbWVuc2lvbiA9PT0gMykge1xuICAgICAgICBjb2RlLnB1c2goXG5cInZhciBzMD1NYXRoLmFicyh0aGlzLnN0cmlkZVswXSksczE9TWF0aC5hYnModGhpcy5zdHJpZGVbMV0pLHMyPU1hdGguYWJzKHRoaXMuc3RyaWRlWzJdKTtcXFxuaWYoczA+czEpe1xcXG5pZihzMT5zMil7XFxcbnJldHVybiBbMiwxLDBdO1xcXG59ZWxzZSBpZihzMD5zMil7XFxcbnJldHVybiBbMSwyLDBdO1xcXG59ZWxzZXtcXFxucmV0dXJuIFsxLDAsMl07XFxcbn1cXFxufWVsc2UgaWYoczA+czIpe1xcXG5yZXR1cm4gWzIsMCwxXTtcXFxufWVsc2UgaWYoczI+czEpe1xcXG5yZXR1cm4gWzAsMSwyXTtcXFxufWVsc2V7XFxcbnJldHVybiBbMCwyLDFdO1xcXG59fX0pXCIpXG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvZGUucHVzaChcIk9SREVSfSlcIilcbiAgICB9XG4gIH1cblxuICAvL3ZpZXcuc2V0KGkwLCAuLi4sIHYpOlxuICBjb2RlLnB1c2goXG5cInByb3RvLnNldD1mdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfc2V0KFwiK2FyZ3Muam9pbihcIixcIikrXCIsdil7XCIpXG4gIGlmKHVzZUdldHRlcnMpIHtcbiAgICBjb2RlLnB1c2goXCJyZXR1cm4gdGhpcy5kYXRhLnNldChcIitpbmRleF9zdHIrXCIsdil9XCIpXG4gIH0gZWxzZSB7XG4gICAgY29kZS5wdXNoKFwicmV0dXJuIHRoaXMuZGF0YVtcIitpbmRleF9zdHIrXCJdPXZ9XCIpXG4gIH1cblxuICAvL3ZpZXcuZ2V0KGkwLCAuLi4pOlxuICBjb2RlLnB1c2goXCJwcm90by5nZXQ9ZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX2dldChcIithcmdzLmpvaW4oXCIsXCIpK1wiKXtcIilcbiAgaWYodXNlR2V0dGVycykge1xuICAgIGNvZGUucHVzaChcInJldHVybiB0aGlzLmRhdGEuZ2V0KFwiK2luZGV4X3N0citcIil9XCIpXG4gIH0gZWxzZSB7XG4gICAgY29kZS5wdXNoKFwicmV0dXJuIHRoaXMuZGF0YVtcIitpbmRleF9zdHIrXCJdfVwiKVxuICB9XG5cbiAgLy92aWV3LmluZGV4OlxuICBjb2RlLnB1c2goXG4gICAgXCJwcm90by5pbmRleD1mdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfaW5kZXgoXCIsIGFyZ3Muam9pbigpLCBcIil7cmV0dXJuIFwiK2luZGV4X3N0citcIn1cIilcblxuICAvL3ZpZXcuaGkoKTpcbiAgY29kZS5wdXNoKFwicHJvdG8uaGk9ZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX2hpKFwiK2FyZ3Muam9pbihcIixcIikrXCIpe3JldHVybiBuZXcgXCIrY2xhc3NOYW1lK1wiKHRoaXMuZGF0YSxcIitcbiAgICBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7XG4gICAgICByZXR1cm4gW1wiKHR5cGVvZiBpXCIsaSxcIiE9PSdudW1iZXInfHxpXCIsaSxcIjwwKT90aGlzLnNoYXBlW1wiLCBpLCBcIl06aVwiLCBpLFwifDBcIl0uam9pbihcIlwiKVxuICAgIH0pLmpvaW4oXCIsXCIpK1wiLFwiK1xuICAgIGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHtcbiAgICAgIHJldHVybiBcInRoaXMuc3RyaWRlW1wiK2kgKyBcIl1cIlxuICAgIH0pLmpvaW4oXCIsXCIpK1wiLHRoaXMub2Zmc2V0KX1cIilcblxuICAvL3ZpZXcubG8oKTpcbiAgdmFyIGFfdmFycyA9IGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHsgcmV0dXJuIFwiYVwiK2krXCI9dGhpcy5zaGFwZVtcIitpK1wiXVwiIH0pXG4gIHZhciBjX3ZhcnMgPSBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7IHJldHVybiBcImNcIitpK1wiPXRoaXMuc3RyaWRlW1wiK2krXCJdXCIgfSlcbiAgY29kZS5wdXNoKFwicHJvdG8ubG89ZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX2xvKFwiK2FyZ3Muam9pbihcIixcIikrXCIpe3ZhciBiPXRoaXMub2Zmc2V0LGQ9MCxcIithX3ZhcnMuam9pbihcIixcIikrXCIsXCIrY192YXJzLmpvaW4oXCIsXCIpKVxuICBmb3IodmFyIGk9MDsgaTxkaW1lbnNpb247ICsraSkge1xuICAgIGNvZGUucHVzaChcblwiaWYodHlwZW9mIGlcIitpK1wiPT09J251bWJlcicmJmlcIitpK1wiPj0wKXtcXFxuZD1pXCIraStcInwwO1xcXG5iKz1jXCIraStcIipkO1xcXG5hXCIraStcIi09ZH1cIilcbiAgfVxuICBjb2RlLnB1c2goXCJyZXR1cm4gbmV3IFwiK2NsYXNzTmFtZStcIih0aGlzLmRhdGEsXCIrXG4gICAgaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkge1xuICAgICAgcmV0dXJuIFwiYVwiK2lcbiAgICB9KS5qb2luKFwiLFwiKStcIixcIitcbiAgICBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7XG4gICAgICByZXR1cm4gXCJjXCIraVxuICAgIH0pLmpvaW4oXCIsXCIpK1wiLGIpfVwiKVxuXG4gIC8vdmlldy5zdGVwKCk6XG4gIGNvZGUucHVzaChcInByb3RvLnN0ZXA9ZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX3N0ZXAoXCIrYXJncy5qb2luKFwiLFwiKStcIil7dmFyIFwiK1xuICAgIGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHtcbiAgICAgIHJldHVybiBcImFcIitpK1wiPXRoaXMuc2hhcGVbXCIraStcIl1cIlxuICAgIH0pLmpvaW4oXCIsXCIpK1wiLFwiK1xuICAgIGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHtcbiAgICAgIHJldHVybiBcImJcIitpK1wiPXRoaXMuc3RyaWRlW1wiK2krXCJdXCJcbiAgICB9KS5qb2luKFwiLFwiKStcIixjPXRoaXMub2Zmc2V0LGQ9MCxjZWlsPU1hdGguY2VpbFwiKVxuICBmb3IodmFyIGk9MDsgaTxkaW1lbnNpb247ICsraSkge1xuICAgIGNvZGUucHVzaChcblwiaWYodHlwZW9mIGlcIitpK1wiPT09J251bWJlcicpe1xcXG5kPWlcIitpK1wifDA7XFxcbmlmKGQ8MCl7XFxcbmMrPWJcIitpK1wiKihhXCIraStcIi0xKTtcXFxuYVwiK2krXCI9Y2VpbCgtYVwiK2krXCIvZClcXFxufWVsc2V7XFxcbmFcIitpK1wiPWNlaWwoYVwiK2krXCIvZClcXFxufVxcXG5iXCIraStcIio9ZFxcXG59XCIpXG4gIH1cbiAgY29kZS5wdXNoKFwicmV0dXJuIG5ldyBcIitjbGFzc05hbWUrXCIodGhpcy5kYXRhLFwiK1xuICAgIGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHtcbiAgICAgIHJldHVybiBcImFcIiArIGlcbiAgICB9KS5qb2luKFwiLFwiKStcIixcIitcbiAgICBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7XG4gICAgICByZXR1cm4gXCJiXCIgKyBpXG4gICAgfSkuam9pbihcIixcIikrXCIsYyl9XCIpXG5cbiAgLy92aWV3LnRyYW5zcG9zZSgpOlxuICB2YXIgdFNoYXBlID0gbmV3IEFycmF5KGRpbWVuc2lvbilcbiAgdmFyIHRTdHJpZGUgPSBuZXcgQXJyYXkoZGltZW5zaW9uKVxuICBmb3IodmFyIGk9MDsgaTxkaW1lbnNpb247ICsraSkge1xuICAgIHRTaGFwZVtpXSA9IFwiYVtpXCIraStcIl1cIlxuICAgIHRTdHJpZGVbaV0gPSBcImJbaVwiK2krXCJdXCJcbiAgfVxuICBjb2RlLnB1c2goXCJwcm90by50cmFuc3Bvc2U9ZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX3RyYW5zcG9zZShcIithcmdzK1wiKXtcIitcbiAgICBhcmdzLm1hcChmdW5jdGlvbihuLGlkeCkgeyByZXR1cm4gbiArIFwiPShcIiArIG4gKyBcIj09PXVuZGVmaW5lZD9cIiArIGlkeCArIFwiOlwiICsgbiArIFwifDApXCJ9KS5qb2luKFwiO1wiKSxcbiAgICBcInZhciBhPXRoaXMuc2hhcGUsYj10aGlzLnN0cmlkZTtyZXR1cm4gbmV3IFwiK2NsYXNzTmFtZStcIih0aGlzLmRhdGEsXCIrdFNoYXBlLmpvaW4oXCIsXCIpK1wiLFwiK3RTdHJpZGUuam9pbihcIixcIikrXCIsdGhpcy5vZmZzZXQpfVwiKVxuXG4gIC8vdmlldy5waWNrKCk6XG4gIGNvZGUucHVzaChcInByb3RvLnBpY2s9ZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX3BpY2soXCIrYXJncytcIil7dmFyIGE9W10sYj1bXSxjPXRoaXMub2Zmc2V0XCIpXG4gIGZvcih2YXIgaT0wOyBpPGRpbWVuc2lvbjsgKytpKSB7XG4gICAgY29kZS5wdXNoKFwiaWYodHlwZW9mIGlcIitpK1wiPT09J251bWJlcicmJmlcIitpK1wiPj0wKXtjPShjK3RoaXMuc3RyaWRlW1wiK2krXCJdKmlcIitpK1wiKXwwfWVsc2V7YS5wdXNoKHRoaXMuc2hhcGVbXCIraStcIl0pO2IucHVzaCh0aGlzLnN0cmlkZVtcIitpK1wiXSl9XCIpXG4gIH1cbiAgY29kZS5wdXNoKFwidmFyIGN0b3I9Q1RPUl9MSVNUW2EubGVuZ3RoKzFdO3JldHVybiBjdG9yKHRoaXMuZGF0YSxhLGIsYyl9XCIpXG5cbiAgLy9BZGQgcmV0dXJuIHN0YXRlbWVudFxuICBjb2RlLnB1c2goXCJyZXR1cm4gZnVuY3Rpb24gY29uc3RydWN0X1wiK2NsYXNzTmFtZStcIihkYXRhLHNoYXBlLHN0cmlkZSxvZmZzZXQpe3JldHVybiBuZXcgXCIrY2xhc3NOYW1lK1wiKGRhdGEsXCIrXG4gICAgaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkge1xuICAgICAgcmV0dXJuIFwic2hhcGVbXCIraStcIl1cIlxuICAgIH0pLmpvaW4oXCIsXCIpK1wiLFwiK1xuICAgIGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHtcbiAgICAgIHJldHVybiBcInN0cmlkZVtcIitpK1wiXVwiXG4gICAgfSkuam9pbihcIixcIikrXCIsb2Zmc2V0KX1cIilcblxuICAvL0NvbXBpbGUgcHJvY2VkdXJlXG4gIHZhciBwcm9jZWR1cmUgPSBuZXcgRnVuY3Rpb24oXCJDVE9SX0xJU1RcIiwgXCJPUkRFUlwiLCBjb2RlLmpvaW4oXCJcXG5cIikpXG4gIHJldHVybiBwcm9jZWR1cmUoQ0FDSEVEX0NPTlNUUlVDVE9SU1tkdHlwZV0sIG9yZGVyKVxufVxuXG5mdW5jdGlvbiBhcnJheURUeXBlKGRhdGEpIHtcbiAgaWYoaXNCdWZmZXIoZGF0YSkpIHtcbiAgICByZXR1cm4gXCJidWZmZXJcIlxuICB9XG4gIGlmKGhhc1R5cGVkQXJyYXlzKSB7XG4gICAgc3dpdGNoKE9iamVjdC5wcm90b3R5cGUudG9TdHJpbmcuY2FsbChkYXRhKSkge1xuICAgICAgY2FzZSBcIltvYmplY3QgRmxvYXQ2NEFycmF5XVwiOlxuICAgICAgICByZXR1cm4gXCJmbG9hdDY0XCJcbiAgICAgIGNhc2UgXCJbb2JqZWN0IEZsb2F0MzJBcnJheV1cIjpcbiAgICAgICAgcmV0dXJuIFwiZmxvYXQzMlwiXG4gICAgICBjYXNlIFwiW29iamVjdCBJbnQ4QXJyYXldXCI6XG4gICAgICAgIHJldHVybiBcImludDhcIlxuICAgICAgY2FzZSBcIltvYmplY3QgSW50MTZBcnJheV1cIjpcbiAgICAgICAgcmV0dXJuIFwiaW50MTZcIlxuICAgICAgY2FzZSBcIltvYmplY3QgSW50MzJBcnJheV1cIjpcbiAgICAgICAgcmV0dXJuIFwiaW50MzJcIlxuICAgICAgY2FzZSBcIltvYmplY3QgVWludDhBcnJheV1cIjpcbiAgICAgICAgcmV0dXJuIFwidWludDhcIlxuICAgICAgY2FzZSBcIltvYmplY3QgVWludDE2QXJyYXldXCI6XG4gICAgICAgIHJldHVybiBcInVpbnQxNlwiXG4gICAgICBjYXNlIFwiW29iamVjdCBVaW50MzJBcnJheV1cIjpcbiAgICAgICAgcmV0dXJuIFwidWludDMyXCJcbiAgICAgIGNhc2UgXCJbb2JqZWN0IFVpbnQ4Q2xhbXBlZEFycmF5XVwiOlxuICAgICAgICByZXR1cm4gXCJ1aW50OF9jbGFtcGVkXCJcbiAgICB9XG4gIH1cbiAgaWYoQXJyYXkuaXNBcnJheShkYXRhKSkge1xuICAgIHJldHVybiBcImFycmF5XCJcbiAgfVxuICByZXR1cm4gXCJnZW5lcmljXCJcbn1cblxudmFyIENBQ0hFRF9DT05TVFJVQ1RPUlMgPSB7XG4gIFwiZmxvYXQzMlwiOltdLFxuICBcImZsb2F0NjRcIjpbXSxcbiAgXCJpbnQ4XCI6W10sXG4gIFwiaW50MTZcIjpbXSxcbiAgXCJpbnQzMlwiOltdLFxuICBcInVpbnQ4XCI6W10sXG4gIFwidWludDE2XCI6W10sXG4gIFwidWludDMyXCI6W10sXG4gIFwiYXJyYXlcIjpbXSxcbiAgXCJ1aW50OF9jbGFtcGVkXCI6W10sXG4gIFwiYnVmZmVyXCI6W10sXG4gIFwiZ2VuZXJpY1wiOltdXG59XG5cbjsoZnVuY3Rpb24oKSB7XG4gIGZvcih2YXIgaWQgaW4gQ0FDSEVEX0NPTlNUUlVDVE9SUykge1xuICAgIENBQ0hFRF9DT05TVFJVQ1RPUlNbaWRdLnB1c2goY29tcGlsZUNvbnN0cnVjdG9yKGlkLCAtMSkpXG4gIH1cbn0pO1xuXG5mdW5jdGlvbiB3cmFwcGVkTkRBcnJheUN0b3IoZGF0YSwgc2hhcGUsIHN0cmlkZSwgb2Zmc2V0KSB7XG4gIGlmKGRhdGEgPT09IHVuZGVmaW5lZCkge1xuICAgIHZhciBjdG9yID0gQ0FDSEVEX0NPTlNUUlVDVE9SUy5hcnJheVswXVxuICAgIHJldHVybiBjdG9yKFtdKVxuICB9IGVsc2UgaWYodHlwZW9mIGRhdGEgPT09IFwibnVtYmVyXCIpIHtcbiAgICBkYXRhID0gW2RhdGFdXG4gIH1cbiAgaWYoc2hhcGUgPT09IHVuZGVmaW5lZCkge1xuICAgIHNoYXBlID0gWyBkYXRhLmxlbmd0aCBdXG4gIH1cbiAgdmFyIGQgPSBzaGFwZS5sZW5ndGhcbiAgaWYoc3RyaWRlID09PSB1bmRlZmluZWQpIHtcbiAgICBzdHJpZGUgPSBuZXcgQXJyYXkoZClcbiAgICBmb3IodmFyIGk9ZC0xLCBzej0xOyBpPj0wOyAtLWkpIHtcbiAgICAgIHN0cmlkZVtpXSA9IHN6XG4gICAgICBzeiAqPSBzaGFwZVtpXVxuICAgIH1cbiAgfVxuICBpZihvZmZzZXQgPT09IHVuZGVmaW5lZCkge1xuICAgIG9mZnNldCA9IDBcbiAgICBmb3IodmFyIGk9MDsgaTxkOyArK2kpIHtcbiAgICAgIGlmKHN0cmlkZVtpXSA8IDApIHtcbiAgICAgICAgb2Zmc2V0IC09IChzaGFwZVtpXS0xKSpzdHJpZGVbaV1cbiAgICAgIH1cbiAgICB9XG4gIH1cbiAgdmFyIGR0eXBlID0gYXJyYXlEVHlwZShkYXRhKVxuICB2YXIgY3Rvcl9saXN0ID0gQ0FDSEVEX0NPTlNUUlVDVE9SU1tkdHlwZV1cbiAgd2hpbGUoY3Rvcl9saXN0Lmxlbmd0aCA8PSBkKzEpIHtcbiAgICBjdG9yX2xpc3QucHVzaChjb21waWxlQ29uc3RydWN0b3IoZHR5cGUsIGN0b3JfbGlzdC5sZW5ndGgtMSkpXG4gIH1cbiAgdmFyIGN0b3IgPSBjdG9yX2xpc3RbZCsxXVxuICByZXR1cm4gY3RvcihkYXRhLCBzaGFwZSwgc3RyaWRlLCBvZmZzZXQpXG59XG5cbm1vZHVsZS5leHBvcnRzID0gd3JhcHBlZE5EQXJyYXlDdG9yXG4iLCIvKipcbnNwcmludGYoKSBmb3IgSmF2YVNjcmlwdCAwLjctYmV0YTFcbmh0dHA6Ly93d3cuZGl2ZWludG9qYXZhc2NyaXB0LmNvbS9wcm9qZWN0cy9qYXZhc2NyaXB0LXNwcmludGZcblxuQ29weXJpZ2h0IChjKSBBbGV4YW5kcnUgTWFyYXN0ZWFudSA8YWxleGFob2xpYyBbYXQpIGdtYWlsIChkb3RdIGNvbT5cbkFsbCByaWdodHMgcmVzZXJ2ZWQuXG5cblJlZGlzdHJpYnV0aW9uIGFuZCB1c2UgaW4gc291cmNlIGFuZCBiaW5hcnkgZm9ybXMsIHdpdGggb3Igd2l0aG91dFxubW9kaWZpY2F0aW9uLCBhcmUgcGVybWl0dGVkIHByb3ZpZGVkIHRoYXQgdGhlIGZvbGxvd2luZyBjb25kaXRpb25zIGFyZSBtZXQ6XG4gICAgKiBSZWRpc3RyaWJ1dGlvbnMgb2Ygc291cmNlIGNvZGUgbXVzdCByZXRhaW4gdGhlIGFib3ZlIGNvcHlyaWdodFxuICAgICAgbm90aWNlLCB0aGlzIGxpc3Qgb2YgY29uZGl0aW9ucyBhbmQgdGhlIGZvbGxvd2luZyBkaXNjbGFpbWVyLlxuICAgICogUmVkaXN0cmlidXRpb25zIGluIGJpbmFyeSBmb3JtIG11c3QgcmVwcm9kdWNlIHRoZSBhYm92ZSBjb3B5cmlnaHRcbiAgICAgIG5vdGljZSwgdGhpcyBsaXN0IG9mIGNvbmRpdGlvbnMgYW5kIHRoZSBmb2xsb3dpbmcgZGlzY2xhaW1lciBpbiB0aGVcbiAgICAgIGRvY3VtZW50YXRpb24gYW5kL29yIG90aGVyIG1hdGVyaWFscyBwcm92aWRlZCB3aXRoIHRoZSBkaXN0cmlidXRpb24uXG4gICAgKiBOZWl0aGVyIHRoZSBuYW1lIG9mIHNwcmludGYoKSBmb3IgSmF2YVNjcmlwdCBub3IgdGhlXG4gICAgICBuYW1lcyBvZiBpdHMgY29udHJpYnV0b3JzIG1heSBiZSB1c2VkIHRvIGVuZG9yc2Ugb3IgcHJvbW90ZSBwcm9kdWN0c1xuICAgICAgZGVyaXZlZCBmcm9tIHRoaXMgc29mdHdhcmUgd2l0aG91dCBzcGVjaWZpYyBwcmlvciB3cml0dGVuIHBlcm1pc3Npb24uXG5cblRISVMgU09GVFdBUkUgSVMgUFJPVklERUQgQlkgVEhFIENPUFlSSUdIVCBIT0xERVJTIEFORCBDT05UUklCVVRPUlMgXCJBUyBJU1wiIEFORFxuQU5ZIEVYUFJFU1MgT1IgSU1QTElFRCBXQVJSQU5USUVTLCBJTkNMVURJTkcsIEJVVCBOT1QgTElNSVRFRCBUTywgVEhFIElNUExJRURcbldBUlJBTlRJRVMgT0YgTUVSQ0hBTlRBQklMSVRZIEFORCBGSVRORVNTIEZPUiBBIFBBUlRJQ1VMQVIgUFVSUE9TRSBBUkVcbkRJU0NMQUlNRUQuIElOIE5PIEVWRU5UIFNIQUxMIEFsZXhhbmRydSBNYXJhc3RlYW51IEJFIExJQUJMRSBGT1IgQU5ZXG5ESVJFQ1QsIElORElSRUNULCBJTkNJREVOVEFMLCBTUEVDSUFMLCBFWEVNUExBUlksIE9SIENPTlNFUVVFTlRJQUwgREFNQUdFU1xuKElOQ0xVRElORywgQlVUIE5PVCBMSU1JVEVEIFRPLCBQUk9DVVJFTUVOVCBPRiBTVUJTVElUVVRFIEdPT0RTIE9SIFNFUlZJQ0VTO1xuTE9TUyBPRiBVU0UsIERBVEEsIE9SIFBST0ZJVFM7IE9SIEJVU0lORVNTIElOVEVSUlVQVElPTikgSE9XRVZFUiBDQVVTRUQgQU5EXG5PTiBBTlkgVEhFT1JZIE9GIExJQUJJTElUWSwgV0hFVEhFUiBJTiBDT05UUkFDVCwgU1RSSUNUIExJQUJJTElUWSwgT1IgVE9SVFxuKElOQ0xVRElORyBORUdMSUdFTkNFIE9SIE9USEVSV0lTRSkgQVJJU0lORyBJTiBBTlkgV0FZIE9VVCBPRiBUSEUgVVNFIE9GIFRISVNcblNPRlRXQVJFLCBFVkVOIElGIEFEVklTRUQgT0YgVEhFIFBPU1NJQklMSVRZIE9GIFNVQ0ggREFNQUdFLlxuXG5cbkNoYW5nZWxvZzpcbjIwMTAuMTEuMDcgLSAwLjctYmV0YTEtbm9kZVxuICAtIGNvbnZlcnRlZCBpdCB0byBhIG5vZGUuanMgY29tcGF0aWJsZSBtb2R1bGVcblxuMjAxMC4wOS4wNiAtIDAuNy1iZXRhMVxuICAtIGZlYXR1cmVzOiB2c3ByaW50Ziwgc3VwcG9ydCBmb3IgbmFtZWQgcGxhY2Vob2xkZXJzXG4gIC0gZW5oYW5jZW1lbnRzOiBmb3JtYXQgY2FjaGUsIHJlZHVjZWQgZ2xvYmFsIG5hbWVzcGFjZSBwb2xsdXRpb25cblxuMjAxMC4wNS4yMiAtIDAuNjpcbiAtIHJldmVydGVkIHRvIDAuNCBhbmQgZml4ZWQgdGhlIGJ1ZyByZWdhcmRpbmcgdGhlIHNpZ24gb2YgdGhlIG51bWJlciAwXG4gTm90ZTpcbiBUaGFua3MgdG8gUmFwaGFlbCBQaWd1bGxhIDxyYXBoIChhdF0gbjNyZCBbZG90KSBvcmc+IChodHRwOi8vd3d3Lm4zcmQub3JnLylcbiB3aG8gd2FybmVkIG1lIGFib3V0IGEgYnVnIGluIDAuNSwgSSBkaXNjb3ZlcmVkIHRoYXQgdGhlIGxhc3QgdXBkYXRlIHdhc1xuIGEgcmVncmVzcy4gSSBhcHBvbG9naXplIGZvciB0aGF0LlxuXG4yMDEwLjA1LjA5IC0gMC41OlxuIC0gYnVnIGZpeDogMCBpcyBub3cgcHJlY2VlZGVkIHdpdGggYSArIHNpZ25cbiAtIGJ1ZyBmaXg6IHRoZSBzaWduIHdhcyBub3QgYXQgdGhlIHJpZ2h0IHBvc2l0aW9uIG9uIHBhZGRlZCByZXN1bHRzIChLYW1hbCBBYmRhbGkpXG4gLSBzd2l0Y2hlZCBmcm9tIEdQTCB0byBCU0QgbGljZW5zZVxuXG4yMDA3LjEwLjIxIC0gMC40OlxuIC0gdW5pdCB0ZXN0IGFuZCBwYXRjaCAoRGF2aWQgQmFpcmQpXG5cbjIwMDcuMDkuMTcgLSAwLjM6XG4gLSBidWcgZml4OiBubyBsb25nZXIgdGhyb3dzIGV4Y2VwdGlvbiBvbiBlbXB0eSBwYXJhbWVudGVycyAoSGFucyBQdWZhbClcblxuMjAwNy4wOS4xMSAtIDAuMjpcbiAtIGZlYXR1cmU6IGFkZGVkIGFyZ3VtZW50IHN3YXBwaW5nXG5cbjIwMDcuMDQuMDMgLSAwLjE6XG4gLSBpbml0aWFsIHJlbGVhc2VcbioqL1xuXG52YXIgc3ByaW50ZiA9IChmdW5jdGlvbigpIHtcblx0ZnVuY3Rpb24gZ2V0X3R5cGUodmFyaWFibGUpIHtcblx0XHRyZXR1cm4gT2JqZWN0LnByb3RvdHlwZS50b1N0cmluZy5jYWxsKHZhcmlhYmxlKS5zbGljZSg4LCAtMSkudG9Mb3dlckNhc2UoKTtcblx0fVxuXHRmdW5jdGlvbiBzdHJfcmVwZWF0KGlucHV0LCBtdWx0aXBsaWVyKSB7XG5cdFx0Zm9yICh2YXIgb3V0cHV0ID0gW107IG11bHRpcGxpZXIgPiAwOyBvdXRwdXRbLS1tdWx0aXBsaWVyXSA9IGlucHV0KSB7LyogZG8gbm90aGluZyAqL31cblx0XHRyZXR1cm4gb3V0cHV0LmpvaW4oJycpO1xuXHR9XG5cblx0dmFyIHN0cl9mb3JtYXQgPSBmdW5jdGlvbigpIHtcblx0XHRpZiAoIXN0cl9mb3JtYXQuY2FjaGUuaGFzT3duUHJvcGVydHkoYXJndW1lbnRzWzBdKSkge1xuXHRcdFx0c3RyX2Zvcm1hdC5jYWNoZVthcmd1bWVudHNbMF1dID0gc3RyX2Zvcm1hdC5wYXJzZShhcmd1bWVudHNbMF0pO1xuXHRcdH1cblx0XHRyZXR1cm4gc3RyX2Zvcm1hdC5mb3JtYXQuY2FsbChudWxsLCBzdHJfZm9ybWF0LmNhY2hlW2FyZ3VtZW50c1swXV0sIGFyZ3VtZW50cyk7XG5cdH07XG5cblx0Ly8gY29udmVydCBvYmplY3QgdG8gc2ltcGxlIG9uZSBsaW5lIHN0cmluZyB3aXRob3V0IGluZGVudGF0aW9uIG9yXG5cdC8vIG5ld2xpbmVzLiBOb3RlIHRoYXQgdGhpcyBpbXBsZW1lbnRhdGlvbiBkb2VzIG5vdCBwcmludCBhcnJheVxuXHQvLyB2YWx1ZXMgdG8gdGhlaXIgYWN0dWFsIHBsYWNlIGZvciBzcGFyc2UgYXJyYXlzLiBcblx0Ly9cblx0Ly8gRm9yIGV4YW1wbGUgc3BhcnNlIGFycmF5IGxpa2UgdGhpc1xuXHQvLyAgICBsID0gW11cblx0Ly8gICAgbFs0XSA9IDFcblx0Ly8gV291bGQgYmUgcHJpbnRlZCBhcyBcIlsxXVwiIGluc3RlYWQgb2YgXCJbLCAsICwgLCAxXVwiXG5cdC8vIFxuXHQvLyBJZiBhcmd1bWVudCAnc2VlbicgaXMgbm90IG51bGwgYW5kIGFycmF5IHRoZSBmdW5jdGlvbiB3aWxsIGNoZWNrIGZvciBcblx0Ly8gY2lyY3VsYXIgb2JqZWN0IHJlZmVyZW5jZXMgZnJvbSBhcmd1bWVudC5cblx0c3RyX2Zvcm1hdC5vYmplY3Rfc3RyaW5naWZ5ID0gZnVuY3Rpb24ob2JqLCBkZXB0aCwgbWF4ZGVwdGgsIHNlZW4pIHtcblx0XHR2YXIgc3RyID0gJyc7XG5cdFx0aWYgKG9iaiAhPSBudWxsKSB7XG5cdFx0XHRzd2l0Y2goIHR5cGVvZihvYmopICkge1xuXHRcdFx0Y2FzZSAnZnVuY3Rpb24nOiBcblx0XHRcdFx0cmV0dXJuICdbRnVuY3Rpb24nICsgKG9iai5uYW1lID8gJzogJytvYmoubmFtZSA6ICcnKSArICddJztcblx0XHRcdCAgICBicmVhaztcblx0XHRcdGNhc2UgJ29iamVjdCc6XG5cdFx0XHRcdGlmICggb2JqIGluc3RhbmNlb2YgRXJyb3IpIHsgcmV0dXJuICdbJyArIG9iai50b1N0cmluZygpICsgJ10nIH07XG5cdFx0XHRcdGlmIChkZXB0aCA+PSBtYXhkZXB0aCkgcmV0dXJuICdbT2JqZWN0XSdcblx0XHRcdFx0aWYgKHNlZW4pIHtcblx0XHRcdFx0XHQvLyBhZGQgb2JqZWN0IHRvIHNlZW4gbGlzdFxuXHRcdFx0XHRcdHNlZW4gPSBzZWVuLnNsaWNlKDApXG5cdFx0XHRcdFx0c2Vlbi5wdXNoKG9iaik7XG5cdFx0XHRcdH1cblx0XHRcdFx0aWYgKG9iai5sZW5ndGggIT0gbnVsbCkgeyAvL2FycmF5XG5cdFx0XHRcdFx0c3RyICs9ICdbJztcblx0XHRcdFx0XHR2YXIgYXJyID0gW11cblx0XHRcdFx0XHRmb3IgKHZhciBpIGluIG9iaikge1xuXHRcdFx0XHRcdFx0aWYgKHNlZW4gJiYgc2Vlbi5pbmRleE9mKG9ialtpXSkgPj0gMCkgYXJyLnB1c2goJ1tDaXJjdWxhcl0nKTtcblx0XHRcdFx0XHRcdGVsc2UgYXJyLnB1c2goc3RyX2Zvcm1hdC5vYmplY3Rfc3RyaW5naWZ5KG9ialtpXSwgZGVwdGgrMSwgbWF4ZGVwdGgsIHNlZW4pKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0c3RyICs9IGFyci5qb2luKCcsICcpICsgJ10nO1xuXHRcdFx0XHR9IGVsc2UgaWYgKCdnZXRNb250aCcgaW4gb2JqKSB7IC8vIGRhdGVcblx0XHRcdFx0XHRyZXR1cm4gJ0RhdGUoJyArIG9iaiArICcpJztcblx0XHRcdFx0fSBlbHNlIHsgLy8gb2JqZWN0XG5cdFx0XHRcdFx0c3RyICs9ICd7Jztcblx0XHRcdFx0XHR2YXIgYXJyID0gW11cblx0XHRcdFx0XHRmb3IgKHZhciBrIGluIG9iaikgeyBcblx0XHRcdFx0XHRcdGlmKG9iai5oYXNPd25Qcm9wZXJ0eShrKSkge1xuXHRcdFx0XHRcdFx0XHRpZiAoc2VlbiAmJiBzZWVuLmluZGV4T2Yob2JqW2tdKSA+PSAwKSBhcnIucHVzaChrICsgJzogW0NpcmN1bGFyXScpO1xuXHRcdFx0XHRcdFx0XHRlbHNlIGFyci5wdXNoKGsgKyc6ICcgK3N0cl9mb3JtYXQub2JqZWN0X3N0cmluZ2lmeShvYmpba10sIGRlcHRoKzEsIG1heGRlcHRoLCBzZWVuKSk7IFxuXHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdH1cblx0XHRcdFx0XHRzdHIgKz0gYXJyLmpvaW4oJywgJykgKyAnfSc7XG5cdFx0XHRcdH1cblx0XHRcdFx0cmV0dXJuIHN0cjtcblx0XHRcdFx0YnJlYWs7XG5cdFx0XHRjYXNlICdzdHJpbmcnOlx0XHRcdFx0XG5cdFx0XHRcdHJldHVybiAnXCInICsgb2JqICsgJ1wiJztcblx0XHRcdFx0YnJlYWtcblx0XHRcdH1cblx0XHR9XG5cdFx0cmV0dXJuICcnICsgb2JqO1xuXHR9XG5cblx0c3RyX2Zvcm1hdC5mb3JtYXQgPSBmdW5jdGlvbihwYXJzZV90cmVlLCBhcmd2KSB7XG5cdFx0dmFyIGN1cnNvciA9IDEsIHRyZWVfbGVuZ3RoID0gcGFyc2VfdHJlZS5sZW5ndGgsIG5vZGVfdHlwZSA9ICcnLCBhcmcsIG91dHB1dCA9IFtdLCBpLCBrLCBtYXRjaCwgcGFkLCBwYWRfY2hhcmFjdGVyLCBwYWRfbGVuZ3RoO1xuXHRcdGZvciAoaSA9IDA7IGkgPCB0cmVlX2xlbmd0aDsgaSsrKSB7XG5cdFx0XHRub2RlX3R5cGUgPSBnZXRfdHlwZShwYXJzZV90cmVlW2ldKTtcblx0XHRcdGlmIChub2RlX3R5cGUgPT09ICdzdHJpbmcnKSB7XG5cdFx0XHRcdG91dHB1dC5wdXNoKHBhcnNlX3RyZWVbaV0pO1xuXHRcdFx0fVxuXHRcdFx0ZWxzZSBpZiAobm9kZV90eXBlID09PSAnYXJyYXknKSB7XG5cdFx0XHRcdG1hdGNoID0gcGFyc2VfdHJlZVtpXTsgLy8gY29udmVuaWVuY2UgcHVycG9zZXMgb25seVxuXHRcdFx0XHRpZiAobWF0Y2hbMl0pIHsgLy8ga2V5d29yZCBhcmd1bWVudFxuXHRcdFx0XHRcdGFyZyA9IGFyZ3ZbY3Vyc29yXTtcblx0XHRcdFx0XHRmb3IgKGsgPSAwOyBrIDwgbWF0Y2hbMl0ubGVuZ3RoOyBrKyspIHtcblx0XHRcdFx0XHRcdGlmICghYXJnLmhhc093blByb3BlcnR5KG1hdGNoWzJdW2tdKSkge1xuXHRcdFx0XHRcdFx0XHR0aHJvdyBuZXcgRXJyb3Ioc3ByaW50ZignW3NwcmludGZdIHByb3BlcnR5IFwiJXNcIiBkb2VzIG5vdCBleGlzdCcsIG1hdGNoWzJdW2tdKSk7XG5cdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0XHRhcmcgPSBhcmdbbWF0Y2hbMl1ba11dO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0fVxuXHRcdFx0XHRlbHNlIGlmIChtYXRjaFsxXSkgeyAvLyBwb3NpdGlvbmFsIGFyZ3VtZW50IChleHBsaWNpdClcblx0XHRcdFx0XHRhcmcgPSBhcmd2W21hdGNoWzFdXTtcblx0XHRcdFx0fVxuXHRcdFx0XHRlbHNlIHsgLy8gcG9zaXRpb25hbCBhcmd1bWVudCAoaW1wbGljaXQpXG5cdFx0XHRcdFx0YXJnID0gYXJndltjdXJzb3IrK107XG5cdFx0XHRcdH1cblxuXHRcdFx0XHRpZiAoL1tec09dLy50ZXN0KG1hdGNoWzhdKSAmJiAoZ2V0X3R5cGUoYXJnKSAhPSAnbnVtYmVyJykpIHtcblx0XHRcdFx0XHR0aHJvdyBuZXcgRXJyb3Ioc3ByaW50ZignW3NwcmludGZdIGV4cGVjdGluZyBudW1iZXIgYnV0IGZvdW5kICVzIFwiJyArIGFyZyArICdcIicsIGdldF90eXBlKGFyZykpKTtcblx0XHRcdFx0fVxuXHRcdFx0XHRzd2l0Y2ggKG1hdGNoWzhdKSB7XG5cdFx0XHRcdFx0Y2FzZSAnYic6IGFyZyA9IGFyZy50b1N0cmluZygyKTsgYnJlYWs7XG5cdFx0XHRcdFx0Y2FzZSAnYyc6IGFyZyA9IFN0cmluZy5mcm9tQ2hhckNvZGUoYXJnKTsgYnJlYWs7XG5cdFx0XHRcdFx0Y2FzZSAnZCc6IGFyZyA9IHBhcnNlSW50KGFyZywgMTApOyBicmVhaztcblx0XHRcdFx0XHRjYXNlICdlJzogYXJnID0gbWF0Y2hbN10gPyBhcmcudG9FeHBvbmVudGlhbChtYXRjaFs3XSkgOiBhcmcudG9FeHBvbmVudGlhbCgpOyBicmVhaztcblx0XHRcdFx0XHRjYXNlICdmJzogYXJnID0gbWF0Y2hbN10gPyBwYXJzZUZsb2F0KGFyZykudG9GaXhlZChtYXRjaFs3XSkgOiBwYXJzZUZsb2F0KGFyZyk7IGJyZWFrO1xuXHRcdFx0XHQgICAgY2FzZSAnTyc6IGFyZyA9IHN0cl9mb3JtYXQub2JqZWN0X3N0cmluZ2lmeShhcmcsIDAsIHBhcnNlSW50KG1hdGNoWzddKSB8fCA1KTsgYnJlYWs7XG5cdFx0XHRcdFx0Y2FzZSAnbyc6IGFyZyA9IGFyZy50b1N0cmluZyg4KTsgYnJlYWs7XG5cdFx0XHRcdFx0Y2FzZSAncyc6IGFyZyA9ICgoYXJnID0gU3RyaW5nKGFyZykpICYmIG1hdGNoWzddID8gYXJnLnN1YnN0cmluZygwLCBtYXRjaFs3XSkgOiBhcmcpOyBicmVhaztcblx0XHRcdFx0XHRjYXNlICd1JzogYXJnID0gTWF0aC5hYnMoYXJnKTsgYnJlYWs7XG5cdFx0XHRcdFx0Y2FzZSAneCc6IGFyZyA9IGFyZy50b1N0cmluZygxNik7IGJyZWFrO1xuXHRcdFx0XHRcdGNhc2UgJ1gnOiBhcmcgPSBhcmcudG9TdHJpbmcoMTYpLnRvVXBwZXJDYXNlKCk7IGJyZWFrO1xuXHRcdFx0XHR9XG5cdFx0XHRcdGFyZyA9ICgvW2RlZl0vLnRlc3QobWF0Y2hbOF0pICYmIG1hdGNoWzNdICYmIGFyZyA+PSAwID8gJysnKyBhcmcgOiBhcmcpO1xuXHRcdFx0XHRwYWRfY2hhcmFjdGVyID0gbWF0Y2hbNF0gPyBtYXRjaFs0XSA9PSAnMCcgPyAnMCcgOiBtYXRjaFs0XS5jaGFyQXQoMSkgOiAnICc7XG5cdFx0XHRcdHBhZF9sZW5ndGggPSBtYXRjaFs2XSAtIFN0cmluZyhhcmcpLmxlbmd0aDtcblx0XHRcdFx0cGFkID0gbWF0Y2hbNl0gPyBzdHJfcmVwZWF0KHBhZF9jaGFyYWN0ZXIsIHBhZF9sZW5ndGgpIDogJyc7XG5cdFx0XHRcdG91dHB1dC5wdXNoKG1hdGNoWzVdID8gYXJnICsgcGFkIDogcGFkICsgYXJnKTtcblx0XHRcdH1cblx0XHR9XG5cdFx0cmV0dXJuIG91dHB1dC5qb2luKCcnKTtcblx0fTtcblxuXHRzdHJfZm9ybWF0LmNhY2hlID0ge307XG5cblx0c3RyX2Zvcm1hdC5wYXJzZSA9IGZ1bmN0aW9uKGZtdCkge1xuXHRcdHZhciBfZm10ID0gZm10LCBtYXRjaCA9IFtdLCBwYXJzZV90cmVlID0gW10sIGFyZ19uYW1lcyA9IDA7XG5cdFx0d2hpbGUgKF9mbXQpIHtcblx0XHRcdGlmICgobWF0Y2ggPSAvXlteXFx4MjVdKy8uZXhlYyhfZm10KSkgIT09IG51bGwpIHtcblx0XHRcdFx0cGFyc2VfdHJlZS5wdXNoKG1hdGNoWzBdKTtcblx0XHRcdH1cblx0XHRcdGVsc2UgaWYgKChtYXRjaCA9IC9eXFx4MjV7Mn0vLmV4ZWMoX2ZtdCkpICE9PSBudWxsKSB7XG5cdFx0XHRcdHBhcnNlX3RyZWUucHVzaCgnJScpO1xuXHRcdFx0fVxuXHRcdFx0ZWxzZSBpZiAoKG1hdGNoID0gL15cXHgyNSg/OihbMS05XVxcZCopXFwkfFxcKChbXlxcKV0rKVxcKSk/KFxcKyk/KDB8J1teJF0pPygtKT8oXFxkKyk/KD86XFwuKFxcZCspKT8oW2ItZm9zT3V4WF0pLy5leGVjKF9mbXQpKSAhPT0gbnVsbCkge1xuXHRcdFx0XHRpZiAobWF0Y2hbMl0pIHtcblx0XHRcdFx0XHRhcmdfbmFtZXMgfD0gMTtcblx0XHRcdFx0XHR2YXIgZmllbGRfbGlzdCA9IFtdLCByZXBsYWNlbWVudF9maWVsZCA9IG1hdGNoWzJdLCBmaWVsZF9tYXRjaCA9IFtdO1xuXHRcdFx0XHRcdGlmICgoZmllbGRfbWF0Y2ggPSAvXihbYS16X11bYS16X1xcZF0qKS9pLmV4ZWMocmVwbGFjZW1lbnRfZmllbGQpKSAhPT0gbnVsbCkge1xuXHRcdFx0XHRcdFx0ZmllbGRfbGlzdC5wdXNoKGZpZWxkX21hdGNoWzFdKTtcblx0XHRcdFx0XHRcdHdoaWxlICgocmVwbGFjZW1lbnRfZmllbGQgPSByZXBsYWNlbWVudF9maWVsZC5zdWJzdHJpbmcoZmllbGRfbWF0Y2hbMF0ubGVuZ3RoKSkgIT09ICcnKSB7XG5cdFx0XHRcdFx0XHRcdGlmICgoZmllbGRfbWF0Y2ggPSAvXlxcLihbYS16X11bYS16X1xcZF0qKS9pLmV4ZWMocmVwbGFjZW1lbnRfZmllbGQpKSAhPT0gbnVsbCkge1xuXHRcdFx0XHRcdFx0XHRcdGZpZWxkX2xpc3QucHVzaChmaWVsZF9tYXRjaFsxXSk7XG5cdFx0XHRcdFx0XHRcdH1cblx0XHRcdFx0XHRcdFx0ZWxzZSBpZiAoKGZpZWxkX21hdGNoID0gL15cXFsoXFxkKylcXF0vLmV4ZWMocmVwbGFjZW1lbnRfZmllbGQpKSAhPT0gbnVsbCkge1xuXHRcdFx0XHRcdFx0XHRcdGZpZWxkX2xpc3QucHVzaChmaWVsZF9tYXRjaFsxXSk7XG5cdFx0XHRcdFx0XHRcdH1cblx0XHRcdFx0XHRcdFx0ZWxzZSB7XG5cdFx0XHRcdFx0XHRcdFx0dGhyb3cgbmV3IEVycm9yKCdbc3ByaW50Zl0gJyArIHJlcGxhY2VtZW50X2ZpZWxkKTtcblx0XHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdH1cblx0XHRcdFx0XHRlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcignW3NwcmludGZdICcgKyByZXBsYWNlbWVudF9maWVsZCk7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHRcdG1hdGNoWzJdID0gZmllbGRfbGlzdDtcblx0XHRcdFx0fVxuXHRcdFx0XHRlbHNlIHtcblx0XHRcdFx0XHRhcmdfbmFtZXMgfD0gMjtcblx0XHRcdFx0fVxuXHRcdFx0XHRpZiAoYXJnX25hbWVzID09PSAzKSB7XG5cdFx0XHRcdFx0dGhyb3cgbmV3IEVycm9yKCdbc3ByaW50Zl0gbWl4aW5nIHBvc2l0aW9uYWwgYW5kIG5hbWVkIHBsYWNlaG9sZGVycyBpcyBub3QgKHlldCkgc3VwcG9ydGVkJyk7XG5cdFx0XHRcdH1cblx0XHRcdFx0cGFyc2VfdHJlZS5wdXNoKG1hdGNoKTtcblx0XHRcdH1cblx0XHRcdGVsc2Uge1xuXHRcdFx0XHR0aHJvdyBuZXcgRXJyb3IoJ1tzcHJpbnRmXSAnICsgX2ZtdCk7XG5cdFx0XHR9XG5cdFx0XHRfZm10ID0gX2ZtdC5zdWJzdHJpbmcobWF0Y2hbMF0ubGVuZ3RoKTtcblx0XHR9XG5cdFx0cmV0dXJuIHBhcnNlX3RyZWU7XG5cdH07XG5cblx0cmV0dXJuIHN0cl9mb3JtYXQ7XG59KSgpO1xuXG52YXIgdnNwcmludGYgPSBmdW5jdGlvbihmbXQsIGFyZ3YpIHtcblx0dmFyIGFyZ3ZDbG9uZSA9IGFyZ3Yuc2xpY2UoKTtcblx0YXJndkNsb25lLnVuc2hpZnQoZm10KTtcblx0cmV0dXJuIHNwcmludGYuYXBwbHkobnVsbCwgYXJndkNsb25lKTtcbn07XG5cbm1vZHVsZS5leHBvcnRzID0gc3ByaW50ZjtcbnNwcmludGYuc3ByaW50ZiA9IHNwcmludGY7XG5zcHJpbnRmLnZzcHJpbnRmID0gdnNwcmludGY7XG4iLCJpbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5cbmV4cG9ydCBkZWZhdWx0IHtcblx0aGFyZF9zaWdtb2lkOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9oYXJkX3NpZ21vaWQuZ2xzbCcsICd1dGY4JyksXG5cdGxpbmVhcjogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvbGluZWFyLmdsc2wnLCAndXRmOCcpLFxuXHRyZWx1OiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9yZWx1Lmdsc2wnLCAndXRmOCcpLFxuXHRyZ2I6IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3JnYi5nbHNsJywgJ3V0ZjgnKSxcblx0c2lnbW9pZDogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvc2lnbW9pZC5nbHNsJywgJ3V0ZjgnKSxcblx0dGFuaDogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvdGFuaC5nbHNsJywgJ3V0ZjgnKSxcbn0iLCJpbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5cbmV4cG9ydCBjb25zdCBlbmNvZGVTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9lbmNvZGUuZ2xzbCcsICd1dGY4Jyk7XG5leHBvcnQgY29uc3QgZGVjb2RlU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvZGVjb2RlLmdsc2wnLCAndXRmOCcpO1xuXG5leHBvcnQgZnVuY3Rpb24gaW5pdChzaGFwZSwgZm9ybWF0KXtcblx0cmV0dXJuIHtcblx0XHRyYW5nZTogZm9ybWF0LnJhbmdlIHx8IDQwOTZcblx0fVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZW5jb2RlKGJ1ZiwgdmFsdWUsIGluZm8pe1xuXHR2YXIgeiA9IE1hdGgubWluKDEsIE1hdGgubWF4KDAsIHZhbHVlIC8gaW5mby5yYW5nZSArIDAuNSkpO1xuXHRidWZbMF0gPSAoeiAqIDI1NiAqIDI1NiAqIDI1NiAqIDI1NikgJSAyNTZcblx0YnVmWzFdID0gKHogKiAyNTYgKiAyNTYgKiAyNTYpICUgMjU2XG5cdGJ1ZlsyXSA9ICh6ICogMjU2ICogMjU2KSAlIDI1NlxuXHRidWZbM10gPSAoeiAqIDI1NikgJSAyNTZcbn1cblxuXG5leHBvcnQgZnVuY3Rpb24gZGVjb2RlKGJ1Zil7XG5cdHJldHVybiBidWZbMF0gLyAyNTYuMCAvIDI1Ni4wIC8gMjU2LjAgLyAyNTYuMCArXG5cdFx0ICAgYnVmWzFdIC8gMjU2LjAgLyAyNTYuMCAvIDI1Ni4wICtcblx0XHQgICBidWZbMl0gLyAyNTYuMCAvIDI1Ni4wICtcblx0XHQgICBidWZbM10gLyAyNTYuMDtcbn1cbiIsImltcG9ydCB7IHJlYWRGaWxlU3luYyB9IGZyb20gJ2ZzJztcblxuZXhwb3J0IGNvbnN0IGVuY29kZVNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL2VuY29kZS5nbHNsJywgJ3V0ZjgnKTtcbmV4cG9ydCBjb25zdCBkZWNvZGVTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9kZWNvZGUuZ2xzbCcsICd1dGY4Jyk7XG5cbmV4cG9ydCBmdW5jdGlvbiBpbml0KHNoYXBlLCBmb3JtYXQpe1xuXHRyZXR1cm4geyB9XG59XG5cbnZhciB0bXBfZmxvYXQgPSBuZXcgRmxvYXQzMkFycmF5KDEpLFxuXHR0bXBfaW50ID0gbmV3IFVpbnQ4QXJyYXkodG1wX2Zsb2F0LmJ1ZmZlcik7XG5cbmV4cG9ydCBmdW5jdGlvbiBlbmNvZGUoYnVmLCB2YWx1ZSl7XG5cdHRtcF9mbG9hdFswXSA9IHZhbHVlO1xuXHRidWYuc2V0KHRtcF9pbnQsIDApXG59XG5cbmV4cG9ydCBmdW5jdGlvbiBkZWNvZGUoYnVmKXtcblx0dG1wX2ludC5zZXQoYnVmKVxuXHRyZXR1cm4gdG1wX2Zsb2F0WzBdXG59IiwiaW1wb3J0ICogYXMgcGFja19zdHJpZGUgZnJvbSAnLi9wYWNrL3N0cmlkZS9pbmRleC5qcydcbmltcG9ydCAqIGFzIHBhY2tfdGlsZSBmcm9tICcuL3BhY2svdGlsZS9pbmRleC5qcydcblxuaW1wb3J0ICogYXMgY29kZWNfZml4bnVtIGZyb20gJy4vY29kZWMvZml4bnVtL2luZGV4LmpzJ1xuaW1wb3J0ICogYXMgY29kZWNfc29mdGZsb2F0IGZyb20gJy4vY29kZWMvc29mdGZsb2F0L2luZGV4LmpzJ1xuXG5pbXBvcnQgYWN0aXZhdGlvbnMgZnJvbSAnLi9hY3RpdmF0aW9uL2luZGV4LmpzJ1xuXG5pbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5cbmV4cG9ydCBkZWZhdWx0IHtcblx0cGFjazoge1xuXHRcdHN0cmlkZTogcGFja19zdHJpZGUsXG5cdFx0dGlsZTogcGFja190aWxlXG5cdH0sXG5cblx0cmVhZF9zaGltOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9wYWNrL3JlYWRfc2hpbS5nbHNsJywgJ3V0ZjgnKSxcblx0d3JpdGVfc2hpbTogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvcGFjay93cml0ZV9zaGltLmdsc2wnLCAndXRmOCcpLFxuXG5cdGNvZGVjOiB7XG5cdFx0Zml4bnVtOiBjb2RlY19maXhudW0sXG5cdFx0c29mdGZsb2F0OiBjb2RlY19zb2Z0ZmxvYXQsXG5cdH0sXG5cdGFjdGl2YXRpb25zOiBhY3RpdmF0aW9uc1xufSIsImltcG9ydCB7IHJlYWRGaWxlU3luYyB9IGZyb20gJ2ZzJztcbmltcG9ydCBuZGFycmF5IGZyb20gJ25kYXJyYXknXG5cbmV4cG9ydCBjb25zdCByZWFkU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvcmVhZC5nbHNsJywgJ3V0ZjgnKTtcbmV4cG9ydCBjb25zdCB3cml0ZVNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3dyaXRlLmdsc2wnLCAndXRmOCcpO1xuXG5leHBvcnQgZnVuY3Rpb24gaW5pdChzaGFwZSl7XG4gICAgLy8gdmFyIGxlbmd0aCA9IDQgKiBNYXRoLmNlaWwoc2hhcGVbMl0gLyA0KSAqIHNoYXBlWzNdICogc2hhcGVbMV0gKiBzaGFwZVswXTtcbiAgICAvLyB2YXIgY29scyA9IE1hdGguY2VpbChNYXRoLnNxcnQobGVuZ3RoKSAvIDQpICogNDtcblxuICAgIHZhciBsZW5ndGggPSBzaGFwZVsyXSAqIHNoYXBlWzNdICogc2hhcGVbMV0gKiBzaGFwZVswXTtcbiAgICB2YXIgY29scyA9IE1hdGguY2VpbChNYXRoLnNxcnQobGVuZ3RoKSk7XG4gICAgdmFyIHRleFNpemUgPSBbY29scywgTWF0aC5jZWlsKGxlbmd0aCAvIGNvbHMpXVxuICAgIHJldHVybiB7XG4gICAgICAgIHRleFNpemU6IHRleFNpemUsXG4gICAgICAgIHNoYXBlOiBzaGFwZSxcbiAgICAgICAgLy8gdmVjNCgxLCBAc2hhcGUueCwgQHNoYXBlLnggKiBAc2hhcGUueSwgQHNoYXBlLnggKiBAc2hhcGUueSAqIEBzaGFwZS56KVxuICAgICAgICBzdHJpZGU6IFsxLCBzaGFwZVswXSwgc2hhcGVbMF0gKiBzaGFwZVsxXSwgc2hhcGVbMF0gKiBzaGFwZVsxXSAqIHNoYXBlWzJdXVxuICAgIH1cbn1cblxuXG5leHBvcnQgZnVuY3Rpb24gcGFjayhpbmZvLCBhcnJheSwgZW5jb2RlMSwgZm9ybWF0KXtcbiAgICAvLyByZXR1cm4gVWludDhBcnJheSBvciBGbG9hdDMyQXJyYXlcbiAgICBhcnJheSA9IG5kYXJyYXkoYXJyYXkuZGF0YSwgXG4gICAgICAgIGFycmF5LnNoYXBlLmNvbmNhdChbMSwgMSwgMSwgMV0pLnNsaWNlKDAsIDQpLFxuICAgICAgICBhcnJheS5zdHJpZGUuY29uY2F0KFsxLCAxLCAxLCAxXSkuc2xpY2UoMCwgNCksXG4gICAgICAgIGFycmF5Lm9mZnNldClcblxuICAgIHZhciBzaGFwZSA9IGluZm8uc2hhcGU7XG4gICAgdmFyIGxlbmd0aCA9IGluZm8udGV4U2l6ZVswXSAqIGluZm8udGV4U2l6ZVsxXSAqIDQ7XG5cbiAgICBpZihmb3JtYXQudHlwZSA9PT0gJ2Zsb2F0MzInKXtcbiAgICAgICAgdmFyIGRhdGEgPSBuZXcgRmxvYXQzMkFycmF5KGxlbmd0aCk7ICAgIFxuICAgIH1lbHNlIGlmKGZvcm1hdC50eXBlID09PSAndWludDgnKXtcbiAgICAgICAgdmFyIGRhdGEgPSBuZXcgVWludDhBcnJheShsZW5ndGgpOyAgICBcbiAgICB9XG5cbiAgICBmb3IodmFyIHggPSAwOyB4IDwgc2hhcGVbMF07IHgrKyl7XG4gICAgICAgIGZvcih2YXIgeSA9IDA7IHkgPCBzaGFwZVsxXTsgeSsrKXtcbiAgICAgICAgICAgIGZvcih2YXIgeiA9IDA7IHogPCBzaGFwZVsyXTsgeisrKXtcbiAgICAgICAgICAgICAgICBmb3IodmFyIHcgPSAwOyB3IDwgc2hhcGVbM107IHcrKyl7XG4gICAgICAgICAgICAgICAgICAgIHZhciB0aWxlICA9IHggKyBcbiAgICAgICAgICAgICAgICAgICAgICAgIHkgKiBzaGFwZVswXSArIFxuICAgICAgICAgICAgICAgICAgICAgICAgeiAqIHNoYXBlWzBdICogc2hhcGVbMV0gK1xuICAgICAgICAgICAgICAgICAgICAgICAgdyAqIHNoYXBlWzBdICogc2hhcGVbMV0gKiBzaGFwZVsyXTtcblxuICAgICAgICAgICAgICAgICAgICBlbmNvZGUxKGRhdGEuc3ViYXJyYXkoNCp0aWxlLCA0KnRpbGUrNCksIGFycmF5LmdldCh4LCB5LCB6LCB3KSwgaW5mbylcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gZGF0YTtcbn1cblxuXG5leHBvcnQgZnVuY3Rpb24gdW5wYWNrKGluZm8sIGRhdGEsIGRlY29kZTEsIHR5cGUpe1xuICAgIGlmKHR5cGUgIT0gJ2Zsb2F0MzInKSB0aHJvdyBuZXcgRXJyb3IoJ25vdCBpbXBsJyk7XG5cbiAgICB2YXIgc2hhcGUgPSBpbmZvLnNoYXBlO1xuICAgIHZhciBsZW5ndGggPSBzaGFwZS5yZWR1Y2UoKGEsIGIpID0+IGEgKiBiKTtcblxuICAgIHZhciBhcnJheSA9IG5kYXJyYXkobmV3IEZsb2F0MzJBcnJheShsZW5ndGgpLCBcbiAgICAgICAgc2hhcGUuY29uY2F0KFsxLCAxLCAxLCAxXSkuc2xpY2UoMCwgNCkpXG5cblxuICAgIGZvcih2YXIgeCA9IDA7IHggPCBzaGFwZVswXTsgeCsrKXtcbiAgICAgICAgZm9yKHZhciB5ID0gMDsgeSA8IHNoYXBlWzFdOyB5Kyspe1xuICAgICAgICAgICAgZm9yKHZhciB6ID0gMDsgeiA8IHNoYXBlWzJdOyB6Kyspe1xuICAgICAgICAgICAgICAgIGZvcih2YXIgdyA9IDA7IHcgPCBzaGFwZVszXTsgdysrKXtcbiAgICAgICAgICAgICAgICAgICAgdmFyIHRpbGUgID0geCArIFxuICAgICAgICAgICAgICAgICAgICAgICAgeSAqIHNoYXBlWzBdICsgXG4gICAgICAgICAgICAgICAgICAgICAgICB6ICogc2hhcGVbMF0gKiBzaGFwZVsxXSArXG4gICAgICAgICAgICAgICAgICAgICAgICB3ICogc2hhcGVbMF0gKiBzaGFwZVsxXSAqIHNoYXBlWzJdO1xuXG4gICAgICAgICAgICAgICAgICAgIGFycmF5LnNldCh4LCB5LCB6LCB3LCBkZWNvZGUxKGRhdGEuc3ViYXJyYXkoNCp0aWxlLCA0KnRpbGUrNCksIGluZm8pKVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gYXJyYXk7XG59IiwiaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xuXG5leHBvcnQgY29uc3QgcmVhZFNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3JlYWQuZ2xzbCcsICd1dGY4Jyk7XG5leHBvcnQgY29uc3Qgd3JpdGVTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy93cml0ZS5nbHNsJywgJ3V0ZjgnKTtcbmltcG9ydCBuZGFycmF5IGZyb20gJ25kYXJyYXknXG5cblxuZXhwb3J0IGZ1bmN0aW9uIGluaXQoc2hhcGUpe1xuICAgIHZhciB3aWR0aCA9IHNoYXBlWzBdO1xuICAgIC8vIHdlIHBpY2sgdGhlIG51bWJlciBvZiBjb2x1bW5zIHNvIHdlIGNhbiBrZWVwXG4gICAgLy8gdGhlIHRleHR1cmUgYXMgc3F1YXJlIGFzIHBvc3NpYmxlLCB3aXRoIHRoZVxuICAgIC8vIG1pbmltYWwgYW1vdW50IG9mIHdhc3RlZCBzcGFjZS5cblxuICAgIHZhciB0aWxlcyA9IHNoYXBlWzJdICogc2hhcGVbM10sXG4gICAgICAgIGNvbHMgPSBNYXRoLm1heCgxLCBNYXRoLm1pbih0aWxlcywgTWF0aC5jZWlsKFxuICAgICAgICAgICAgTWF0aC5zcXJ0KHNoYXBlWzBdICogc2hhcGVbMV0gKiB0aWxlcykgLyB3aWR0aCkpKTtcblxuICAgIHZhciB0ZXhTaXplID0gW3dpZHRoICogY29scywgc2hhcGVbMV0gKiBNYXRoLmNlaWwodGlsZXMgLyBjb2xzKV1cblxuICAgIHJldHVybiB7XG4gICAgICAgIHRleFNpemU6IHRleFNpemUsXG4gICAgICAgIGNvbHM6IGNvbHMsXG4gICAgICAgIHNoYXBlOiBzaGFwZSxcbiAgICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBwYWNrKGluZm8sIG5kYXJyYXkpe1xuICAgIC8vIHJldHVybiBVaW50OEFycmF5IG9yIEZsb2F0MzJBcnJheVxuXG5cbi8vIHVuaWZvcm0gc2FtcGxlcjJEIEBfdGV4O1xuLy8gdW5pZm9ybSBpdmVjMiBAX3RleFNpemU7XG4vLyB1bmlmb3JtIGl2ZWM0IEBfc2hhcGU7XG4vLyB1bmlmb3JtIGludCBAX2NvbHM7XG5cbiAgICAvLyByZXR1cm4ge1xuICAgIC8vICB0ZXg6XG4gICAgLy8gIHRleFNpemU6XG4gICAgLy8gIHNoYXBlOlxuICAgIC8vICBjb2xzOlxuICAgIC8vIH1cbiAgICB0aHJvdyBuZXcgRXJyb3IoXCJub3QgaW1wbGVtZW50ZWQ6IGZvcm1hdC8xLTQvcGFjay90aWxlL2luZGV4LmpzOnBhY2tcIilcbn1cblxuXG5leHBvcnQgZnVuY3Rpb24gdW5wYWNrKGluZm8sIGFycil7XG4gICAgLy8gcmV0dXJuIG5kYXJyYXlcbiAgICB0aHJvdyBuZXcgRXJyb3IoXCJub3QgaW1wbGVtZW50ZWQ6IGZvcm1hdC8xLTQvcGFjay90aWxlL2luZGV4LmpzOnVucGFja1wiKVxufSIsImltcG9ydCB7IHJlYWRGaWxlU3luYyB9IGZyb20gJ2ZzJztcblxuZXhwb3J0IGRlZmF1bHQge1xuXHRoYXJkX3NpZ21vaWQ6IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL2hhcmRfc2lnbW9pZC5nbHNsJywgJ3V0ZjgnKSxcblx0bGluZWFyOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9saW5lYXIuZ2xzbCcsICd1dGY4JyksXG5cdHJlbHU6IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3JlbHUuZ2xzbCcsICd1dGY4JyksXG5cdHJnYjogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvcmdiLmdsc2wnLCAndXRmOCcpLFxuXHRzaWdtb2lkOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9zaWdtb2lkLmdsc2wnLCAndXRmOCcpLFxuXHR0YW5oOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy90YW5oLmdsc2wnLCAndXRmOCcpLFxufSIsImltcG9ydCB7IHJlYWRGaWxlU3luYyB9IGZyb20gJ2ZzJztcblxuZXhwb3J0IGNvbnN0IGVuY29kZVNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL2VuY29kZS5nbHNsJywgJ3V0ZjgnKTtcbmV4cG9ydCBjb25zdCBkZWNvZGVTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9kZWNvZGUuZ2xzbCcsICd1dGY4Jyk7XG5cbmV4cG9ydCBmdW5jdGlvbiBpbml0KHNoYXBlLCBmb3JtYXQpe1xuXHRyZXR1cm4ge1xuXHRcdHJhbmdlOiBbXG5cdFx0XHRpc0Zpbml0ZShmb3JtYXQubWluKSA/IGZvcm1hdC5taW4gOiAwLFxuXHRcdFx0aXNGaW5pdGUoZm9ybWF0Lm1heCkgPyBmb3JtYXQubWF4IDogMVxuXHRcdF1cblx0XHQvLyBtYXg6ICxcblx0XHQvLyBtaW46ICxcblx0fVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZW5jb2RlKGRhdGEsIHIsIGcsIGIsIGEsIGluZm8pe1xuXG5cdGRhdGFbMF0gPSBNYXRoLnJvdW5kKDI1NSAqIE1hdGgubWluKDEsIE1hdGgubWF4KDAsIChyIC0gaW5mby5yYW5nZVswXSkvKGluZm8ucmFuZ2VbMV0gLSBpbmZvLnJhbmdlWzBdKSApKSlcblx0ZGF0YVsxXSA9IE1hdGgucm91bmQoMjU1ICogTWF0aC5taW4oMSwgTWF0aC5tYXgoMCwgKGcgLSBpbmZvLnJhbmdlWzBdKS8oaW5mby5yYW5nZVsxXSAtIGluZm8ucmFuZ2VbMF0pICkpKVxuXHRkYXRhWzJdID0gTWF0aC5yb3VuZCgyNTUgKiBNYXRoLm1pbigxLCBNYXRoLm1heCgwLCAoYiAtIGluZm8ucmFuZ2VbMF0pLyhpbmZvLnJhbmdlWzFdIC0gaW5mby5yYW5nZVswXSkgKSkpXG5cdGRhdGFbM10gPSBNYXRoLnJvdW5kKDI1NSAqIE1hdGgubWluKDEsIE1hdGgubWF4KDAsIChhIC0gaW5mby5yYW5nZVswXSkvKGluZm8ucmFuZ2VbMV0gLSBpbmZvLnJhbmdlWzBdKSApKSlcblx0Ly8gY29uc29sZS5sb2coZGF0YVswXSwgZGF0YVsxXSwgZGF0YVsyXSlcbn1cblxuXG5leHBvcnQgZnVuY3Rpb24gZGVjb2RlKGRhdGEsIHIsIGcsIGIsIGEsIGluZm8pe1xuXHRkYXRhWzBdID0gKHIgLyAyNTUpICogKGluZm8ucmFuZ2VbMV0gLSBpbmZvLnJhbmdlWzBdKSArIGluZm8ucmFuZ2VbMF07XG5cdGRhdGFbMV0gPSAoZyAvIDI1NSkgKiAoaW5mby5yYW5nZVsxXSAtIGluZm8ucmFuZ2VbMF0pICsgaW5mby5yYW5nZVswXTtcblx0ZGF0YVsyXSA9IChiIC8gMjU1KSAqIChpbmZvLnJhbmdlWzFdIC0gaW5mby5yYW5nZVswXSkgKyBpbmZvLnJhbmdlWzBdO1xuXHRkYXRhWzNdID0gKGEgLyAyNTUpICogKGluZm8ucmFuZ2VbMV0gLSBpbmZvLnJhbmdlWzBdKSArIGluZm8ucmFuZ2VbMF07XG59IiwiaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xuXG5leHBvcnQgY29uc3QgZW5jb2RlU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvZW5jb2RlLmdsc2wnLCAndXRmOCcpO1xuZXhwb3J0IGNvbnN0IGRlY29kZVNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL2RlY29kZS5nbHNsJywgJ3V0ZjgnKTtcblxuZXhwb3J0IGZ1bmN0aW9uIGluaXQoc2hhcGUsIGZvcm1hdCl7XG5cdHJldHVybiB7IH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGVuY29kZShkYXRhLCByLCBnLCBiLCBhKXtcblx0ZGF0YVswXSA9IHI7XG5cdGRhdGFbMV0gPSBnO1xuXHRkYXRhWzJdID0gYjtcblx0ZGF0YVszXSA9IGE7XG59XG5cblxuZXhwb3J0IGZ1bmN0aW9uIGRlY29kZShkYXRhLCByLCBnLCBiLCBhKXtcblx0ZGF0YVswXSA9IHI7XG5cdGRhdGFbMV0gPSBnO1xuXHRkYXRhWzJdID0gYjtcblx0ZGF0YVszXSA9IGE7XG59IiwiaW1wb3J0ICogYXMgcGFja19zdHJpZGUgZnJvbSAnLi9wYWNrL3N0cmlkZS9pbmRleC5qcydcbmltcG9ydCAqIGFzIHBhY2tfdGlsZSBmcm9tICcuL3BhY2svdGlsZS9pbmRleC5qcydcblxuaW1wb3J0ICogYXMgY29kZWNfcmF3IGZyb20gJy4vY29kZWMvcmF3L2luZGV4LmpzJ1xuaW1wb3J0ICogYXMgY29kZWNfbGlucXVhbnQgZnJvbSAnLi9jb2RlYy9saW5xdWFudC9pbmRleC5qcydcblxuaW1wb3J0IGFjdGl2YXRpb25zIGZyb20gJy4vYWN0aXZhdGlvbi9pbmRleC5qcydcblxuaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xuXG5leHBvcnQgZGVmYXVsdCB7XG5cdHBhY2s6IHtcblx0XHRzdHJpZGU6IHBhY2tfc3RyaWRlLFxuXHRcdHRpbGU6IHBhY2tfdGlsZVxuXHR9LFxuXG5cblx0cmVhZF9zaGltOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9wYWNrL3JlYWRfc2hpbS5nbHNsJywgJ3V0ZjgnKSxcblx0d3JpdGVfc2hpbTogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvcGFjay93cml0ZV9zaGltLmdsc2wnLCAndXRmOCcpLFxuXG5cdGNvZGVjOiB7XG5cdFx0cmF3OiBjb2RlY19yYXcsXG5cdFx0bGlucXVhbnQ6IGNvZGVjX2xpbnF1YW50LFxuXHR9LFxuXHRhY3RpdmF0aW9uczogYWN0aXZhdGlvbnNcbn0iLCJpbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5cbmV4cG9ydCBjb25zdCByZWFkU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvcmVhZC5nbHNsJywgJ3V0ZjgnKTtcbmV4cG9ydCBjb25zdCB3cml0ZVNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3dyaXRlLmdsc2wnLCAndXRmOCcpO1xuaW1wb3J0IG5kYXJyYXkgZnJvbSAnbmRhcnJheSdcblxuZXhwb3J0IGZ1bmN0aW9uIGluaXQoc2hhcGUpe1xuICAgIHZhciBsZW5ndGggPSBNYXRoLmNlaWwoc2hhcGVbMl0gLyA0KSAqIHNoYXBlWzNdICogc2hhcGVbMV0gKiBzaGFwZVswXTtcbiAgICB2YXIgY29scyA9IE1hdGguY2VpbChNYXRoLnNxcnQobGVuZ3RoKSk7XG4gICAgdmFyIHRleFNpemUgPSBbY29scywgTWF0aC5jZWlsKGxlbmd0aCAvIGNvbHMpXVxuXG4gICAgY29uc29sZS5hc3NlcnQodGV4U2l6ZVswXSAqIHRleFNpemVbMV0gPj0gbGVuZ3RoKVxuICAgIHJldHVybiB7XG4gICAgICAgIHRleFNpemU6IHRleFNpemUsXG4gICAgICAgIHNoYXBlOiBzaGFwZSxcblxuICAgICAgICBzdHJpZGU6IFtcbiAgICAgICAgICAgIDEsIFxuICAgICAgICAgICAgc2hhcGVbMF0sIFxuICAgICAgICAgICAgc2hhcGVbMF0gKiBzaGFwZVsxXSAvIDQsICAvLyB0aGUgLzQgaXMgYmVjYXVzZSBvZiB0aGUgY29sb3IgY2hhbm5lbFxuICAgICAgICAgICAgc2hhcGVbMF0gKiBzaGFwZVsxXSAqIE1hdGguY2VpbChzaGFwZVsyXSAvIDQpXG4gICAgICAgIF0sXG4gICAgICAgIC8vIGRlY3ZlYzogWzEsIHNoYXBlWzBdLCBzaGFwZVswXSAqIHNoYXBlWzFdLCBzaGFwZVswXSAqIHNoYXBlWzFdICogTWF0aC5jZWlsKHNoYXBlWzJdIC8gNCldXG4gICAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gcGFjayhpbmZvLCBhcnJheSwgZW5jb2RlNCwgZm9ybWF0KXtcbiAgICAvLyByZXR1cm4gVWludDhBcnJheSBvciBGbG9hdDMyQXJyYXlcblxuICAgIGFycmF5ID0gbmRhcnJheShhcnJheS5kYXRhLCBcbiAgICAgICAgYXJyYXkuc2hhcGUuY29uY2F0KFsxLCAxLCAxLCAxXSkuc2xpY2UoMCwgNCksXG4gICAgICAgIGFycmF5LnN0cmlkZS5jb25jYXQoWzEsIDEsIDEsIDFdKS5zbGljZSgwLCA0KSxcbiAgICAgICAgYXJyYXkub2Zmc2V0KVxuICAgIFxuICAgIHZhciBbd2lkdGgsIGhlaWdodF0gPSBpbmZvLnRleFNpemUsXG4gICAgICAgIGxlbmd0aCA9IHdpZHRoICogaGVpZ2h0ICogNDtcbiAgICB2YXIgc2hhcGUgPSBpbmZvLnNoYXBlO1xuXG4gICAgaWYoZm9ybWF0LnR5cGUgPT09ICdmbG9hdDMyJyl7XG4gICAgICAgIHZhciBkYXRhID0gbmV3IEZsb2F0MzJBcnJheShsZW5ndGgpOyAgICBcbiAgICB9ZWxzZSBpZihmb3JtYXQudHlwZSA9PT0gJ3VpbnQ4Jyl7XG4gICAgICAgIHZhciBkYXRhID0gbmV3IFVpbnQ4QXJyYXkobGVuZ3RoKTsgICAgXG4gICAgfVxuXG4gICAgdmFyIGNoYW5zID0gTWF0aC5jZWlsKGluZm8uc2hhcGVbMl0gLyA0KTtcblxuICAgIGZvcih2YXIgaSA9IDA7IGkgPCBpbmZvLnNoYXBlWzBdOyBpKyspe1xuICAgICAgICBmb3IodmFyIGogPSAwOyBqIDwgaW5mby5zaGFwZVsxXTsgaisrKXtcbiAgICAgICAgICAgIGZvcih2YXIgayA9IDA7IGsgPCBjaGFuczsgaysrKXtcbiAgICAgICAgICAgICAgICB2YXIgYiA9IE1hdGgubWluKGsqNCs0LCBzaGFwZVsyXSktayo0O1xuICAgICAgICAgICAgICAgIGZvcih2YXIgdyA9IDA7IHcgPCBpbmZvLnNoYXBlWzNdOyB3Kyspe1xuXG4gICAgICAgICAgICAgICAgICAgIHZhciB0aWxlICA9IGkgKyBcbiAgICAgICAgICAgICAgICAgICAgICAgIGogKiBzaGFwZVswXSArIFxuICAgICAgICAgICAgICAgICAgICAgICAgayAqIHNoYXBlWzBdICogc2hhcGVbMV0gK1xuICAgICAgICAgICAgICAgICAgICAgICAgdyAqIHNoYXBlWzBdICogc2hhcGVbMV0gKiBjaGFucztcblxuXG4gICAgICAgICAgICAgICAgICAgIHZhciBwb3MgPSA0ICogdGlsZTtcbiAgICAgICAgICAgICAgICAgICAgZW5jb2RlNChcbiAgICAgICAgICAgICAgICAgICAgICAgIGRhdGEuc3ViYXJyYXkocG9zLCBwb3MgKyA0KSxcbiAgICAgICAgICAgICAgICAgICAgICAgIGIgPCAxID8gMCA6IGFycmF5LmdldChpLCBqLCA0KmsrMCwgdyksIFxuICAgICAgICAgICAgICAgICAgICAgICAgYiA8IDIgPyAwIDogYXJyYXkuZ2V0KGksIGosIDQqaysxLCB3KSwgXG4gICAgICAgICAgICAgICAgICAgICAgICBiIDwgMyA/IDAgOiBhcnJheS5nZXQoaSwgaiwgNCprKzIsIHcpLCBcbiAgICAgICAgICAgICAgICAgICAgICAgIGIgPCA0ID8gMCA6IGFycmF5LmdldChpLCBqLCA0KmsrMywgdyksIGluZm8pXG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIGRhdGFcbn1cblxuXG5leHBvcnQgZnVuY3Rpb24gdW5wYWNrKGluZm8sIGRhdGEsIGRlY29kZTQsIHR5cGUpe1xuXG5cblxuICAgIHZhciBzaGFwZSA9IGluZm8uc2hhcGU7XG4gICAgdmFyIHNoYXBlbGVuZ3RoID0gc2hhcGUucmVkdWNlKChhLCBiKSA9PiBhICogYik7XG4gICAgXG4gICAgdmFyIFt3aWR0aCwgaGVpZ2h0XSA9IGluZm8udGV4U2l6ZSxcbiAgICAgICAgbGVuZ3RoID0gd2lkdGggKiBoZWlnaHQgKiA0O1xuICAgIHZhciBjaGFucyA9IE1hdGguY2VpbChpbmZvLnNoYXBlWzJdIC8gNCk7XG5cbiAgICAvLyBpZih0eXBlID09PSAnZmxvYXQzMicpe1xuICAgIHZhciBhcnJheSA9IG5kYXJyYXkobmV3IEZsb2F0MzJBcnJheShzaGFwZWxlbmd0aCksIHNoYXBlKVxuICAgIHZhciBidWYgPSBuZXcgRmxvYXQzMkFycmF5KDQpO1xuICAgIC8vIH1lbHNlIGlmKHR5cGUgPT0gJ3VpbnQ4Jyl7XG4gICAgLy8gICAgIHZhciBhcnJheSA9IG5kYXJyYXkobmV3IFVpbnQ4QXJyYXkoc2hhcGVsZW5ndGgpLCBzaGFwZSlcbiAgICAvLyAgICAgdmFyIGJ1ZiA9IG5ldyBVaW50OEFycmF5KDQpO1xuICAgIC8vIH1lbHNlIHRocm93IG5ldyBFcnJvcigndW5pbXBsZW1lbnRlZCB0eXBlJyk7XG4gICAgXG5cbiAgICBmb3IodmFyIGkgPSAwOyBpIDwgaW5mby5zaGFwZVswXTsgaSsrKXtcbiAgICAgICAgZm9yKHZhciBqID0gMDsgaiA8IGluZm8uc2hhcGVbMV07IGorKyl7XG4gICAgICAgICAgICBmb3IodmFyIGsgPSAwOyBrIDwgY2hhbnM7IGsrKyl7XG4gICAgICAgICAgICAgICAgdmFyIGIgPSBNYXRoLm1pbihrKjQrNCwgc2hhcGVbMl0pLWsqNDtcbiAgICAgICAgICAgICAgICBmb3IodmFyIHcgPSAwOyB3IDwgaW5mby5zaGFwZVszXTsgdysrKXtcblxuICAgICAgICAgICAgICAgICAgICB2YXIgdGlsZSAgPSBcbiAgICAgICAgICAgICAgICAgICAgICAgIGkgKyBcbiAgICAgICAgICAgICAgICAgICAgICAgIGogKiBzaGFwZVswXSArIFxuICAgICAgICAgICAgICAgICAgICAgICAgayAqIHNoYXBlWzBdICogc2hhcGVbMV0gK1xuICAgICAgICAgICAgICAgICAgICAgICAgdyAqIHNoYXBlWzBdICogc2hhcGVbMV0gKiBjaGFucztcblxuICAgICAgICAgICAgICAgICAgICBkZWNvZGU0KGJ1ZiwgXG4gICAgICAgICAgICAgICAgICAgICAgICBkYXRhWzQgKiB0aWxlICsgMF0sXG4gICAgICAgICAgICAgICAgICAgICAgICBkYXRhWzQgKiB0aWxlICsgMV0sXG4gICAgICAgICAgICAgICAgICAgICAgICBkYXRhWzQgKiB0aWxlICsgMl0sXG4gICAgICAgICAgICAgICAgICAgICAgICBkYXRhWzQgKiB0aWxlICsgM10sIGluZm8pXG5cblxuICAgICAgICAgICAgICAgICAgICBmb3IodmFyIHggPSAwOyB4IDwgYjsgeCsrKXtcbiAgICAgICAgICAgICAgICAgICAgICAgIGFycmF5LnNldChpLCBqLCA0KmsreCwgdywgYnVmW3hdKVxuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIGFycmF5O1xuXG59XG4iLCJpbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5cbmV4cG9ydCBjb25zdCByZWFkU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvcmVhZC5nbHNsJywgJ3V0ZjgnKTtcbmV4cG9ydCBjb25zdCB3cml0ZVNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3dyaXRlLmdsc2wnLCAndXRmOCcpO1xuXG5leHBvcnQgZnVuY3Rpb24gaW5pdChzaGFwZSl7XG4gICAgdmFyIHdpZHRoID0gc2hhcGVbMF07IC8vIHZhciB3aWR0aCA9IHNoYXBlWzBdICogNDsgICAgXG4gICAgLy8gd2UgcGljayB0aGUgbnVtYmVyIG9mIGNvbHVtbnMgc28gd2UgY2FuIGtlZXBcbiAgICAvLyB0aGUgdGV4dHVyZSBhcyBzcXVhcmUgYXMgcG9zc2libGUsIHdpdGggdGhlXG4gICAgLy8gbWluaW1hbCBhbW91bnQgb2Ygd2FzdGVkIHNwYWNlLlxuXG4gICAgdmFyIHRpbGVzID0gTWF0aC5jZWlsKHNoYXBlWzJdIC8gNCkgKiBzaGFwZVszXSxcbiAgICAgICAgY29scyA9IE1hdGgubWF4KDEsIE1hdGgubWluKHRpbGVzLCBNYXRoLnJvdW5kKFxuICAgICAgICAgICAgTWF0aC5zcXJ0KHNoYXBlWzBdICogc2hhcGVbMV0gKiB0aWxlcykgLyB3aWR0aCkpKTtcblxuICAgIHZhciB0ZXhTaXplID0gW3dpZHRoICogY29scywgc2hhcGVbMV0gKiBNYXRoLmNlaWwodGlsZXMgLyBjb2xzKV1cblxuICAgIHJldHVybiB7XG4gICAgXHR0ZXhTaXplOiB0ZXhTaXplLFxuICAgIFx0Y29sczogY29scyxcbiAgICBcdHNoYXBlOiBzaGFwZSxcbiAgICB9XG59XG5cbmltcG9ydCBuZGFycmF5IGZyb20gXCJuZGFycmF5XCJcblxuZXhwb3J0IGZ1bmN0aW9uIHBhY2soaW5mbywgYXJyYXksIGVuY29kZTQsIGZvcm1hdCl7XG4gICAgYXJyYXkgPSBuZGFycmF5KGFycmF5LmRhdGEsIFxuICAgICAgICBhcnJheS5zaGFwZS5jb25jYXQoWzEsIDEsIDEsIDFdKS5zbGljZSgwLCA0KSxcbiAgICAgICAgYXJyYXkuc3RyaWRlLmNvbmNhdChbMSwgMSwgMSwgMV0pLnNsaWNlKDAsIDQpLFxuICAgICAgICBhcnJheS5vZmZzZXQpXG5cbiAgICB2YXIgc2hhcGUgPSBhcnJheS5zaGFwZSxcbiAgICAgICAgdGlsZXMgPSBNYXRoLmNlaWwoc2hhcGVbMl0gLyA0KSAqIHNoYXBlWzNdLFxuICAgICAgICB0dyA9IHNoYXBlWzBdLFxuICAgICAgICB0aCA9IHNoYXBlWzFdLFxuICAgICAgICBjb2xzID0gaW5mby5jb2xzLFxuICAgICAgICBbd2lkdGgsIGhlaWdodF0gPSBpbmZvLnRleFNpemUsXG4gICAgICAgIGNodW5rcyA9IE1hdGguY2VpbChzaGFwZVsyXSAvIDQpLFxuICAgICAgICBsZW5ndGggPSB3aWR0aCAqIGhlaWdodCAqIDQ7XG5cbiAgICBpZihmb3JtYXQudHlwZSA9PT0gJ2Zsb2F0MzInKXtcbiAgICAgICAgdmFyIGRhdGEgPSBuZXcgRmxvYXQzMkFycmF5KGxlbmd0aCk7ICAgIFxuICAgIH1lbHNlIGlmKGZvcm1hdC50eXBlID09PSAndWludDgnKXtcbiAgICAgICAgdmFyIGRhdGEgPSBuZXcgVWludDhBcnJheShsZW5ndGgpOyAgICBcbiAgICB9XG4gICAgXG5cbiAgICBmb3IodmFyIHogPSAwOyB6IDwgY2h1bmtzOyB6Kyspe1xuICAgICAgICBmb3IodmFyIHcgPSAwOyB3IDwgc2hhcGVbM107IHcrKyl7XG4gICAgICAgICAgICB2YXIgdGlsZSA9IHcgKiBjaHVua3MgKyB6O1xuICAgICAgICAgICAgdmFyIGIgPSBNYXRoLm1pbih6KjQrNCwgc2hhcGVbMl0pLXoqNDtcbiAgICAgICAgICAgIFxuICAgICAgICAgICAgdmFyIGloID0gdGggKiBNYXRoLmZsb29yKHRpbGUgLyBjb2xzKTtcbiAgICAgICAgICAgIHZhciBqdyA9IHR3ICogKHRpbGUgJSBjb2xzKTtcblxuICAgICAgICAgICAgZm9yKHZhciBpID0gMDsgaSA8IHR3OyBpKyspe1xuICAgICAgICAgICAgICAgIGZvcih2YXIgaiA9IDA7IGogPCB0aDsgaisrKXtcblxuICAgICAgICAgICAgICAgICAgICB2YXIgcG9zID0gNCAqICgoaWgraikgKiB3aWR0aCArIGp3ICsgaSk7XG4gICAgICAgICAgICAgICAgICAgIGVuY29kZTQoXG4gICAgICAgICAgICAgICAgICAgICAgICBkYXRhLnN1YmFycmF5KHBvcywgcG9zICsgNCksXG4gICAgICAgICAgICAgICAgICAgICAgICBiIDwgMSA/IDAgOiBhcnJheS5nZXQoaSwgaiwgNCp6KzAsIHcpLCBcbiAgICAgICAgICAgICAgICAgICAgICAgIGIgPCAyID8gMCA6IGFycmF5LmdldChpLCBqLCA0KnorMSwgdyksIFxuICAgICAgICAgICAgICAgICAgICAgICAgYiA8IDMgPyAwIDogYXJyYXkuZ2V0KGksIGosIDQqeisyLCB3KSwgXG4gICAgICAgICAgICAgICAgICAgICAgICBiIDwgNCA/IDAgOiBhcnJheS5nZXQoaSwgaiwgNCp6KzMsIHcpLCBpbmZvKVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gZGF0YTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHVucGFjayhpbmZvLCBkYXRhLCBkZWNvZGU0LCB0eXBlKXtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXCJub3QgaW1wbGVtZW50ZWQ6IGZvcm1hdC80LTQvcGFjay90aWxlL2luZGV4LmpzOnVucGFja1wiKVxufSIsImltcG9ydCBGb3JtYXQ0NCBmcm9tICcuLzQtNC9pbmRleC5qcydcbmltcG9ydCBGb3JtYXQxNCBmcm9tICcuLzEtNC9pbmRleC5qcydcblxuZXhwb3J0IGRlZmF1bHQge1xuXHQnNDo0JzogRm9ybWF0NDQsXG5cdCcxOjQnOiBGb3JtYXQxNCxcbn0iLCIvLyBkbyB5b3UgZXZlciBob3BlIHRoYXQgcGVyaGFwcyBpbmRleCBmaWxlcyBzaG91bGQgXG4vLyBhY3R1YWxseSBiZSBpbmRleCBmaWxlcyBsYWNraW5nIGFueSBpbXBsZW1lbnRhdGlvbiBcbi8vIGNvZGU/IHdlbGwsIHRvZGF5IHlvdSdyZSBpbiBsdWNrIVxuXG5leHBvcnQgeyBUZW5zb3IsIE91dHB1dFRlbnNvciwgSW5QbGFjZVRlbnNvciB9IGZyb20gJy4vdGVuc29yL2luZGV4LmpzJ1xuZXhwb3J0IHsgUnVuLCBDb21waWxlIH0gZnJvbSAnLi9ydW50aW1lL2luZGV4LmpzJ1xuZXhwb3J0IHsgY3JlYXRlR0wgfSBmcm9tICcuL3V0aWwuanMnIiwiLy8gY29kZSBmb3IgcHJldHR5IHByaW50aW5nIHNoYWRlciBlcnJvcnMgZnJvbSByZWdsXG5cbmV4cG9ydCBmdW5jdGlvbiBjaGVja0xpbmtFcnJvciAoZ2wsIHByb2dyYW0sIGZyYWdTaGFkZXIsIHZlcnRTaGFkZXIsIGNvbW1hbmQpIHtcbiAgICBpZiAoIWdsLmdldFByb2dyYW1QYXJhbWV0ZXIocHJvZ3JhbSwgZ2wuTElOS19TVEFUVVMpKSB7XG4gICAgICAgIHZhciBlcnJMb2cgPSBnbC5nZXRQcm9ncmFtSW5mb0xvZyhwcm9ncmFtKVxuICAgICAgICB2YXIgZnJhZ1BhcnNlID0gcGFyc2VTb3VyY2UoZnJhZ1NoYWRlciwgY29tbWFuZClcbiAgICAgICAgdmFyIHZlcnRQYXJzZSA9IHBhcnNlU291cmNlKHZlcnRTaGFkZXIsIGNvbW1hbmQpXG5cbiAgICAgICAgdmFyIGhlYWRlciA9ICdFcnJvciBsaW5raW5nIHByb2dyYW0gd2l0aCB2ZXJ0ZXggc2hhZGVyLCBcIicgK1xuICAgICAgICAgICAgdmVydFBhcnNlWzBdLm5hbWUgKyAnXCIsIGFuZCBmcmFnbWVudCBzaGFkZXIgXCInICsgZnJhZ1BhcnNlWzBdLm5hbWUgKyAnXCInXG5cbiAgICAgICAgaWYgKHR5cGVvZiBkb2N1bWVudCAhPT0gJ3VuZGVmaW5lZCcpIHtcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKCclYycgKyBoZWFkZXIgKyAnXFxuJWMnICsgZXJyTG9nLFxuICAgICAgICAgICAgICAgICdjb2xvcjpyZWQ7dGV4dC1kZWNvcmF0aW9uOnVuZGVybGluZTtmb250LXdlaWdodDpib2xkJyxcbiAgICAgICAgICAgICAgICAnY29sb3I6cmVkJylcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIGNvbnNvbGUubG9nKGhlYWRlciArICdcXG4nICsgZXJyTG9nKVxuICAgICAgICB9XG5cbiAgICAgICAgY29uc29sZS5sb2coZnJhZ1NoYWRlcik7XG4gICAgICAgIFxuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoaGVhZGVyKVxuICAgIH1cbn1cblxuXG5leHBvcnQgZnVuY3Rpb24gY2hlY2tTaGFkZXJFcnJvciAoZ2wsIHNoYWRlciwgc291cmNlLCB0eXBlLCBjb21tYW5kKSB7XG4gICAgaWYgKCFnbC5nZXRTaGFkZXJQYXJhbWV0ZXIoc2hhZGVyLCBnbC5DT01QSUxFX1NUQVRVUykpIHtcbiAgICAgICAgdmFyIGVyckxvZyA9IGdsLmdldFNoYWRlckluZm9Mb2coc2hhZGVyKVxuICAgICAgICB2YXIgdHlwZU5hbWUgPSB0eXBlID09PSBnbC5GUkFHTUVOVF9TSEFERVIgPyAnZnJhZ21lbnQnIDogJ3ZlcnRleCdcbiAgICAgICAgLy8gY2hlY2tDb21tYW5kVHlwZShzb3VyY2UsICdzdHJpbmcnLCB0eXBlTmFtZSArICcgc2hhZGVyIHNvdXJjZSBtdXN0IGJlIGEgc3RyaW5nJywgY29tbWFuZClcblxuICAgICAgICB2YXIgZmlsZXMgPSBwYXJzZVNvdXJjZShzb3VyY2UsIGNvbW1hbmQpXG4gICAgICAgIHZhciBlcnJvcnMgPSBwYXJzZUVycm9yTG9nKGVyckxvZylcbiAgICAgICAgYW5ub3RhdGVGaWxlcyhmaWxlcywgZXJyb3JzKVxuXG4gICAgICAgIE9iamVjdC5rZXlzKGZpbGVzKS5mb3JFYWNoKGZ1bmN0aW9uIChmaWxlTnVtYmVyKSB7XG4gICAgICAgICAgICB2YXIgZmlsZSA9IGZpbGVzW2ZpbGVOdW1iZXJdXG4gICAgICAgICAgICBpZiAoIWZpbGUuaGFzRXJyb3JzKSB7XG4gICAgICAgICAgICAgICAgcmV0dXJuXG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIHZhciBzdHJpbmdzID0gWycnXVxuICAgICAgICAgICAgdmFyIHN0eWxlcyA9IFsnJ11cblxuICAgICAgICAgICAgZnVuY3Rpb24gcHVzaCAoc3RyLCBzdHlsZSkge1xuICAgICAgICAgICAgICAgIHN0cmluZ3MucHVzaChzdHIpXG4gICAgICAgICAgICAgICAgc3R5bGVzLnB1c2goc3R5bGUgfHwgJycpXG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIHB1c2goJ2ZpbGUgbnVtYmVyICcgKyBmaWxlTnVtYmVyICsgJzogJyArIGZpbGUubmFtZSArICdcXG4nLCAnY29sb3I6cmVkO3RleHQtZGVjb3JhdGlvbjp1bmRlcmxpbmU7Zm9udC13ZWlnaHQ6Ym9sZCcpXG5cbiAgICAgICAgICAgIGZpbGUubGluZXMuZm9yRWFjaChmdW5jdGlvbiAobGluZSkge1xuICAgICAgICAgICAgICAgIGlmIChsaW5lLmVycm9ycy5sZW5ndGggPiAwKSB7XG4gICAgICAgICAgICAgICAgICAgIHB1c2gobGVmdFBhZChsaW5lLm51bWJlciwgNCkgKyAnfCAgJywgJ2JhY2tncm91bmQtY29sb3I6eWVsbG93OyBmb250LXdlaWdodDpib2xkJylcbiAgICAgICAgICAgICAgICAgICAgcHVzaChsaW5lLmxpbmUgKyAnXFxuJywgJ2NvbG9yOnJlZDsgYmFja2dyb3VuZC1jb2xvcjp5ZWxsb3c7IGZvbnQtd2VpZ2h0OmJvbGQnKVxuXG4gICAgICAgICAgICAgICAgICAgIC8vIHRyeSB0byBndWVzcyB0b2tlblxuICAgICAgICAgICAgICAgICAgICB2YXIgb2Zmc2V0ID0gMFxuICAgICAgICAgICAgICAgICAgICBsaW5lLmVycm9ycy5mb3JFYWNoKGZ1bmN0aW9uIChlcnJvcikge1xuICAgICAgICAgICAgICAgICAgICAgICAgdmFyIG1lc3NhZ2UgPSBlcnJvci5tZXNzYWdlXG4gICAgICAgICAgICAgICAgICAgICAgICB2YXIgdG9rZW4gPSAvXlxccypcXCcoLiopXFwnXFxzKlxcOlxccyooLiopJC8uZXhlYyhtZXNzYWdlKVxuICAgICAgICAgICAgICAgICAgICAgICAgaWYgKHRva2VuKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgdmFyIHRva2VuUGF0ID0gdG9rZW5bMV1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBtZXNzYWdlID0gdG9rZW5bMl1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBzd2l0Y2ggKHRva2VuUGF0KSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNhc2UgJ2Fzc2lnbic6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0b2tlblBhdCA9ICc9J1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgYnJlYWtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgb2Zmc2V0ID0gTWF0aC5tYXgobGluZS5saW5lLmluZGV4T2YodG9rZW5QYXQsIG9mZnNldCksIDApXG4gICAgICAgICAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIG9mZnNldCA9IDBcbiAgICAgICAgICAgICAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgICAgICAgICAgICAgcHVzaChsZWZ0UGFkKCd8ICcsIDYpKVxuICAgICAgICAgICAgICAgICAgICAgICAgcHVzaChsZWZ0UGFkKCdeXl4nLCBvZmZzZXQgKyAzKSArICdcXG4nLCAnZm9udC13ZWlnaHQ6Ym9sZCcpXG4gICAgICAgICAgICAgICAgICAgICAgICBwdXNoKGxlZnRQYWQoJ3wgJywgNikpXG4gICAgICAgICAgICAgICAgICAgICAgICBwdXNoKG1lc3NhZ2UgKyAnXFxuJywgJ2ZvbnQtd2VpZ2h0OmJvbGQnKVxuICAgICAgICAgICAgICAgICAgICB9KVxuICAgICAgICAgICAgICAgICAgICBwdXNoKGxlZnRQYWQoJ3wgJywgNikgKyAnXFxuJylcbiAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICBwdXNoKGxlZnRQYWQobGluZS5udW1iZXIsIDQpICsgJ3wgICcpXG4gICAgICAgICAgICAgICAgICAgIHB1c2gobGluZS5saW5lICsgJ1xcbicsICdjb2xvcjpyZWQnKVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH0pXG4gICAgICAgICAgICBpZiAodHlwZW9mIGRvY3VtZW50ICE9PSAndW5kZWZpbmVkJykge1xuICAgICAgICAgICAgICAgIHN0eWxlc1swXSA9IHN0cmluZ3Muam9pbignJWMnKVxuICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nLmFwcGx5KGNvbnNvbGUsIHN0eWxlcylcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgY29uc29sZS5sb2coc3RyaW5ncy5qb2luKCcnKSlcbiAgICAgICAgICAgIH1cbiAgICAgICAgfSlcblxuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ0Vycm9yIGNvbXBpbGluZyAnICsgdHlwZU5hbWUgKyAnIHNoYWRlciwgJyArIGZpbGVzWzBdLm5hbWUpXG4gICAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gY2hlY2tGcmFtZWJ1ZmZlckVycm9yKGdsKXtcbiAgICBcbiAgICB2YXIgc3RhdHVzID0gZ2wuY2hlY2tGcmFtZWJ1ZmZlclN0YXR1cyhnbC5GUkFNRUJVRkZFUik7XG4gICAgaWYoc3RhdHVzICE9IGdsLkZSQU1FQlVGRkVSX0NPTVBMRVRFKXtcbiAgICAgICAgdmFyIHN0YXR1c0NvZGUgPSB7fVxuICAgICAgICBzdGF0dXNDb2RlW2dsLkZSQU1FQlVGRkVSX0NPTVBMRVRFXSA9ICdjb21wbGV0ZSdcbiAgICAgICAgc3RhdHVzQ29kZVtnbC5GUkFNRUJVRkZFUl9JTkNPTVBMRVRFX0FUVEFDSE1FTlRdID0gJ2luY29tcGxldGUgYXR0YWNobWVudCdcbiAgICAgICAgc3RhdHVzQ29kZVtnbC5GUkFNRUJVRkZFUl9JTkNPTVBMRVRFX0RJTUVOU0lPTlNdID0gJ2luY29tcGxldGUgZGltZW5zaW9ucydcbiAgICAgICAgc3RhdHVzQ29kZVtnbC5GUkFNRUJVRkZFUl9JTkNPTVBMRVRFX01JU1NJTkdfQVRUQUNITUVOVF0gPSAnaW5jb21wbGV0ZSwgbWlzc2luZyBhdHRhY2htZW50J1xuICAgICAgICBzdGF0dXNDb2RlW2dsLkZSQU1FQlVGRkVSX1VOU1VQUE9SVEVEXSA9ICd1bnN1cHBvcnRlZCdcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdmcmFtZWJ1ZmZlciBjb25maWd1cmF0aW9uIG5vdCBzdXBwb3J0ZWQsIHN0YXR1cyA9ICcgKyBzdGF0dXNDb2RlW3N0YXR1c10pXG4gICAgfVxufVxuXG5cbmZ1bmN0aW9uIGxlZnRQYWQgKHN0ciwgbikge1xuICAgIHN0ciA9IHN0ciArICcnXG4gICAgd2hpbGUgKHN0ci5sZW5ndGggPCBuKSB7XG4gICAgICAgIHN0ciA9ICcgJyArIHN0clxuICAgIH1cbiAgICByZXR1cm4gc3RyXG59XG5cbmZ1bmN0aW9uIFNoYWRlckZpbGUgKCkge1xuICAgIHRoaXMubmFtZSA9ICd1bmtub3duJ1xuICAgIHRoaXMubGluZXMgPSBbXVxuICAgIHRoaXMuaW5kZXggPSB7fVxuICAgIHRoaXMuaGFzRXJyb3JzID0gZmFsc2Vcbn1cblxuZnVuY3Rpb24gU2hhZGVyTGluZSAobnVtYmVyLCBsaW5lKSB7XG4gICAgdGhpcy5udW1iZXIgPSBudW1iZXJcbiAgICB0aGlzLmxpbmUgPSBsaW5lXG4gICAgdGhpcy5lcnJvcnMgPSBbXVxufVxuXG5mdW5jdGlvbiBTaGFkZXJFcnJvciAoZmlsZU51bWJlciwgbGluZU51bWJlciwgbWVzc2FnZSkge1xuICAgIHRoaXMuZmlsZSA9IGZpbGVOdW1iZXJcbiAgICB0aGlzLmxpbmUgPSBsaW5lTnVtYmVyXG4gICAgdGhpcy5tZXNzYWdlID0gbWVzc2FnZVxufVxuXG5mdW5jdGlvbiBwYXJzZVNvdXJjZSAoc291cmNlLCBjb21tYW5kKSB7XG4gICAgdmFyIGxpbmVzID0gc291cmNlLnNwbGl0KCdcXG4nKVxuICAgIHZhciBsaW5lTnVtYmVyID0gMVxuICAgIHZhciBmaWxlTnVtYmVyID0gMFxuICAgIHZhciBmaWxlcyA9IHtcbiAgICAgICAgdW5rbm93bjogbmV3IFNoYWRlckZpbGUoKSxcbiAgICAgICAgMDogbmV3IFNoYWRlckZpbGUoKVxuICAgIH1cbiAgICBmaWxlcy51bmtub3duLm5hbWUgPSBmaWxlc1swXS5uYW1lID0gJ3Vua25vd24nXG4gICAgZmlsZXMudW5rbm93bi5saW5lcy5wdXNoKG5ldyBTaGFkZXJMaW5lKDAsICcnKSlcbiAgICBmb3IgKHZhciBpID0gMDsgaSA8IGxpbmVzLmxlbmd0aDsgKytpKSB7XG4gICAgICAgIHZhciBsaW5lID0gbGluZXNbaV1cbiAgICAgICAgdmFyIHBhcnRzID0gL15cXHMqXFwjXFxzKihcXHcrKVxccysoLispXFxzKiQvLmV4ZWMobGluZSlcbiAgICAgICAgaWYgKHBhcnRzKSB7XG4gICAgICAgICAgICBzd2l0Y2ggKHBhcnRzWzFdKSB7XG4gICAgICAgICAgICAgICAgY2FzZSAnbGluZSc6XG4gICAgICAgICAgICAgICAgICAgIHZhciBsaW5lTnVtYmVySW5mbyA9IC8oXFxkKykoXFxzK1xcZCspPy8uZXhlYyhwYXJ0c1syXSlcbiAgICAgICAgICAgICAgICAgICAgaWYgKGxpbmVOdW1iZXJJbmZvKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBsaW5lTnVtYmVyID0gbGluZU51bWJlckluZm9bMV0gfCAwXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAobGluZU51bWJlckluZm9bMl0pIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBmaWxlTnVtYmVyID0gbGluZU51bWJlckluZm9bMl0gfCAwXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgaWYgKCEoZmlsZU51bWJlciBpbiBmaWxlcykpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgZmlsZXNbZmlsZU51bWJlcl0gPSBuZXcgU2hhZGVyRmlsZSgpXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIGJyZWFrXG4gICAgICAgICAgICAgICAgY2FzZSAnZGVmaW5lJzpcbiAgICAgICAgICAgICAgICAgICAgdmFyIG5hbWVJbmZvID0gL1NIQURFUl9OQU1FKF9CNjQpP1xccysoLiopJC8uZXhlYyhwYXJ0c1syXSlcbiAgICAgICAgICAgICAgICAgICAgaWYgKG5hbWVJbmZvKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICBmaWxlc1tmaWxlTnVtYmVyXS5uYW1lID0gKG5hbWVJbmZvWzFdXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgID8gZGVjb2RlQjY0KG5hbWVJbmZvWzJdKVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA6IG5hbWVJbmZvWzJdKVxuICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIGJyZWFrXG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgZmlsZXNbZmlsZU51bWJlcl0ubGluZXMucHVzaChuZXcgU2hhZGVyTGluZShsaW5lTnVtYmVyKyssIGxpbmUpKVxuICAgIH1cbiAgICBPYmplY3Qua2V5cyhmaWxlcykuZm9yRWFjaChmdW5jdGlvbiAoZmlsZU51bWJlcikge1xuICAgICAgICB2YXIgZmlsZSA9IGZpbGVzW2ZpbGVOdW1iZXJdXG4gICAgICAgIGZpbGUubGluZXMuZm9yRWFjaChmdW5jdGlvbiAobGluZSkge1xuICAgICAgICAgICAgZmlsZS5pbmRleFtsaW5lLm51bWJlcl0gPSBsaW5lXG4gICAgICAgIH0pXG4gICAgfSlcbiAgICByZXR1cm4gZmlsZXNcbn1cblxuZnVuY3Rpb24gcGFyc2VFcnJvckxvZyAoZXJyTG9nKSB7XG4gICAgdmFyIHJlc3VsdCA9IFtdXG4gICAgZXJyTG9nLnNwbGl0KCdcXG4nKS5mb3JFYWNoKGZ1bmN0aW9uIChlcnJNc2cpIHtcbiAgICAgICAgaWYgKGVyck1zZy5sZW5ndGggPCA1KSB7XG4gICAgICAgICAgICByZXR1cm5cbiAgICAgICAgfVxuICAgICAgICB2YXIgcGFydHMgPSAvXkVSUk9SXFw6XFxzKyhcXGQrKVxcOihcXGQrKVxcOlxccyooLiopJC8uZXhlYyhlcnJNc2cpXG4gICAgICAgIGlmIChwYXJ0cykge1xuICAgICAgICAgICAgcmVzdWx0LnB1c2gobmV3IFNoYWRlckVycm9yKFxuICAgICAgICAgICAgICAgIHBhcnRzWzFdIHwgMCxcbiAgICAgICAgICAgICAgICBwYXJ0c1syXSB8IDAsXG4gICAgICAgICAgICAgICAgcGFydHNbM10udHJpbSgpKSlcbiAgICAgICAgfSBlbHNlIGlmIChlcnJNc2cubGVuZ3RoID4gMCkge1xuICAgICAgICAgICAgcmVzdWx0LnB1c2gobmV3IFNoYWRlckVycm9yKCd1bmtub3duJywgMCwgZXJyTXNnKSlcbiAgICAgICAgfVxuICAgIH0pXG4gICAgcmV0dXJuIHJlc3VsdFxufVxuXG5mdW5jdGlvbiBhbm5vdGF0ZUZpbGVzIChmaWxlcywgZXJyb3JzKSB7XG4gICAgZXJyb3JzLmZvckVhY2goZnVuY3Rpb24gKGVycm9yKSB7XG4gICAgICAgIHZhciBmaWxlID0gZmlsZXNbZXJyb3IuZmlsZV1cbiAgICAgICAgaWYgKGZpbGUpIHtcbiAgICAgICAgICAgIHZhciBsaW5lID0gZmlsZS5pbmRleFtlcnJvci5saW5lXVxuICAgICAgICAgICAgaWYgKGxpbmUpIHtcbiAgICAgICAgICAgICAgICBsaW5lLmVycm9ycy5wdXNoKGVycm9yKVxuICAgICAgICAgICAgICAgIGZpbGUuaGFzRXJyb3JzID0gdHJ1ZVxuICAgICAgICAgICAgICAgIHJldHVyblxuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIGZpbGVzLnVua25vd24uaGFzRXJyb3JzID0gdHJ1ZVxuICAgICAgICBmaWxlcy51bmtub3duLmxpbmVzWzBdLmVycm9ycy5wdXNoKGVycm9yKVxuICAgIH0pXG59XG4iLCIvLyBpbXBvcnQgeyBUZW5zb3IsIE91dHB1dFRlbnNvciwgSW5QbGFjZVRlbnNvciB9IGZyb20gJy4uL3RlbnNvci9pbmRleC5qcydcbmltcG9ydCBCYXNlVGVuc29yIGZyb20gJy4uL3RlbnNvci9iYXNlLmpzJ1xuXG5pbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5cbmNvbnN0IFRFTlNPUl9GUkFHTUVOVF9IRUFERVIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy8uLi9mb3JtYXQvdXRpbC5nbHNsJywgJ3V0ZjgnKVxuXG5cbmV4cG9ydCBkZWZhdWx0IGZ1bmN0aW9uIGFzc2VtYmxlRnJhZ21lbnRTaGFkZXIoc2hhZGVyR2VuLCBvdXRwdXQsIHVuaWZvcm1zKXtcbiAgICB2YXIgdGVuc29yU2hhZGVyID0gc2hhZGVyR2VuKHVuaWZvcm1zLCBvdXRwdXQpO1xuICAgIFxuICAgIHZhciBmcmFnbWVudFNoYWRlciA9IFRFTlNPUl9GUkFHTUVOVF9IRUFERVI7XG4gICAgZm9yKGxldCB1bmlmb3JtIGluIHVuaWZvcm1zKXtcbiAgICAgICAgaWYodW5pZm9ybXNbdW5pZm9ybV0gaW5zdGFuY2VvZiBCYXNlVGVuc29yKXtcbiAgICAgICAgICAgIGxldCB0ZW5zb3IgPSB1bmlmb3Jtc1t1bmlmb3JtXTtcblxuICAgICAgICAgICAgZnJhZ21lbnRTaGFkZXIgKz0gdGVuc29yLl9mb3JtYXQuY29kZWMuZGVjb2RlU2hhZGVyLnJlcGxhY2UoL0AvZywgdW5pZm9ybSArICdfJykgKyAnXFxuJ1xuICAgICAgICAgICAgZnJhZ21lbnRTaGFkZXIgKz0gdGVuc29yLl9mb3JtYXQucGFjay5yZWFkU2hhZGVyLnJlcGxhY2UoL0AvZywgdW5pZm9ybSArICdfJykgKyAnXFxuJ1xuXG4gICAgICAgICAgICBpZigodGVuc29yLmZvcm1hdC5kZW5zaXR5ID09ICcxOjQnICYmIChuZXcgUmVnRXhwKHVuaWZvcm0gKyAnX3JlYWQ0XFxcXGInKSkudGVzdCh0ZW5zb3JTaGFkZXIpKSB8fCBcbiAgICAgICAgICAgICAgICAodGVuc29yLmZvcm1hdC5kZW5zaXR5ID09ICc0OjQnICYmIChuZXcgUmVnRXhwKHVuaWZvcm0gKyAnX3JlYWRcXFxcYicpKS50ZXN0KHRlbnNvclNoYWRlcikpKXtcbiAgICAgICAgICAgICAgICBmcmFnbWVudFNoYWRlciArPSB0ZW5zb3IuX2Zvcm1hdC5yZWFkX3NoaW0ucmVwbGFjZSgvQC9nLCB1bmlmb3JtICsgJ18nKSArICdcXG4nO1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgfVxuXG4gICAgdmFyIGFjdGl2YXRpb24gPSAodHlwZW9mIHVuaWZvcm1zLl9hY3RpdmF0aW9uID09ICdzdHJpbmcnICYmIHVuaWZvcm1zLl9hY3RpdmF0aW9uICE9ICdsaW5lYXInKSA/XG4gICAgICAgIHVuaWZvcm1zLl9hY3RpdmF0aW9uLnRvTG93ZXJDYXNlKCkgOiAnbGluZWFyJztcblxuICAgIGlmKCEoYWN0aXZhdGlvbiBpbiBvdXRwdXQuX2Zvcm1hdC5hY3RpdmF0aW9ucykpXG4gICAgICAgIHRocm93IG5ldyBFcnJvcignVW5rbm93biBhY3RpdmF0aW9uIHR5cGUgJyArIGFjdGl2YXRpb24pO1xuXG4gICAgZnJhZ21lbnRTaGFkZXIgKz0gb3V0cHV0Ll9mb3JtYXQuYWN0aXZhdGlvbnNbYWN0aXZhdGlvbl0ucmVwbGFjZSgvQC9nLCAnb3V0XycpICsgJ1xcbic7XG4gICAgZnJhZ21lbnRTaGFkZXIgKz0gb3V0cHV0Ll9mb3JtYXQuY29kZWMuZW5jb2RlU2hhZGVyLnJlcGxhY2UoL0AvZywgJ291dF8nKSArICdcXG4nO1xuICAgIGZyYWdtZW50U2hhZGVyICs9IG91dHB1dC5fZm9ybWF0LnBhY2sud3JpdGVTaGFkZXIucmVwbGFjZSgvQC9nLCAnb3V0XycpICsgJ1xcbic7XG5cblxuICAgIGlmKChvdXRwdXQuZm9ybWF0LmRlbnNpdHkgPT0gJzE6NCcgJiYgL3Byb2Nlc3M0XFxiLy50ZXN0KHRlbnNvclNoYWRlcikpIHx8IFxuICAgICAgICAob3V0cHV0LmZvcm1hdC5kZW5zaXR5ID09ICc0OjQnICYmIC9wcm9jZXNzXFxiLy50ZXN0KHRlbnNvclNoYWRlcikpKXtcbiAgICAgICAgZnJhZ21lbnRTaGFkZXIgKz0gb3V0cHV0Ll9mb3JtYXQud3JpdGVfc2hpbS5yZXBsYWNlKC9AL2csICdvdXRfJykgKyAnXFxuJztcbiAgICB9XG5cbiAgICBmcmFnbWVudFNoYWRlciArPSB0ZW5zb3JTaGFkZXIucmVwbGFjZSgvQC9nLCAnb3V0XycpXG5cbiAgICAvLyBjb25zb2xlLmxvZyhmcmFnbWVudFNoYWRlcilcblxuICAgIHJldHVybiBmcmFnbWVudFNoYWRlcjtcbn0iLCJpbXBvcnQgZ2V0VGVuc29yUHJvZ3JhbSBmcm9tICcuL3Byb2dyYW0uanMnXG5pbXBvcnQgYXNzZW1ibGVGcmFnbWVudFNoYWRlciBmcm9tICcuL2ZyYWcuanMnXG5pbXBvcnQgeyBUZW5zb3IsIE91dHB1dFRlbnNvciwgSW5QbGFjZVRlbnNvciB9IGZyb20gJy4uL3RlbnNvci9pbmRleC5qcydcbmltcG9ydCB7IGNoZWNrRnJhbWVidWZmZXJFcnJvciB9IGZyb20gJy4vY2hlY2suanMnXG5pbXBvcnQgVE5TTCBmcm9tICcuL3Ruc2wuanMnXG5pbXBvcnQgeyBiZWdpblRpbWVyLCBlbmRUaW1lciwgbm93IH0gZnJvbSAnLi90aW1lci5qcydcblxuXG5leHBvcnQgZnVuY3Rpb24gQ29tcGlsZShzaGFkZXJHZW4sIG91dHB1dCwgdW5pZm9ybXMgPSB7fSl7XG4gICAgdmFyIHN0YXJ0VGltZSA9IG5vdygpO1xuICAgIGlmKCEob3V0cHV0IGluc3RhbmNlb2YgT3V0cHV0VGVuc29yKSkgXG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcIkZpcnN0IGFyZ3VtZW50IG11c3QgYmUgYW4gaW5zdGFuY2Ugb2YgT3V0cHV0VGVuc29yXCIpO1xuICAgIFxuICAgIGlmKHR5cGVvZiBzaGFkZXJHZW4gPT09ICdzdHJpbmcnKSBzaGFkZXJHZW4gPSBUTlNMKHNoYWRlckdlbik7XG4gICAgXG4gICAgdmFyIGdsID0gb3V0cHV0LmdsO1xuICAgIHZhciBwcm9ncmFtID0gZ2V0VGVuc29yUHJvZ3JhbShnbCwgYXNzZW1ibGVGcmFnbWVudFNoYWRlcihzaGFkZXJHZW4sIG91dHB1dCwgdW5pZm9ybXMpKTtcbiAgICB2YXIgY29tcGlsZVRpbWUgPSBub3coKSAtIHN0YXJ0VGltZTtcbiAgICAvLyBjb25zb2xlLmxvZygnQ29tcGlsZSBUaW1lJywgY29tcGlsZVRpbWUpXG4gICAgcmV0dXJuIHByb2dyYW07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBSdW4oc2hhZGVyR2VuLCBvdXRwdXQsIHVuaWZvcm1zID0ge30sIGNhbGxiYWNrID0gbnVsbCl7XG4gICAgdmFyIHRwID0gQ29tcGlsZShzaGFkZXJHZW4sIG91dHB1dCwgdW5pZm9ybXMpO1xuXG4gICAgdmFyIGdsID0gb3V0cHV0LmdsO1xuICAgIFxuICAgIGlmKGNhbGxiYWNrICYmIHR5cGVvZiBjYWxsYmFjayAhPSAnZnVuY3Rpb24nKSB0aHJvdyBuZXcgRXJyb3IoJ0NhbGxiYWNrIG11c3QgYmUgYSBmdW5jdGlvbicpO1xuICAgIGlmKGNhbGxiYWNrKXtcbiAgICAgICAgYmVnaW5UaW1lcihnbCwge1xuICAgICAgICAgICAgc2hhZGVyOiBzaGFkZXJHZW4sXG4gICAgICAgICAgICBvdXRwdXQ6IG91dHB1dFxuICAgICAgICB9KVxuICAgIH1cblxuICAgIGdsLnVzZVByb2dyYW0odHAucHJvZ3JhbSk7XG4gICAgZ2wuZGlzYWJsZShnbC5ERVBUSF9URVNUKTtcbiAgICBnbC5kaXNhYmxlKGdsLkJMRU5EKTtcblxuICAgIHZhciBzZXRVbmlmb3JtID0gdHAuc2V0VW5pZm9ybSxcbiAgICAgICAgdGV4SW5kZXggPSAwLFxuICAgICAgICBtdXN0U3dhcCA9IGZhbHNlO1xuICAgICAgICBcbiAgICBmb3IobGV0IG5hbWUgaW4gdW5pZm9ybXMpe1xuICAgICAgICBpZihuYW1lLnN0YXJ0c1dpdGgoJ18nKSkgY29udGludWU7XG4gICAgICAgIFxuICAgICAgICBpZigobmFtZSArICdfdGV4JykgaW4gdHAudW5pZm9ybVR5cGVzKXtcbiAgICAgICAgICAgIGxldCB0ZW5zb3IgPSB1bmlmb3Jtc1tuYW1lXTtcbiAgICAgICAgICAgIGlmKHRlbnNvci5nbCAhPT0gb3V0cHV0LmdsKSB0aHJvdyBuZXcgRXJyb3IoJ1VuaWZvcm1zIG11c3QgYmVsb25nIHRvIHNhbWUgR0wgY29udGV4dCBhcyBvdXRwdXQnKTtcbiAgICAgICAgICAgIGlmKHRlbnNvciA9PT0gb3V0cHV0KSBtdXN0U3dhcCA9IHRydWU7XG5cbiAgICAgICAgICAgIGZvcihsZXQgdW5pZm9ybSBpbiB0ZW5zb3IuaW5mbyl7XG4gICAgICAgICAgICAgICAgc2V0VW5pZm9ybShuYW1lICsgJ18nICsgdW5pZm9ybSwgdGVuc29yLmluZm9bdW5pZm9ybV0pXG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGdsLmFjdGl2ZVRleHR1cmUoZ2xbJ1RFWFRVUkUnICsgdGV4SW5kZXhdKTtcbiAgICAgICAgICAgIGdsLmJpbmRUZXh0dXJlKGdsLlRFWFRVUkVfMkQsIHRlbnNvci50ZXgpO1xuICAgICAgICAgICAgc2V0VW5pZm9ybShuYW1lICsgJ190ZXgnLCB0ZXhJbmRleCk7XG5cbiAgICAgICAgICAgIHRleEluZGV4KytcbiAgICAgICAgfWVsc2UgaWYobmFtZSBpbiB0cC51bmlmb3JtVHlwZXMpe1xuICAgICAgICAgICAgc2V0VW5pZm9ybShuYW1lLCB1bmlmb3Jtc1tuYW1lXSlcbiAgICAgICAgfWVsc2V7XG4gICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXCJVbmtub3duIHVuaWZvcm0gXCIgKyBuYW1lKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIC8vIE9yZGluYXJpbHkgd2UgY2FuJ3Qgd3JpdGUgdG8gdGhlIHNhbWUgdGV4dHVyZSB0aGF0IHdlJ3JlIHVzaW5nIGFzXG4gICAgLy8gYW4gaW5wdXQsIGFzIHRoaXMgY291bGQgbGVhZCB0byBhbGwgc29ydHMgb2YgdGVycmlibGUgcmFjZSBjb25kaXRpb25zLFxuICAgIC8vIHVuZGVmaW5lZCBiZWhhdmlvciwgYW5kIGludmFsaWQgc3RhdGUuIEluUGxhY2VUZW5zb3JzIGFjdHVhbGx5IGNvbnNpc3RcbiAgICAvLyBvZiBhIHBhaXIgb2YgdGV4dHVyZXMgd2hpY2ggYXJlIHN3YXBwZWQgZm9yIHRoZXNlIGluLXBsYWNlIG9wZXJhdGlvbnMuIFxuICAgIGlmKG11c3RTd2FwKSBvdXRwdXQuc3dhcCgpO1xuXG4gICAgZm9yKGxldCB1bmlmb3JtIGluIG91dHB1dC5pbmZvKXtcbiAgICAgICAgc2V0VW5pZm9ybSgnb3V0XycgKyB1bmlmb3JtLCBvdXRwdXQuaW5mb1t1bmlmb3JtXSlcbiAgICB9XG5cbiAgICBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIG91dHB1dC5mYm8pO1xuICAgIGdsLnZpZXdwb3J0KDAsIDAsIG91dHB1dC5pbmZvLnRleFNpemVbMF0sIG91dHB1dC5pbmZvLnRleFNpemVbMV0pO1xuICAgIGdsLmRyYXdBcnJheXMoZ2wuVFJJQU5HTEVfU1RSSVAsIDAsIDQpOyAvLyBkcmF3IHRvIGZyYW1lYnVmZmVyXG5cbiAgICBjaGVja0ZyYW1lYnVmZmVyRXJyb3IoZ2wpO1xuICAgIFxuICAgIC8vIHZhciBydW5UaW1lID0gbm93KCkgLSBzdGFydFRpbWU7XG4gICAgLy8gdGltZXIuZW5kKClcbiAgICBpZihjYWxsYmFjayl7XG4gICAgICAgIGVuZFRpbWVyKGdsLCBmdW5jdGlvbihpbmZvKXtcbiAgICAgICAgICAgIC8vIGNvbnNvbGUubG9nKCdHUFUgdGltZTogJywgaW5mbylcbiAgICAgICAgICAgIGNhbGxiYWNrKGluZm8pO1xuICAgICAgICB9KSAgICBcbiAgICB9XG4gICAgLy8gY29uc29sZS5sb2coJ0NQVSBSdW4gVGltZScsIHJ1blRpbWUpXG5cbiAgICByZXR1cm4gb3V0cHV0O1xufSIsImltcG9ydCB7IGNoZWNrTGlua0Vycm9yLCBjaGVja1NoYWRlckVycm9yIH0gZnJvbSAnLi9jaGVjay5qcydcblxuY29uc3QgVEVOU09SX1ZFUlRFWF9TSEFERVIgPSBgXG4gICAgcHJlY2lzaW9uIGhpZ2hwIGZsb2F0O1xuICAgIGF0dHJpYnV0ZSB2ZWMyIGFfcG9zaXRpb247XG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgICBnbF9Qb3NpdGlvbiA9IHZlYzQoYV9wb3NpdGlvbiwgMCwgMSk7XG4gICAgfVxuYFxuXG5cbmNvbnN0IFVOSUZPUk1fU0VUVEVSUyA9IHsgdmVjNDogJzRmdicsIHZlYzM6ICczZnYnLCB2ZWMyOiAnMmZ2JywgZmxvYXQ6ICcxZicsXG4gICAgICAgICAgICAgICAgICAgICAgICAgIGl2ZWM0OiAnNGl2JywgaXZlYzM6ICczaXYnLCBpdmVjMjogJzJpdicsIGludDogJzFpJyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgc2FtcGxlcjJEOiAnMWknIH07XG5cbmV4cG9ydCBkZWZhdWx0IGZ1bmN0aW9uIGdldFRlbnNvclByb2dyYW0oZ2wsIGZyYWdtZW50U2hhZGVyKXtcbiAgICBpZighZ2wuX3RlbnNvclByb2dyYW1zKSBnbC5fdGVuc29yUHJvZ3JhbXMgPSB7fTtcbiAgICBpZihmcmFnbWVudFNoYWRlciBpbiBnbC5fdGVuc29yUHJvZ3JhbXMpe1xuICAgICAgICByZXR1cm4gZ2wuX3RlbnNvclByb2dyYW1zW2ZyYWdtZW50U2hhZGVyXVxuICAgIH1cbiAgICB2YXIgcHJvZ3JhbSA9IGNyZWF0ZVRlbnNvclByb2dyYW0oZ2wsIGZyYWdtZW50U2hhZGVyKTtcbiAgICBnbC5fdGVuc29yUHJvZ3JhbXNbZnJhZ21lbnRTaGFkZXJdID0gcHJvZ3JhbTtcbiAgICByZXR1cm4gcHJvZ3JhbTtcbn1cblxuZnVuY3Rpb24gY3JlYXRlVGVuc29yUHJvZ3JhbShnbCwgZnJhZ21lbnRTaGFkZXIpe1xuICAgIHZhciBwcm9ncmFtID0gY3JlYXRlU2hhZGVyUHJvZ3JhbShnbCwgVEVOU09SX1ZFUlRFWF9TSEFERVIsIGZyYWdtZW50U2hhZGVyKTtcbiAgICBcbiAgICBnbC51c2VQcm9ncmFtKHByb2dyYW0pO1xuICAgIGJpbmRBdHRyaWJ1dGVCdWZmZXIoZ2wsIHByb2dyYW0pO1xuXG4gICAgdmFyIHVuaWZvcm1UeXBlcyA9IGV4dHJhY3RVbmlmb3JtRGVjbGFyYXRpb25zKGZyYWdtZW50U2hhZGVyKSxcbiAgICAgICAgdW5pZm9ybUxvY3MgPSB7fTtcblxuICAgIGZ1bmN0aW9uIGFkZFVuaWZvcm0obmFtZSwgdHlwZSl7XG4gICAgICAgIHVuaWZvcm1Mb2NzW25hbWVdID0geyBsb2M6IGdsLmdldFVuaWZvcm1Mb2NhdGlvbihwcm9ncmFtLCBuYW1lKSwgdHlwZTogdHlwZSB9XG4gICAgfVxuXG4gICAgZm9yKGxldCBuYW1lIGluIHVuaWZvcm1UeXBlcyl7XG4gICAgICAgIGxldCB0eXBlID0gdW5pZm9ybVR5cGVzW25hbWVdO1xuICAgICAgICBpZigodHlwZSkgaW4gVU5JRk9STV9TRVRURVJTKXtcbiAgICAgICAgICAgIGFkZFVuaWZvcm0obmFtZSwgdHlwZSk7XG4gICAgICAgIH1lbHNlIHRocm93IG5ldyBFcnJvcihcIlVua25vd24gdW5pZm9ybSB0eXBlIFwiICsgdHlwZSk7XG4gICAgfVxuXG4gICAgZnVuY3Rpb24gc2V0VW5pZm9ybShuYW1lLCB2YWx1ZSl7XG4gICAgICAgIGlmKCEobmFtZSBpbiB1bmlmb3JtTG9jcykpe1xuICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKFwiQ291bGQgbm90IGZpbmQgdW5pZm9ybSBcIiArIG5hbWUpO1xuICAgICAgICB9XG4gICAgICAgIGdsWyd1bmlmb3JtJyArIFVOSUZPUk1fU0VUVEVSU1t1bmlmb3JtTG9jc1tuYW1lXS50eXBlXV0odW5pZm9ybUxvY3NbbmFtZV0ubG9jLCB2YWx1ZSlcbiAgICB9XG5cbiAgICByZXR1cm4ge1xuICAgICAgICBwcm9ncmFtOiBwcm9ncmFtLFxuICAgICAgICB1bmlmb3JtTG9jczogdW5pZm9ybUxvY3MsXG4gICAgICAgIHVuaWZvcm1UeXBlczogdW5pZm9ybVR5cGVzLFxuICAgICAgICBzZXRVbmlmb3JtOiBzZXRVbmlmb3JtLFxuICAgIH1cbn1cblxuXG5leHBvcnQgZnVuY3Rpb24gYmluZEF0dHJpYnV0ZUJ1ZmZlcihnbCwgcHJvZ3JhbSkge1xuICAgIGdsLmJpbmRCdWZmZXIoZ2wuQVJSQVlfQlVGRkVSLCBnbC5jcmVhdGVCdWZmZXIoKSk7XG4gICAgZ2wuYnVmZmVyRGF0YShnbC5BUlJBWV9CVUZGRVIsIG5ldyBGbG9hdDMyQXJyYXkoWyAtMSwtMSwgMSwtMSwgLTEsIDEsIDEsIDFdKSwgZ2wuU1RBVElDX0RSQVcpO1xuXG4gICAgdmFyIHBvc2l0aW9uTG9jYXRpb24gPSBnbC5nZXRBdHRyaWJMb2NhdGlvbihwcm9ncmFtLCBcImFfcG9zaXRpb25cIik7XG4gICAgZ2wuZW5hYmxlVmVydGV4QXR0cmliQXJyYXkocG9zaXRpb25Mb2NhdGlvbik7XG4gICAgZ2wudmVydGV4QXR0cmliUG9pbnRlcihwb3NpdGlvbkxvY2F0aW9uLCAyLCBnbC5GTE9BVCwgZmFsc2UsIDAsIDApO1xufVxuXG5cbmZ1bmN0aW9uIGV4dHJhY3RVbmlmb3JtRGVjbGFyYXRpb25zKHN0cil7XG4gICAgdmFyIHVuaWZvcm1zID0ge307XG4gICAgc3RyID0gc3RyLnJlcGxhY2UoLygoPzpcXC9cXCooPzpbXipdfCg/OlxcKitbXipcXC9dKSkqXFwqK1xcLyl8KD86XFwvXFwvLiopKS9nLCAnJylcbiAgICBzdHIgPSBzdHIucmVwbGFjZSgvXFwvXFwvLipcXG4vZywgJycpXG4gICAgdmFyIG0sIHJlID0gL3VuaWZvcm1cXHMqKFtcXHdfXSspXFxzKihbXFx3X10rKS9nO1xuICAgIHdoaWxlIChtID0gcmUuZXhlYyhzdHIpKSB1bmlmb3Jtc1ttWzJdXSA9IG1bMV07XG4gICAgcmV0dXJuIHVuaWZvcm1zO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlU2hhZGVyUHJvZ3JhbShnbCwgdmVydGV4U291cmNlLCBmcmFnbWVudFNvdXJjZSkge1xuICAgIHZhciB2ZXJ0ZXhTaGFkZXIgPSBjb21waWxlU2hhZGVyKGdsLCB2ZXJ0ZXhTb3VyY2UsIGdsLlZFUlRFWF9TSEFERVIpO1xuICAgIHZhciBmcmFnbWVudFNoYWRlciA9IGNvbXBpbGVTaGFkZXIoZ2wsIGZyYWdtZW50U291cmNlLCBnbC5GUkFHTUVOVF9TSEFERVIpO1xuXG4gICAgLy8gdmFyIGRlYnVnID0gZ2wuZ2V0RXh0ZW5zaW9uKCdXRUJHTF9kZWJ1Z19zaGFkZXJzJylcbiAgICAvLyBjb25zb2xlLmxvZyhkZWJ1Zy5nZXRUcmFuc2xhdGVkU2hhZGVyU291cmNlKHZlcnRleFNoYWRlcikpO1xuICAgIC8vIGNvbnNvbGUubG9nKGRlYnVnLmdldFRyYW5zbGF0ZWRTaGFkZXJTb3VyY2UoZnJhZ21lbnRTaGFkZXIpKTtcblxuICAgIHZhciBwcm9ncmFtID0gZ2wuY3JlYXRlUHJvZ3JhbSgpO1xuICAgIGdsLmF0dGFjaFNoYWRlcihwcm9ncmFtLCB2ZXJ0ZXhTaGFkZXIpO1xuICAgIGdsLmF0dGFjaFNoYWRlcihwcm9ncmFtLCBmcmFnbWVudFNoYWRlcik7XG4gICAgZ2wubGlua1Byb2dyYW0ocHJvZ3JhbSk7XG5cbiAgICAvLyBpbnRlcmVzdGluZ2x5IGVub3VnaCBpdCBzZWVtcyBsaWtlIFNhZmFyaSBuZXZlciBlbWl0c1xuICAgIC8vIGEgc2hhZGVyIHByb2dyYW0gbGluayBlcnJvci4gXG4gICAgY2hlY2tMaW5rRXJyb3IoZ2wsIHByb2dyYW0sIGZyYWdtZW50U291cmNlLCB2ZXJ0ZXhTb3VyY2UpO1xuXG4gICAgcmV0dXJuIHByb2dyYW07XG59XG5cblxuZnVuY3Rpb24gY29tcGlsZVNoYWRlcihnbCwgc2hhZGVyU291cmNlLCBzaGFkZXJUeXBlKSB7XG4gICAgdmFyIHNoYWRlciA9IGdsLmNyZWF0ZVNoYWRlcihzaGFkZXJUeXBlKTtcbiAgICBnbC5zaGFkZXJTb3VyY2Uoc2hhZGVyLCBzaGFkZXJTb3VyY2UpO1xuICAgIGdsLmNvbXBpbGVTaGFkZXIoc2hhZGVyKTtcbiAgICB2YXIgc3VjY2VzcyA9IGdsLmdldFNoYWRlclBhcmFtZXRlcihzaGFkZXIsIGdsLkNPTVBJTEVfU1RBVFVTKTtcbiAgICBjaGVja1NoYWRlckVycm9yKGdsLCBzaGFkZXIsIHNoYWRlclNvdXJjZSwgc2hhZGVyVHlwZSlcbiAgICByZXR1cm4gc2hhZGVyO1xufVxuXG5cbiIsImV4cG9ydCBmdW5jdGlvbiBub3coKSB7XG4gICAgaWYgKHR5cGVvZiBwZXJmb3JtYW5jZSA9PT0gJ3VuZGVmaW5lZCcpIHtcbiAgICAgICAgcmV0dXJuIERhdGUubm93KClcbiAgICB9IGVsc2Uge1xuICAgICAgICByZXR1cm4gcGVyZm9ybWFuY2Uubm93KCk7XG4gICAgfVxufVxuXG5mdW5jdGlvbiBnZXRUaW1lcihnbCl7XG5cdGlmKGdsLk5PX1BST0ZJTEUpIHJldHVybjtcblx0aWYodHlwZW9mIGdsLlRJTUVSX1BPT0wgPT09ICd1bmRlZmluZWQnKXtcblx0XHR2YXIgZXh0VGltZXIgPSBnbC5nZXRFeHRlbnNpb24oJ2V4dF9kaXNqb2ludF90aW1lcl9xdWVyeScpO1xuXHRcdGlmKCFleHRUaW1lciB8fCAhZXh0VGltZXIuY3JlYXRlUXVlcnlFWFQpe1xuXHRcdFx0Z2wuTk9fUFJPRklMRSA9IHRydWU7XG5cdFx0XHRyZXR1cm47XG5cdFx0fVxuXHRcdGdsLlRJTUVSX1BPT0wgPSBjcmVhdGVUaW1lcihnbClcblx0fVxuXHRyZXR1cm4gZ2wuVElNRVJfUE9PTDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGJlZ2luVGltZXIoZ2wsIGluZm89e30pe1xuXHR2YXIgdGltZXIgPSBnZXRUaW1lcihnbCk7XG5cdGlmKHRpbWVyKXtcblx0XHR0aW1lci5iZWdpbihpbmZvKVxuXHR9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBlbmRUaW1lcihnbCwgY2FsbGJhY2spe1xuXHR2YXIgdGltZXIgPSBnZXRUaW1lcihnbCk7XG5cdGlmKHRpbWVyKXtcblx0XHR0aW1lci5lbmQoY2FsbGJhY2spXG5cdH1lbHNlIGlmKGNhbGxiYWNrKXtcblx0XHRjb25zb2xlLndhcm4oXCJDYW4gbm90IHRyaWdnZXIgY2FsbGJhY2s6IGltcGxlbWVudGF0aW9uIGRvZXMgbm90IHN1cHBvcnQgZXh0X2Rpc2pvaW50X3RpbWVyX3F1ZXJ5XCIpXG5cdH1cbn1cblxuZnVuY3Rpb24gY3JlYXRlVGltZXIoZ2wpe1x0XG5cdHZhciBleHRUaW1lciA9IGdsLmdldEV4dGVuc2lvbignZXh0X2Rpc2pvaW50X3RpbWVyX3F1ZXJ5Jyk7XG5cblx0dmFyIHF1ZXJ5UG9vbCA9IFtdXG4gICAgZnVuY3Rpb24gYWxsb2NRdWVyeSAoKSB7XG4gICAgICAgIHJldHVybiBxdWVyeVBvb2wucG9wKCkgfHwgZXh0VGltZXIuY3JlYXRlUXVlcnlFWFQoKVxuICAgIH1cbiAgICBmdW5jdGlvbiBmcmVlUXVlcnkgKHF1ZXJ5KSB7XG4gICAgICAgIHF1ZXJ5UG9vbC5wdXNoKHF1ZXJ5KVxuICAgIH1cblxuXHR2YXIgcGVuZGluZ1F1ZXJpZXMgPSBbXVxuXHRmdW5jdGlvbiBiZWdpblF1ZXJ5IChpbmZvKSB7XG5cdFx0dmFyIHF1ZXJ5ID0gYWxsb2NRdWVyeSgpXG5cdFx0ZXh0VGltZXIuYmVnaW5RdWVyeUVYVChleHRUaW1lci5USU1FX0VMQVBTRURfRVhULCBxdWVyeSlcblx0XHRwZW5kaW5nUXVlcmllcy5wdXNoKFtxdWVyeSwgaW5mb10pXG5cdH1cblxuXHRmdW5jdGlvbiBlbmRRdWVyeSAoKSB7XG5cdFx0ZXh0VGltZXIuZW5kUXVlcnlFWFQoZXh0VGltZXIuVElNRV9FTEFQU0VEX0VYVClcblx0fVxuXG5cdGZ1bmN0aW9uIGNhbGxiYWNrKGluZm8sIHRpbWUpe1xuXHRcdHZhciBmbiA9IGluZm8uY2FsbGJhY2s7XG5cdFx0aW5mby5ncHVUaW1lID0gdGltZTtcblx0XHRkZWxldGUgaW5mby5jYWxsYmFjaztcblx0XHRpZihmbikgZm4oaW5mbyk7XG5cdH1cblxuXHRmdW5jdGlvbiBtb25pdG9yUGVuZGluZygpe1xuXHRcdGZvciAodmFyIGkgPSAwOyBpIDwgcGVuZGluZ1F1ZXJpZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIFx0XHR2YXIgcXVlcnkgPSBwZW5kaW5nUXVlcmllc1tpXVswXVxuICAgICAgXHRcdGlmIChleHRUaW1lci5nZXRRdWVyeU9iamVjdEVYVChxdWVyeSwgZXh0VGltZXIuUVVFUllfUkVTVUxUX0FWQUlMQUJMRV9FWFQpKSB7XG4gICAgICAgIFx0XHR2YXIgcXVlcnlUaW1lID0gZXh0VGltZXIuZ2V0UXVlcnlPYmplY3RFWFQocXVlcnksIGV4dFRpbWVyLlFVRVJZX1JFU1VMVF9FWFQpXG4gICAgICAgIFx0XHRjYWxsYmFjayhwZW5kaW5nUXVlcmllc1tpXVsxXSwgcXVlcnlUaW1lIC8gMWU2KVxuICAgICAgICBcdFx0ZnJlZVF1ZXJ5KHF1ZXJ5KVxuICAgICAgICBcdFx0cGVuZGluZ1F1ZXJpZXMuc3BsaWNlKGksIDEpXG4gICAgICAgIFx0XHRpLS1cbiAgICAgIFx0XHR9XG5cdCAgICB9XG5cdH1cblxuXG5cdHZhciBpc1BvbGxpbmcgPSBmYWxzZTtcblx0ZnVuY3Rpb24gbG9vcCgpe1xuXHRcdGlmKHBlbmRpbmdRdWVyaWVzLmxlbmd0aCA+IDApe1xuXHRcdFx0bW9uaXRvclBlbmRpbmcoKVxuXHRcdFx0cmVxdWVzdEFuaW1hdGlvbkZyYW1lKGxvb3ApXG5cdFx0fWVsc2V7XG5cdFx0XHRpc1BvbGxpbmcgPSBmYWxzZTtcblx0XHR9XG5cdH1cblxuXHR2YXIgY3VycmVudEluZm8gPSBudWxsO1xuICAgIHJldHVybiB7XG4gICAgXHRiZWdpbihpbmZvID0ge30pe1xuICAgIFx0XHRpZihjdXJyZW50SW5mbykgdGhyb3cgbmV3IEVycm9yKCdiZWdpblRpbWVyIHdhcyBjYWxsZWQgYmVmb3JlIHByZXZpb3VzIGVuZFRpbWVyJyk7XG4gICAgXHRcdGN1cnJlbnRJbmZvID0gaW5mb1xuICAgIFx0XHRpbmZvLmNwdVN0YXJ0VGltZSA9IG5vdygpO1xuICAgIFx0XHRiZWdpblF1ZXJ5KGN1cnJlbnRJbmZvKVxuICAgIFx0fSxcblxuICAgIFx0ZW5kKGZuKXtcbiAgICBcdFx0Y3VycmVudEluZm8uY3B1VGltZSA9IG5vdygpIC0gY3VycmVudEluZm8uY3B1U3RhcnRUaW1lXG4gICAgXHRcdGRlbGV0ZSBjdXJyZW50SW5mby5jcHVTdGFydFRpbWU7XG4gICAgXHRcdGN1cnJlbnRJbmZvLmNhbGxiYWNrID0gZm47XG4gICAgXHRcdGN1cnJlbnRJbmZvID0gbnVsbDtcbiAgICBcdFx0ZW5kUXVlcnkoKVxuXG4gICAgXHRcdGlmKGlzUG9sbGluZyA9PT0gZmFsc2Upe1xuICAgIFx0XHRcdGlzUG9sbGluZyA9IHRydWU7XG4gICAgXHRcdFx0cmVxdWVzdEFuaW1hdGlvbkZyYW1lKGxvb3ApXG4gICAgXHRcdH1cbiAgICBcdH1cbiAgICB9XG59IiwiLy8gVE5TTCAocHJvbm91bmNlZCB0aW5zZWwpXG4vLyBpcyBhIGRvbWFpbiBzcGVjaWZpYyBsYW5ndWFnZSBiYXNlZCBvbiBHTFNMXG4vLyBmb3IgaGVscGluZyB3aXRoIHRoZSB3cml0aW5nIGNvZGUgdGhhdFxuLy8gY29tcHV0ZXMgd2l0aCB0ZW5zb3JzLiBcblxuLy8gQSBsaW1pdGF0aW9uIG9mIEdMU0wgaXMgdGhhdCB0aGUgY29uZGl0aW9uXG4vLyBvZiBhbnkgbG9vcCBoYXMgdG8gYmUgc3RhdGljYWxseSBrbm93biBcbi8vIChlLmcuIGNvdW50ZXJzIHVwIHRvIGEgZml4ZWQgY29uc3RhbnRcbi8vIHZhbHVlKSB3aGljaCBpcyBwcm9ibGVtYXRpYyBpZiB3ZSB3YW50XG4vLyB0byB3cml0ZSBnZW5lcmFsIGNvZGUgdGhhdCBkZXBlbmRzIG9uXG4vLyB0aGUgc2l6ZSBvZiB0aGUgaW5wdXQgdGVuc29yc1xuXG4vLyBUTlNMIGFkZHMgdGhlIGZvbGxvd2luZyBzeW50YXg6XG4vLyAgICAgICMoaW1hZ2Uuc2hhcGUpXG4vLyB3aGljaCB3aWxsIGJlIHJlcGxhY2VkIHdpdGggYW4gaXZlYzRcbi8vIGNvbnRhaW5pbmcgdGhlIHNoYXBlIG9mIHRoZSBpbnB1dCB0ZW5zb3IgXCJpbWFnZVwiXG4vLyBhdXRvbWF0aWNhbGx5XG5cbmV4cG9ydCBkZWZhdWx0IGZ1bmN0aW9uIFROU0woc3RyKXtcbiAgICBpZih0eXBlb2Ygc3RyICE9ICdzdHJpbmcnKSBcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdUTlNMIHNoYWRlciBwcmVwcm9jZXNzb3Igb25seSBhY2NlcHRzIHN0cmluZ3MnKTtcbiAgICBcbiAgICByZXR1cm4gZnVuY3Rpb24odW5pZm9ybXMsIG91dHB1dCl7XG4gICAgICAgIHJldHVybiBzdHJcbiAgICAgICAgLy8gY29tbWVudCBvdXQgdGhlIHRlbnNvciBzdHJ1Y3QgZGVmaW5pdGlvbnNcbiAgICAgICAgLnJlcGxhY2UoL3VuaWZvcm1cXHMqVGVuc29yXFxzKihbXFx3X10rKVxccyo7L2csICcvKiAoVGVuc29yICQxKSAqLycpXG5cbiAgICAgICAgLy8gdGhpcyBpcyB0aGUgbWFjcm8gc3ludGF4XG4gICAgICAgIC5yZXBsYWNlKC9cXCNcXCgoW1xcd1xcLlxcc10rKVxcKS9nLCBmdW5jdGlvbihhbGwsIGJvZHkpe1xuICAgICAgICAgICAgdmFyIG9iaiA9IHVuaWZvcm1zO1xuICAgICAgICAgICAgZm9yKGxldCBwYXJ0IG9mIGJvZHkuc3BsaXQoJy4nKSlcbiAgICAgICAgICAgICAgICBvYmogPSBvYmpbcGFydC50cmltKCldO1xuICAgICAgICAgICAgaWYodHlwZW9mIG9iaiA9PSAnbnVtYmVyJyl7XG4gICAgICAgICAgICAgICAgcmV0dXJuIG9iai50b1N0cmluZygpXG4gICAgICAgICAgICB9ZWxzZSBpZihBcnJheS5pc0FycmF5KG9iaikgJiYgb2JqLmxlbmd0aCA8PSA0ICYmIG9iai5sZW5ndGggPiAxKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gKG9iai5ldmVyeShOdW1iZXIuaXNJbnRlZ2VyKSA/ICdpJyA6ICcnKSArIFxuICAgICAgICAgICAgICAgICAgICAndmVjJyArIG9iai5sZW5ndGggKyAnKCcgKyBvYmouam9pbignLCcpICsgJyknXG4gICAgICAgICAgICB9XG4gICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ0NhbiBub3QgaW5saW5lIGV4cHJlc3Npb24gJyArIGJvZHkpO1xuICAgICAgICB9KVxuICAgICAgICAvLyB0ZW5zb3IucmVhZDQoeCwgMCkgPT4gdGVuc29yLnJlYWQ0KGl2ZWM0KHgsIDAsIDAsIDApKVxuICAgICAgICAvLyB0aGlzIHRyYW5zZm9ybWF0aW9uIHRha2VzIHBsYWNlIHdoZW4gdGhlcmUgYXJlIDIgb3IgbW9yZSBhcmd1bWVudHNcbiAgICAgICAgLy8gYXMgb3RoZXJ3aXNlIGl0J3Mgbm90IHBvc3NpYmxlIHRvIHN0YXRpY2FsbHkgZGV0ZXJtaW5lIHdoZXRoZXIgeCBpc1xuICAgICAgICAvLyBvZiB0eXBlIGl2ZWM0IG9yIGEgbnVtYmVyXG4gICAgICAgIC5yZXBsYWNlKC9cXGIoXFx3KylcXHMqXFwuXFxzKihyZWFkND8pXFxiXFxzKlxcKChbXlxcKFxcKV0rKVxcKS9nLCBmdW5jdGlvbihhbGwsIG5hbWUsIHByb3AsIGFyZyl7XG4gICAgICAgICAgICBpZihuYW1lIGluIHVuaWZvcm1zICYmIHVuaWZvcm1zW25hbWVdLnNoYXBlKXtcbiAgICAgICAgICAgICAgICB2YXIgcGFydHMgPSBhcmcuc3BsaXQoJywnKSxcbiAgICAgICAgICAgICAgICAgICAgcGFkZGVkID0gcGFydHMuY29uY2F0KFsnMCcsICcwJywgJzAnLCAnMCddLnNsaWNlKDAsIDQgLSBwYXJ0cy5sZW5ndGgpKTtcbiAgICAgICAgICAgICAgICBpZihwYXJ0cy5sZW5ndGggPCAyIHx8IHBhcnRzLmxlbmd0aCA+IDQpIHJldHVybiBhbGw7XG4gICAgICAgICAgICAgICAgdmFyIHZlYyA9ICdpdmVjNCgnICsgcGFkZGVkLmpvaW4oJywnKSArICcpJztcbiAgICAgICAgICAgICAgICByZXR1cm4gbmFtZSArICdfJyArIHByb3AgKyAnKCcgKyB2ZWMgKyAnKSc7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXR1cm4gYWxsO1xuICAgICAgICB9KVxuXG4gICAgICAgIC8vIHRlbnNvci5zaGFwZSA9PiB0ZW5zb3Jfc2hhcGVcbiAgICAgICAgLnJlcGxhY2UoL1xcYihcXHcrKVxccypcXC5cXHMqKFxcdyspXFxiL2csIGZ1bmN0aW9uKGFsbCwgbmFtZSwgcHJvcCl7XG4gICAgICAgICAgICBpZihuYW1lIGluIHVuaWZvcm1zICYmIHVuaWZvcm1zW25hbWVdLnNoYXBlKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gbmFtZSArICdfJyArIHByb3A7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICByZXR1cm4gYWxsO1xuICAgICAgICB9KVxuICAgICAgICAvLyAucmVwbGFjZSgvXFwjXFxzKihcXHcrKVxccypcXFsoLio/KVxcXS9nLCBmdW5jdGlvbihhbGwsIHRlbnNvciwgYm9keSl7XG4gICAgICAgIC8vICAgICByZXR1cm4gdGVuc29yICsgJ19yZWFkKGl2ZWM0KCcgKyBib2R5ICsgJykpJ1xuICAgICAgICAvLyB9KVxuICAgIH1cbn1cbiIsImltcG9ydCB7IG1ha2VUZXh0dXJlLCBtYWtlRnJhbWVCdWZmZXIsIGNoZWNrUmVuZGVyRmxvYXQgfSBmcm9tICcuL2hlbHBlcnMuanMnXG5pbXBvcnQgRm9ybWF0cyBmcm9tICcuLi9mb3JtYXQvaW5kZXguanMnXG5cbi8vIFRoZSB0ZW5zb3IgZm9ybWF0IGlzIGEgSlNPTiBvYmplY3QgdGhhdCBzcGVjaWZpZXMgaG93IFxuLy8gdGhlIHRlbnNvciBpcyByZXByZXNlbnRlZCBhcyBhIHRleHR1cmVcbi8vIGl0IGNvbnNpc3RzIG9mIHNldmVyYWwga2V5czpcblxuLy8gICAgIHR5cGU6IHVpbnQ4IHwgZmxvYXQzMlxuLy8gICAgIGRlbnNpdHk6IDQ6NCB8IDE6NFxuLy8gICAgIHBhY2s6IHN0cmlkZSB8IHRpbGVcbi8vICAgICBjb2RlYzogXG4vL1x0XHRcdHNvZnRmbG9hdCB8IGZpeG51bSAoMTo0KVxuLy8gICAgICAgICAgcmF3IHwgbGlucXVhbnQgKDQ6NClcblxuZXhwb3J0IGRlZmF1bHQgY2xhc3MgQmFzZVRlbnNvciB7XG5cdC8vIHdlIGFyZW50IHVzaW5nIGEgY29uc3RydWN0b3IgYmVjYXVzZSB3ZSB3YW50IHRvIGJlIGFibGUgdG8gcnVuXG5cdC8vIHRoaXMgaW5zdGFuY2VvZiBPdXRwdXRUZW5zb3IgZnJvbSB3aXRoaW4gdGhlIFRlbnNvciBjb25zdHJ1Y3RvclxuXHRcblx0X2luaXQoZ2wsIGZvcm1hdCwgc2hhcGUsIGRhdGEpe1xuXHRcdC8vIHZhbGlkYXRlIGdsY29udGV4dFxuXHRcdGlmKCFnbC5jcmVhdGVUZXh0dXJlKSB0aHJvdyBuZXcgRXJyb3IoJ0ludmFsaWQgV2ViR0xSZW5kZXJpbmdDb250ZXh0Jyk7XG5cdFx0dGhpcy5nbCA9IGdsO1xuXG5cdFx0Ly8gdmFsaWRhdGUgc2hhcGVcblx0XHRpZighQXJyYXkuaXNBcnJheShzaGFwZSkpIHRocm93IG5ldyBFcnJvcihcInNoYXBlIG11c3QgYmUgQXJyYXlcIik7XG5cdFx0aWYoc2hhcGUubGVuZ3RoID4gNCkgdGhyb3cgbmV3IEVycm9yKFwiVGVuc29yIG11c3QgaGF2ZSBkaW1lbnNpb24gPD0gNFwiKTtcbiAgICAgICAgaWYoc2hhcGUuc29tZShrID0+ICFpc0Zpbml0ZShrKSB8fCBrIDwgMSB8fCAhTnVtYmVyLmlzSW50ZWdlcihrKSkpIFxuICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdJbnZhbGlkIHNoYXBlOiAnICsgc2hhcGUpO1xuICAgICAgICBzaGFwZSA9IHNoYXBlLmNvbmNhdChbMSwgMSwgMSwgMV0pLnNsaWNlKDAsIDQpXG5cdFx0dGhpcy5zaGFwZSA9IHNoYXBlO1xuXHRcdFxuXHRcdC8vIHZhbGlkYXRlIGZvcm1hdFxuXHRcdGlmKCFbJ2Zsb2F0MzInLCAndWludDgnXS5pbmNsdWRlcyhmb3JtYXQudHlwZSkpXG5cdFx0XHR0aHJvdyBuZXcgRXJyb3IoJ2Zvcm1hdC50eXBlIG11c3QgYmUgdWludDggb3IgZmxvYXQzMicpO1xuXHRcdGlmKGZvcm1hdC5kZW5zaXR5IGluIEZvcm1hdHMpe1xuXHRcdFx0bGV0IGZkID0gRm9ybWF0c1tmb3JtYXQuZGVuc2l0eV07XG5cdFx0XHRpZighKGZvcm1hdC5wYWNrIGluIGZkLnBhY2spKSBcblx0XHRcdFx0dGhyb3cgbmV3IEVycm9yKCdmb3JtYXQucGFjayBtdXN0IGJlICcgKyBPYmplY3Qua2V5cyhmZC5wYWNrKS5qb2luKCcgb3IgJykpO1xuXHRcdFx0aWYoIShmb3JtYXQuY29kZWMgaW4gZmQuY29kZWMpKSBcblx0XHRcdFx0dGhyb3cgbmV3IEVycm9yKCdmb3JtYXQuY29kZWMgbXVzdCBiZSAnICsgT2JqZWN0LmtleXMoZmQuY29kZWMpLmpvaW4oJyBvciAnKSk7XG5cdFx0fWVsc2UgdGhyb3cgbmV3IEVycm9yKCdmb3JtYXQuZGVuc2l0eSBtdXN0IGJlICcgKyBPYmplY3Qua2V5cyhGb3JtYXRzKS5qb2luKCcgb3IgJykpO1xuXG5cdFx0dGhpcy5mb3JtYXQgPSBmb3JtYXQ7XG5cblx0XHQvLyBjYWxjdWxhdGUgdGV4dHVyZSBzaXplXG5cdFx0dGhpcy5pbmZvID0gT2JqZWN0LmFzc2lnbih7fSxcblx0XHRcdHRoaXMuX2Zvcm1hdC5wYWNrLmluaXQoc2hhcGUsIGZvcm1hdCksXG5cdFx0XHR0aGlzLl9mb3JtYXQuY29kZWMuaW5pdChzaGFwZSwgZm9ybWF0KVxuXHRcdCk7XG5cdFx0aWYoIXRoaXMuaW5mby50ZXhTaXplKSB0aHJvdyBuZXcgRXJyb3IoJ0Zvcm1hdCBkaWQgbm90IHlpZWxkIHRleFNpemUnKTtcblxuXHRcdC8vIGluaXRpYWxpemUgdGV4dHVyZVxuXHRcdHRoaXMudGV4ID0gbWFrZVRleHR1cmUoZ2wpO1xuXHRcdHRoaXMudXBkYXRlKGRhdGEpXG5cdH1cblx0X3VwZGF0ZShkYXRhKXtcblx0XHRpZihkYXRhICE9PSBudWxsKXtcblx0XHRcdGlmKHRoaXMuZm9ybWF0LnR5cGUgPT09ICd1aW50OCcpe1xuXHRcdFx0XHRpZihBcnJheS5pc0FycmF5KGRhdGEpIHx8IGRhdGEgaW5zdGFuY2VvZiBVaW50OENsYW1wZWRBcnJheSlcblx0XHRcdFx0XHRkYXRhID0gbmV3IFVpbnQ4QXJyYXkoZGF0YSk7XG5cdFx0XHRcdGlmKCEoZGF0YSBpbnN0YW5jZW9mIFVpbnQ4QXJyYXkpKVxuXHRcdFx0XHRcdHRocm93IG5ldyBFcnJvcignZGF0YSBtdXN0IGJlIFVpbnQ4QXJyYXknKTtcblx0XHRcdH1lbHNlIGlmKHRoaXMuZm9ybWF0LnR5cGUgPT09ICdmbG9hdDMyJyl7XG5cdFx0XHRcdGlmKEFycmF5LmlzQXJyYXkoZGF0YSkgfHwgZGF0YSBpbnN0YW5jZW9mIEZsb2F0NjRBcnJheSlcblx0XHRcdFx0XHRkYXRhID0gbmV3IEZsb2F0MzJBcnJheShkYXRhKTtcblx0XHRcdFx0aWYoIShkYXRhIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KSlcblx0XHRcdFx0XHR0aHJvdyBuZXcgRXJyb3IoJ2RhdGEgbXVzdCBiZSBGbG9hdDMyQXJyYXknKTtcblx0XHRcdH1lbHNlIHRocm93IG5ldyBFcnJvcignVHlwZSBtdXN0IGJlIHVpbnQ4IG9yIGZsb2F0MzInKTtcblx0XHRcdGlmKGRhdGEubGVuZ3RoICE9PSB0aGlzLmluZm8udGV4U2l6ZVswXSAqIHRoaXMuaW5mby50ZXhTaXplWzFdICogNClcblx0XHRcdFx0dGhyb3cgbmV3IEVycm9yKCdkYXRhIGlzIHRoZSB3cm9uZyBsZW5ndGgnKTtcblx0XHR9XG5cdFx0Ly8gaWYoZGF0YSkgY29uc29sZS5sb2coJ191cGRhdGUnLCBkYXRhKTtcblx0XHR2YXIgZ2wgPSB0aGlzLmdsO1xuICAgICAgICBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0aGlzLnRleCk7XG4gICAgICAgIGdsLnRleEltYWdlMkQoZ2wuVEVYVFVSRV8yRCwgMCwgZ2wuUkdCQSwgXG4gICAgICAgIFx0dGhpcy5pbmZvLnRleFNpemVbMF0sIHRoaXMuaW5mby50ZXhTaXplWzFdLCAwLCBnbC5SR0JBLCBcbiAgICAgICAgXHR0aGlzLmZvcm1hdC50eXBlID09ICd1aW50OCcgPyBnbC5VTlNJR05FRF9CWVRFIDogZ2wuRkxPQVQsIGRhdGEpO1xuXHR9XG5cblx0dXBkYXRlKGRhdGEpe1xuXHRcdGlmKCFkYXRhKSByZXR1cm4gdGhpcy5fdXBkYXRlKG51bGwpO1xuXHRcdGlmKGRhdGEuc2hhcGUpIHJldHVybiB0aGlzLl91cGRhdGUoXG5cdFx0XHR0aGlzLl9mb3JtYXQucGFjay5wYWNrKHRoaXMuaW5mbywgZGF0YSwgdGhpcy5fZm9ybWF0LmNvZGVjLmVuY29kZSwgdGhpcy5mb3JtYXQpKTtcblx0XHRpZih0aGlzLnR5cGUgIT0gJ3VpbnQ4JykgY29uc29sZS53YXJuKCdDYWxsaW5nIHVwZGF0ZSB3aXRoIHJhdyBUeXBlZEFycmF5IG1heSBub3Qgd29yayBhY3Jvc3MgYWxsIGJyb3dzZXJzLicpO1xuXHRcdHJldHVybiB0aGlzLl91cGRhdGUoZGF0YSk7XG5cdH1cblxuXHRnZXQgX2Zvcm1hdCgpe1xuXHRcdHJldHVybiB7XG5cdFx0XHRwYWNrOiBGb3JtYXRzW3RoaXMuZm9ybWF0LmRlbnNpdHldLnBhY2tbdGhpcy5mb3JtYXQucGFja10sXG5cdFx0XHRjb2RlYzogRm9ybWF0c1t0aGlzLmZvcm1hdC5kZW5zaXR5XS5jb2RlY1t0aGlzLmZvcm1hdC5jb2RlY10sXG5cdFx0XHRhY3RpdmF0aW9uczogRm9ybWF0c1t0aGlzLmZvcm1hdC5kZW5zaXR5XS5hY3RpdmF0aW9ucyxcblx0XHRcdHJlYWRfc2hpbTogRm9ybWF0c1t0aGlzLmZvcm1hdC5kZW5zaXR5XS5yZWFkX3NoaW0sXG5cdFx0XHR3cml0ZV9zaGltOiBGb3JtYXRzW3RoaXMuZm9ybWF0LmRlbnNpdHldLndyaXRlX3NoaW1cblx0XHR9XG5cdH1cblxuICAgIGRlc3Ryb3koKXsgdGhpcy5nbC5kZWxldGVUZXh0dXJlKHRoaXMudGV4KSB9XG59IiwiaW1wb3J0IHsgYmluZEF0dHJpYnV0ZUJ1ZmZlciwgY3JlYXRlU2hhZGVyUHJvZ3JhbSB9IGZyb20gJy4uL3J1bnRpbWUvcHJvZ3JhbS5qcydcbmltcG9ydCB7IG1ha2VGcmFtZUJ1ZmZlciwgbWFrZVRleHR1cmUgfSBmcm9tICcuL2hlbHBlcnMuanMnXG5cbmV4cG9ydCBkZWZhdWx0IGZ1bmN0aW9uIHJ1bkZlYXR1cmVUZXN0cyhnbCl7XG4gICAgXG4gICAgaWYoIWdsLkZMT0FUX1RFWFRVUkVTX1RFU1RFRCAmJiAhZ2wuTk9fRkxPQVRfVEVYVFVSRVMpe1xuICAgICAgICBpZighZ2wuZ2V0RXh0ZW5zaW9uKCdPRVNfdGV4dHVyZV9mbG9hdCcpKXtcbiAgICAgICAgICAgIGNvbnNvbGUuaW5mbyhcIlRoaXMgYnJvd3NlciBkb2VzIG5vdCBzZWVtIHRvIHN1cHBvcnQgT0VTX3RleHR1cmVfZmxvYXQuIFwiXG4gICAgICAgICAgICAgICAgKyBcIlVzaW5nIGZsb2F0IGNvZGVjIHdvcmthcm91bmQgZnJvbSBub3cgb24uXCIpXG4gICAgICAgICAgICBnbC5OT19GTE9BVF9URVhUVVJFUyA9IHRydWU7XG4gICAgICAgIH1cbiAgICAgICAgZ2wuRkxPQVRfVEVYVFVSRVNfVEVTVEVEID0gdHJ1ZTtcbiAgICB9XG5cbiAgICBpZighZ2wuTk9fRkxPQVRfVEVYVFVSRVMpe1xuICAgICAgICBpZighZ2wuUkVOREVSX0ZMT0FUX1RFU1RFRCAmJiAhZ2wuTk9fUkVOREVSX0ZMT0FUKXtcbiAgICAgICAgICAgIGlmKCF0ZXN0UmVuZGVyRmxvYXQoZ2wpKXtcbiAgICAgICAgICAgICAgICBjb25zb2xlLmluZm8oXCJUaGlzIGJyb3dzZXIgc3VwcG9ydHMgT0VTX3RleHR1cmVfZmxvYXQsIFwiICsgXG4gICAgICAgICAgICAgICAgICAgIFwiYnV0IGNhbiBub3QgcmVuZGVyIHRvIGZsb2F0aW5nIHRleHR1cmVzLiBcIiArIFxuICAgICAgICAgICAgICAgICAgICBcIlVzaW5nIGZsb2F0IGNvZGVjIHdvcmthcm91bmQgZm9yIG91dHB1dCB0ZW5zb3JzIGZyb20gbm93IG9uLlwiKVxuICAgICAgICAgICAgICAgIGdsLk5PX1JFTkRFUl9GTE9BVCA9IHRydWU7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBnbC5SRU5ERVJfRkxPQVRfVEVTVEVEID0gdHJ1ZTtcbiAgICAgICAgfVxuXG4gICAgICAgIGlmKCFnbC5SRUFEX0ZMT0FUX1RFU1RFRCAmJiAhZ2wuTk9fUkVBRF9GTE9BVCAmJiAhZ2wuTk9fUkVBRF9GTE9BVCl7XG4gICAgICAgICAgICBpZighdGVzdFJlYWRGbG9hdChnbCkpe1xuICAgICAgICAgICAgICAgIGNvbnNvbGUuaW5mbyhcIlRoaXMgYnJvd3NlciBzdXBwb3J0cyBPRVNfdGV4dHVyZV9mbG9hdCwgXCIgKyBcbiAgICAgICAgICAgICAgICAgICAgXCJjYW4gcmVuZGVyIHRvIGZsb2F0aW5nIHBvaW50IHRleHR1cmVzLCBidXQgY2FuIG5vdCBcIiArXG4gICAgICAgICAgICAgICAgICAgIFwicmVhZCBpbnRvIGEgRmxvYXQzMkFycmF5IGJ1ZmZlci4gVXNpbmcgZmxvYXQgY29kZWMgXCIgK1xuICAgICAgICAgICAgICAgICAgICBcIndvcmthcm91bmQgZm9yIHJlYWRpbmcgZnJvbSBvdXRwdXQgdGVuc29ycyBmcm9tIG5vdyBvbi5cIilcbiAgICAgICAgICAgICAgICBnbC5OT19SRUFEX0ZMT0FUID0gdHJ1ZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGdsLlJFQURfRkxPQVRfVEVTVEVEID0gdHJ1ZTtcbiAgICAgICAgfVxuICAgIH1cblxuXG59XG5cblxuY29uc3QgQ0hFQ0tfRkxPQVRfVkVSVEVYID0gYFxuICAgIGF0dHJpYnV0ZSB2ZWMyIGFfcG9zaXRpb247XG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgICBnbF9Qb3NpdGlvbiA9IHZlYzQoYV9wb3NpdGlvbiwgMCwgMSk7XG4gICAgfVxuYFxuY29uc3QgQ0hFQ0tfRkxPQVRfRlJBR01FTlQgPSBgXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KDMuMTQxNTksIC0yLjcxODI4LCAxLjYxODI4LCA0Mik7XG4gICAgfVxuYDtcblxuLy8gc29tZSBicm93c2VycyAoZS5nLiBtb2JpbGUgc2FmYXJpKSBhcmUgY2FwYWJsZSBvZiBpbml0aWFsaXppbmcgZmxvYXRpbmcgXG4vLyBwb2ludCB0ZXh0dXJlcyBidXQgdW5hYmxlIHRvIHdyaXRlIHRvIHRoZW0uIFRoZSBvbmx5IHdheSBvZiBmaW5kaW5nIHRoaXNcbi8vIG91dCBpcyBieSB0cnlpbmcgdG8gcmVuZGVyIHRvIGEgZmxvYXRpbmcgcG9pbnQgdGV4dHVyZSBhbmQgbm90aWNpbmdcbi8vIHRoZSBpbnZhbGlkIGZyYW1lYnVmZmVyIHN0YXR1cy5cblxuZXhwb3J0IGZ1bmN0aW9uIHRlc3RSZW5kZXJGbG9hdChnbCl7XG4gICAgdmFyIHRleCA9IG1ha2VUZXh0dXJlKGdsKVxuICAgIGdsLnRleEltYWdlMkQoZ2wuVEVYVFVSRV8yRCwgMCwgZ2wuUkdCQSwgMTAsIDEwLCAwLCBnbC5SR0JBLCBnbC5GTE9BVCwgbnVsbCk7XG4gICAgdmFyIGZibyA9IG1ha2VGcmFtZUJ1ZmZlcihnbCwgdGV4KTtcblxuICAgIHZhciBwcm9ncmFtID0gY3JlYXRlU2hhZGVyUHJvZ3JhbShnbCwgQ0hFQ0tfRkxPQVRfVkVSVEVYLCBDSEVDS19GTE9BVF9GUkFHTUVOVCk7XG4gICAgZ2wudXNlUHJvZ3JhbShwcm9ncmFtKTtcbiAgICBiaW5kQXR0cmlidXRlQnVmZmVyKGdsLCBwcm9ncmFtKTtcblxuICAgIGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgZmJvKTtcbiAgICBnbC52aWV3cG9ydCgwLCAwLCAxMCwgMTApO1xuICAgIGdsLmRyYXdBcnJheXMoZ2wuVFJJQU5HTEVfU1RSSVAsIDAsIDQpO1xuXG4gICAgdmFyIHN0YXR1cyA9IGdsLmNoZWNrRnJhbWVidWZmZXJTdGF0dXMoZ2wuRlJBTUVCVUZGRVIpO1xuICAgIGdsLmRlbGV0ZVRleHR1cmUodGV4KVxuICAgIGdsLmRlbGV0ZUZyYW1lYnVmZmVyKGZibylcbiAgICBnbC5kZWxldGVQcm9ncmFtKHByb2dyYW0pXG5cbiAgICByZXR1cm4gc3RhdHVzID09IGdsLkZSQU1FQlVGRkVSX0NPTVBMRVRFO1xufVxuXG5cbmZ1bmN0aW9uIHRlc3RSZWFkRmxvYXQoZ2wpe1xuICAgIHZhciB0ZXggPSBtYWtlVGV4dHVyZShnbClcbiAgICBnbC50ZXhJbWFnZTJEKGdsLlRFWFRVUkVfMkQsIDAsIGdsLlJHQkEsIDEwLCAxMCwgMCwgZ2wuUkdCQSwgZ2wuRkxPQVQsIG51bGwpO1xuICAgIHZhciBmYm8gPSBtYWtlRnJhbWVCdWZmZXIoZ2wsIHRleCk7XG5cbiAgICB2YXIgcHJvZ3JhbSA9IGNyZWF0ZVNoYWRlclByb2dyYW0oZ2wsIENIRUNLX0ZMT0FUX1ZFUlRFWCwgQ0hFQ0tfRkxPQVRfRlJBR01FTlQpO1xuICAgIGdsLnVzZVByb2dyYW0ocHJvZ3JhbSk7XG4gICAgYmluZEF0dHJpYnV0ZUJ1ZmZlcihnbCwgcHJvZ3JhbSk7XG5cbiAgICBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIGZibyk7XG4gICAgZ2wudmlld3BvcnQoMCwgMCwgMTAsIDEwKTtcbiAgICBnbC5kcmF3QXJyYXlzKGdsLlRSSUFOR0xFX1NUUklQLCAwLCA0KTtcblxuICAgIHZhciBzaXplID0gWzMsIDNdO1xuICAgIHZhciBwaXhlbHMgPSBwaXhlbHMgPSBuZXcgRmxvYXQzMkFycmF5KHNpemVbMF0gKiBzaXplWzFdICogNClcbiAgICBnbC5yZWFkUGl4ZWxzKDAsIDAsIHNpemVbMF0sIHNpemVbMV0sIGdsLlJHQkEsIGdsLkZMT0FULCBwaXhlbHMpO1xuXG4gICAgZ2wuZGVsZXRlVGV4dHVyZSh0ZXgpXG4gICAgZ2wuZGVsZXRlRnJhbWVidWZmZXIoZmJvKVxuICAgIGdsLmRlbGV0ZVByb2dyYW0ocHJvZ3JhbSlcblxuXG4gICAgdmFyIHRvdGFsX2Vycm9yID0gTWF0aC5hYnMocGl4ZWxzWzBdIC0gMy4xNDE1OSkgK1xuICAgICAgICAgICAgTWF0aC5hYnMocGl4ZWxzWzFdICsgMi43MTgyOCkgK1xuICAgICAgICAgICAgTWF0aC5hYnMocGl4ZWxzWzJdIC0gMS42MTgyOCkgK1xuICAgICAgICAgICAgTWF0aC5hYnMocGl4ZWxzWzNdIC0gNDIpO1xuXG4gICAgcmV0dXJuIHRvdGFsX2Vycm9yIDwgMC4wMTtcbn1cbiIsImV4cG9ydCBmdW5jdGlvbiBtYWtlRnJhbWVCdWZmZXIoZ2wsIHRleHR1cmUpe1xuICAgIHZhciBmcmFtZWJ1ZmZlciA9IGdsLmNyZWF0ZUZyYW1lYnVmZmVyKCk7XG4gICAgZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBmcmFtZWJ1ZmZlcik7XG4gICAgZ2wuZnJhbWVidWZmZXJUZXh0dXJlMkQoZ2wuRlJBTUVCVUZGRVIsIGdsLkNPTE9SX0FUVEFDSE1FTlQwLCBnbC5URVhUVVJFXzJELCB0ZXh0dXJlLCAwKTtcbiAgICByZXR1cm4gZnJhbWVidWZmZXI7XG59XG5cblxuZXhwb3J0IGZ1bmN0aW9uIG1ha2VUZXh0dXJlKGdsKXtcbiAgICB2YXIgdGV4dHVyZSA9IGdsLmNyZWF0ZVRleHR1cmUoKTtcbiAgICBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0ZXh0dXJlKTtcbiAgICBnbC50ZXhQYXJhbWV0ZXJpKGdsLlRFWFRVUkVfMkQsIGdsLlRFWFRVUkVfV1JBUF9TLCBnbC5DTEFNUF9UT19FREdFKTtcbiAgICBnbC50ZXhQYXJhbWV0ZXJpKGdsLlRFWFRVUkVfMkQsIGdsLlRFWFRVUkVfV1JBUF9ULCBnbC5DTEFNUF9UT19FREdFKTtcbiAgICBnbC50ZXhQYXJhbWV0ZXJpKGdsLlRFWFRVUkVfMkQsIGdsLlRFWFRVUkVfTUlOX0ZJTFRFUiwgZ2wuTkVBUkVTVCk7XG4gICAgZ2wudGV4UGFyYW1ldGVyaShnbC5URVhUVVJFXzJELCBnbC5URVhUVVJFX01BR19GSUxURVIsIGdsLk5FQVJFU1QpO1xuXG4gICAgcmV0dXJuIHRleHR1cmU7XG59XG5cbiIsImltcG9ydCBCYXNlVGVuc29yIGZyb20gJy4vYmFzZS5qcyc7XG5pbXBvcnQgc2hvd1RleHR1cmUgZnJvbSAnLi9zaG93LmpzJ1xuaW1wb3J0IHJ1bkZlYXR1cmVUZXN0cyBmcm9tICcuL2ZlYXR1cmUuanMnXG5pbXBvcnQgeyBtYWtlVGV4dHVyZSwgbWFrZUZyYW1lQnVmZmVyIH0gZnJvbSAnLi9oZWxwZXJzLmpzJ1xuaW1wb3J0IHsgUnVuLCBDb21waWxlIH0gZnJvbSAnLi4vcnVudGltZS9pbmRleC5qcydcbmltcG9ydCBuZHNob3cgZnJvbSAnbmRhcnJheS1zaG93J1xuaW1wb3J0IG5kYXJyYXkgZnJvbSAnbmRhcnJheSdcblxuZXhwb3J0IGNsYXNzIFRlbnNvciBleHRlbmRzIEJhc2VUZW5zb3Ige1xuICAgIC8vIG5ldyBUZW5zb3IoZ2wpXG4gICAgLy8gbmV3IFRlbnNvcihnbCwgWzEsIDFdKVxuICAgIC8vIG5ldyBUZW5zb3IoZ2wsIFsxLCAxXSwgbnVsbClcbiAgICAvLyBuZXcgVGVuc29yKGdsLCBbMSwgMV0sIGRhdGEpXG4gICAgLy8gbmV3IFRlbnNvcihnbCwgWzEsIDFdLCBkYXRhLCB7IHR5cGUsIHBhY2ssIGNvZGVjLCBkZW5zaXR5IH0pXG4gICAgLy8gbmV3IFRlbnNvcihnbCwgWzEsIDFdLCB7IHR5cGUsIHBhY2ssIGNvZGVjLCBkZW5zaXR5IH0pXG4gICAgLy8gbmV3IFRlbnNvcihnbCwgWzEsIDFdLCAnc29mdGZsb2F0JylcbiAgICAvLyBuZXcgVGVuc29yKGdsLCBbMSwgMV0sICdmbG9hdDMyJylcbiAgICAvLyBuZXcgVGVuc29yKGdsLCBbMSwgMV0sICd1aW50OCcpXG4gICAgLy8gbmV3IFRlbnNvcihnbCwgeyBzaGFwZSwgZGF0YSB9KVxuICAgIC8vIG5ldyBUZW5zb3IoZ2wsIHsgd2lkdGgsIGhlaWdodCwgZGF0YSB9KVxuICAgIC8vIHBpeCA9IG5ldyBUZW5zb3IoZ2wsIFsxLCAxLCA0XSwgWzEsIDAuNCwgMywgNF0sICd1aW50OCcpXG5cblx0Y29uc3RydWN0b3IoZ2wsIHNoYXBlID0gW10sIGRhdGEgPSBudWxsLCBmb3JtYXQgPSBudWxsKXtcbiAgICAgICAgc3VwZXIoKVxuICAgICAgICBydW5GZWF0dXJlVGVzdHMoZ2wpO1xuXG4gICAgICAgIHZhciB4ZGF0YSA9IGRhdGE7XG4gICAgICAgIGlmKHNoYXBlLnNoYXBlKXsgLy8gbmRhcnJheXNcbiAgICAgICAgICAgIGZvcm1hdCA9IGRhdGE7XG4gICAgICAgICAgICB4ZGF0YSA9IHNoYXBlLmRhdGE7XG4gICAgICAgICAgICBkYXRhID0gc2hhcGU7XG4gICAgICAgICAgICBzaGFwZSA9IHNoYXBlLnNoYXBlO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYoc2hhcGUud2lkdGggJiYgc2hhcGUuaGVpZ2h0ICYmIHNoYXBlLmRhdGEpeyAvLyBpbWFnZWRhdGFcbiAgICAgICAgICAgIGRhdGEgPSBzaGFwZS5kYXRhO1xuICAgICAgICAgICAgc2hhcGUgPSBbc2hhcGUud2lkdGgsIHNoYXBlLmhlaWdodF1cbiAgICAgICAgfVxuXG4gICAgICAgIGlmKHR5cGVvZiBkYXRhID09PSAnc3RyaW5nJyl7IC8vIGRhdGEgPSB1aW50OCB8IGZsb2F0MzJcbiAgICAgICAgICAgIGlmKGZvcm1hdCAhPT0gbnVsbClcbiAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ0Zvcm1hdCBtdXN0IG5vdCBiZSBzcGVjaWZpZWQgaWYgZGF0YSBpcyBhIHN0cmluZy4nKTtcbiAgICAgICAgICAgIGZvcm1hdCA9IGRhdGE7XG4gICAgICAgICAgICBkYXRhID0gbnVsbDtcbiAgICAgICAgfWVsc2UgaWYoZGF0YSAmJiB0eXBlb2YgZGF0YSA9PT0gJ29iamVjdCcgJiYgZGF0YS50eXBlICYmIGRhdGEuY29kZWMgJiYgZGF0YS5wYWNrICYmIGRhdGEuZGVuc2l0eSl7XG4gICAgICAgICAgICBpZihmb3JtYXQgIT09IG51bGwpXG4gICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdGb3JtYXQgbXVzdCBub3QgYmUgc3BlY2lmaWVkIGlmIGRhdGEgaXMgYW4gb2JqZWN0LicpO1xuICAgICAgICAgICAgZm9ybWF0ID0gZGF0YTtcbiAgICAgICAgICAgIGRhdGEgPSBudWxsO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYoZm9ybWF0ID09PSBudWxsKXsgLy8gYXV0by1pbmZlciBmb3JtYXQgYmFzZWQgb24gZGF0YVxuICAgICAgICAgICAgaWYoZGF0YSA9PT0gbnVsbCl7XG4gICAgICAgICAgICAgICAgZm9ybWF0ID0gJ2Zsb2F0MzInXG4gICAgICAgICAgICB9ZWxzZSBpZih4ZGF0YSBpbnN0YW5jZW9mIFVpbnQ4QXJyYXkgfHwgeGRhdGEgaW5zdGFuY2VvZiBVaW50OENsYW1wZWRBcnJheSl7XG4gICAgICAgICAgICAgICAgZm9ybWF0ID0gJ3VpbnQ4J1xuICAgICAgICAgICAgfWVsc2UgaWYoeGRhdGEgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkgfHwgeGRhdGEgaW5zdGFuY2VvZiBGbG9hdDY0QXJyYXkgfHwgQXJyYXkuaXNBcnJheSh4ZGF0YSkpe1xuICAgICAgICAgICAgICAgIGZvcm1hdCA9ICdmbG9hdDMyJ1xuICAgICAgICAgICAgfWVsc2UgdGhyb3cgbmV3IEVycm9yKFwiSW52YWxpZCBmb3JtYXQgZm9yIGRhdGE6IG11c3QgYmUgVWludDhBcnJheSBvciBGbG9hdDMyQXJyYXkgb3IgbmRhcnJheVwiKTtcbiAgICAgICAgfVxuXG4gICAgICAgIHZhciB0eXBlID0gbnVsbDtcbiAgICAgICAgaWYoKGZvcm1hdCA9PT0gJ2Zsb2F0MzInICYmIFxuICAgICAgICAgICAgKGdsLk5PX0ZMT0FUX1RFWFRVUkVTIHx8IFxuICAgICAgICAgICAgKGdsLk5PX1JFTkRFUl9GTE9BVCAmJiB0aGlzIGluc3RhbmNlb2YgT3V0cHV0VGVuc29yKSkpXG4gICAgICAgICAgICB8fCBmb3JtYXQgPT09ICdzb2Z0ZmxvYXQnKXtcbiAgICAgICAgICAgIGZvcm1hdCA9IHsgdHlwZTogJ3VpbnQ4JywgcGFjazogJ3N0cmlkZScsIGRlbnNpdHk6ICcxOjQnLCBjb2RlYzogJ3NvZnRmbG9hdCcgfVxuICAgICAgICAgICAgdHlwZSA9ICdmbG9hdDMyJ1xuICAgICAgICB9ZWxzZSBpZihmb3JtYXQgPT09ICd1aW50OCcgfHwgZm9ybWF0ID09PSAnZmxvYXQzMicpe1xuICAgICAgICAgICAgZm9ybWF0ID0geyB0eXBlOiBmb3JtYXQsIHBhY2s6ICdzdHJpZGUnLCBkZW5zaXR5OiAnNDo0JywgY29kZWM6ICdyYXcnIH1cbiAgICAgICAgfVxuXG4gICAgICAgIHRoaXMudHlwZSA9IHR5cGUgfHwgZm9ybWF0LnR5cGU7XG4gICAgICAgIHRoaXMuX2luaXQoZ2wsIGZvcm1hdCwgc2hhcGUsIGRhdGEpO1xuXHR9XG5cblxuXHRjb3B5KGZvcm1hdCA9IHRoaXMudHlwZSwgVCA9IE91dHB1dFRlbnNvcil7XG4gICAgICAgIGNvbnN0IFRFTlNPUl9JREVOVElUWSA9IGBcbiAgICAgICAgICAgIHVuaWZvcm0gVGVuc29yIGltYWdlO1xuICAgICAgICAgICAgdmVjNCBwcm9jZXNzNChpdmVjNCBwb3MpIHsgcmV0dXJuIGltYWdlLnJlYWQ0KHBvcyk7IH1cbiAgICAgICAgYDtcbiAgICAgICAgdmFyIG91dCA9IG5ldyBUKHRoaXMuZ2wsIHRoaXMuc2hhcGUsIGZvcm1hdCk7XG4gICAgICAgIG91dC5ydW4oVEVOU09SX0lERU5USVRZLCB7IGltYWdlOiB0aGlzIH0pXG4gICAgICAgIHJldHVybiBvdXRcbiAgICB9XG5cbiAgICB3aXRoQ29weShmbiwgLi4uYXJncyl7XG4gICAgICAgIHZhciBjb3B5ID0gdGhpcy5jb3B5KC4uLmFyZ3MpO1xuICAgICAgICB2YXIgcmVzdWx0ID0gZm4oY29weSlcbiAgICAgICAgY29weS5kZXN0cm95KClcbiAgICAgICAgcmV0dXJuIHJlc3VsdDtcbiAgICB9XG5cblx0X3Nob3cob3B0ID0ge30peyBzaG93VGV4dHVyZSh0aGlzLmdsLCB0aGlzLnRleCwgb3B0KSB9XG4gICAgc2hvdyhvcHQgPSB7fSl7XG4gICAgICAgIHZhciBnbCA9IHRoaXMuZ2w7XG4gICAgICAgIGlmKHRoaXMuZm9ybWF0LnBhY2sgPT0gJ3RpbGUnIFxuICAgICAgICAgICAgJiYgdGhpcy5mb3JtYXQuZGVuc2l0eSA9PSAnNDo0JyBcbiAgICAgICAgICAgICYmIHRoaXMuZm9ybWF0LmNvZGVjID09ICdyYXcnKXtcbiAgICAgICAgICAgIHRoaXMuX3Nob3cob3B0KVxuICAgICAgICB9ZWxzZXtcbiAgICAgICAgICAgIC8vIEMuaW5mby5tYWluX2lucHV0Lm91dHB1dC5jb3B5KHsgdHlwZTogJ3VpbnQ4JywgcGFjazogJ3RpbGUnLCBkZW5zaXR5OiAnNDo0JywgY29kZWM6ICdsaW5xdWFudCcsIG1pbjogMCwgbWF4OiAyNTUgfSkuX3Nob3coeyB9KVxuICAgICAgICAgICAgdGhpcy53aXRoQ29weSh4ID0+IHguc2hvdyhvcHQpLCBcbiAgICAgICAgICAgICAgICB7IHR5cGU6IFxuICAgICAgICAgICAgICAgICAgICAoZ2wuTk9fRkxPQVRfVEVYVFVSRVMgfHwgZ2wuTk9fUkVOREVSX0ZMT0FUKSA/ICd1aW50OCcgOiAnZmxvYXQzMicsIFxuICAgICAgICAgICAgICAgICAgICBwYWNrOiAndGlsZScsIGRlbnNpdHk6ICc0OjQnLCBjb2RlYzogJ3JhdycgfSlcbiAgICAgICAgfTtcbiAgICB9XG5cbiAgICBydW4oc2hhZGVyLCBwYXJhbXMpe1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ09ubHkgT3V0cHV0VGVuc29yIGNhbiBydW4gc2hhZGVycy4nKVxuICAgIH1cbiAgICBjb21waWxlKHNoYWRlciwgcGFyYW1zKXtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdPbmx5IE91dHB1dFRlbnNvciBjYW4gY29tcGlsZSBzaGFkZXJzLicpXG4gICAgfVxuICAgIHJlYWQoKXtcbiAgICAgICAgY29uc29sZS53YXJuKFwiQ29weWluZyBiZWZvcmUgcmVhZC4uLlwiKVxuICAgICAgICByZXR1cm4gdGhpcy53aXRoQ29weSh4ID0+IHgucmVhZCgpKVxuICAgIH1cbiAgICBwcmludCgpe1xuICAgICAgICByZXR1cm4gbmRzaG93KHRoaXMucmVhZCgpKVxuICAgIH1cbiAgICBzd2FwKCl7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcIk9ubHkgSW5QbGFjZVRlbnNvciBjYW4gYmUgYm90aCBhIHBhcmFtZXRlciBhbmQgZGVzdGluYXRpb24uXCIpO1xuICAgIH1cbn1cblxuZXhwb3J0IGNsYXNzIE91dHB1dFRlbnNvciBleHRlbmRzIFRlbnNvciB7XG5cdGNvbnN0cnVjdG9yKC4uLmFyZ3Mpe1xuICAgICAgICBzdXBlciguLi5hcmdzKTtcblx0XHR0aGlzLmZibyA9IG1ha2VGcmFtZUJ1ZmZlcih0aGlzLmdsLCB0aGlzLnRleCk7XG5cdH1cblxuICAgIGRlc3Ryb3koKXtcbiAgICAgICAgc3VwZXIuZGVzdHJveSgpXG4gICAgICAgIHRoaXMuZ2wuZGVsZXRlRnJhbWVidWZmZXIodGhpcy5mYm8pXG4gICAgfVxuXG4gICAgX3JlYWQoKXtcbiAgICAgICAgdmFyIGdsID0gdGhpcy5nbCxcbiAgICAgICAgICAgIHNpemUgPSB0aGlzLmluZm8udGV4U2l6ZTtcblxuICAgICAgICBpZih0aGlzLmZvcm1hdC50eXBlID09ICd1aW50OCcpe1xuICAgICAgICAgICAgdmFyIGdsVHlwZSA9IGdsLlVOU0lHTkVEX0JZVEUsXG4gICAgICAgICAgICAgICAgcGl4ZWxzID0gbmV3IFVpbnQ4QXJyYXkoc2l6ZVswXSAqIHNpemVbMV0gKiA0KVxuICAgICAgICB9ZWxzZSBpZih0aGlzLmZvcm1hdC50eXBlID09PSAnZmxvYXQzMicpe1xuICAgICAgICAgICAgdmFyIGdsVHlwZSA9IGdsLkZMT0FULFxuICAgICAgICAgICAgICAgIHBpeGVscyA9IG5ldyBGbG9hdDMyQXJyYXkoc2l6ZVswXSAqIHNpemVbMV0gKiA0KVxuICAgICAgICB9XG5cbiAgICAgICAgZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCB0aGlzLmZibyk7XG4gICAgICAgIGdsLnJlYWRQaXhlbHMoMCwgMCwgc2l6ZVswXSwgc2l6ZVsxXSwgZ2wuUkdCQSwgZ2xUeXBlLCBwaXhlbHMpO1xuXG4gICAgICAgIC8vIGNvbnNvbGUubG9nKCdfX19yZWFkJywgcGl4ZWxzKVxuICAgICAgICByZXR1cm4gcGl4ZWxzO1xuICAgIH1cblxuICAgIHJ1bihzaGFkZXIsIHBhcmFtcywgY2FsbGJhY2spe1xuICAgICAgICByZXR1cm4gUnVuKHNoYWRlciwgdGhpcywgcGFyYW1zLCBjYWxsYmFjayk7XG4gICAgfVxuICAgIGNvbXBpbGUoc2hhZGVyLCBwYXJhbXMpe1xuICAgICAgICByZXR1cm4gQ29tcGlsZShzaGFkZXIsIHRoaXMsIHBhcmFtcyk7XG4gICAgfVxuXG5cdHJlYWQoKXtcbiAgICAgICAgaWYodGhpcy5mb3JtYXQudHlwZSA9PT0gJ2Zsb2F0MzInICYmIHRoaXMuZ2wuTk9fUkVBRF9GTE9BVCl7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy53aXRoQ29weSh4ID0+IHgucmVhZCgpLCAnc29mdGZsb2F0JylcbiAgICAgICAgfVxuXG5cdFx0dmFyIGFycmF5ID0gdGhpcy5fZm9ybWF0LnBhY2sudW5wYWNrKHRoaXMuaW5mbywgdGhpcy5fcmVhZCgpLCB0aGlzLl9mb3JtYXQuY29kZWMuZGVjb2RlLCB0aGlzLnR5cGUpO1xuICAgICAgICBcbiAgICAgICAgLy8gc3RyaXAgdHJhaWxpbmcgc2luZ2xldG9uIGRpbWVuc2lvbnNcbiAgICAgICAgdmFyIHNoYXBlID0gYXJyYXkuc2hhcGUuc2xpY2UoMCksXG4gICAgICAgICAgICBzdHJpZGUgPSBhcnJheS5zdHJpZGUuc2xpY2UoMCk7XG4gICAgICAgIHdoaWxlKHNoYXBlW3NoYXBlLmxlbmd0aCAtIDFdID09IDEgJiYgc2hhcGUubGVuZ3RoID4gMSl7XG4gICAgICAgICAgICBzaGFwZS5wb3AoKVxuICAgICAgICAgICAgc3RyaWRlLnBvcCgpXG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG5kYXJyYXkoYXJyYXkuZGF0YSwgc2hhcGUsIHN0cmlkZSwgYXJyYXkub2Zmc2V0KTtcblx0fVxufVxuXG5leHBvcnQgY2xhc3MgSW5QbGFjZVRlbnNvciBleHRlbmRzIE91dHB1dFRlbnNvciB7XG5cdGNvbnN0cnVjdG9yKC4uLmFyZ3Mpe1xuXHRcdHN1cGVyKC4uLmFyZ3MpXG5cbiAgICAgICAgdGhpcy50ZXgyID0gdGhpcy50ZXg7XG4gICAgICAgIHRoaXMudGV4ID0gbWFrZVRleHR1cmUodGhpcy5nbCk7XG5cdFx0dGhpcy51cGRhdGUobnVsbCk7XG4gICAgICAgIHRoaXMuc3dhcCgpXG5cdH1cbiAgICBkZXN0cm95KCl7XG4gICAgICAgIHN1cGVyLmRlc3Ryb3koKVxuICAgICAgICB0aGlzLmdsLmRlbGV0ZVRleHR1cmUodGhpcy50ZXgyKVxuICAgIH1cbiAgICBzd2FwKCl7XG4gICAgICAgIHZhciB0bXAgPSB0aGlzLnRleDtcbiAgICAgICAgdGhpcy50ZXggPSB0aGlzLnRleDI7XG4gICAgICAgIHRoaXMudGV4MiA9IHRtcDtcblxuICAgICAgICAvLyBUT0RPOiBpbnZlc3RpZ2F0ZSBwZXJmb3JtYW5jZSBvZiB1c2luZyBtdWx0aXBsZSBGQk9zIGluc3RlYWRcbiAgICAgICAgLy8gb2YgcmViaW5kaW5nIHRoZSBmcmFtZWJ1ZmZlclxuICAgICAgICB2YXIgZ2wgPSB0aGlzLmdsO1xuICAgICAgICBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIHRoaXMuZmJvKTtcbiAgICAgICAgZ2wuZnJhbWVidWZmZXJUZXh0dXJlMkQoZ2wuRlJBTUVCVUZGRVIsIGdsLkNPTE9SX0FUVEFDSE1FTlQwLCBnbC5URVhUVVJFXzJELCB0aGlzLnRleCwgMCk7XG4gICAgfVxufSIsImltcG9ydCB7IGJpbmRBdHRyaWJ1dGVCdWZmZXIsIGNyZWF0ZVNoYWRlclByb2dyYW0gfSBmcm9tICcuLi9ydW50aW1lL3Byb2dyYW0uanMnXG5cbmNvbnN0IFNIT1dfVEVYVFVSRV9WRVJURVggPSBgXG4gICAgYXR0cmlidXRlIHZlYzIgYV9wb3NpdGlvbjtcbiAgICB2YXJ5aW5nIG1lZGl1bXAgdmVjMiBwb3M7XG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgICBwb3MgPSAoYV9wb3NpdGlvbiArIHZlYzIoMSwgMSkpIC8gMi4wO1xuICAgICAgICBnbF9Qb3NpdGlvbiA9IHZlYzQoYV9wb3NpdGlvbiwgMCwgMSk7XG4gICAgfVxuYFxuXG5jb25zdCBTSE9XX1RFWFRVUkVfRlJBR01FTlQgPSBgXG4gICAgcHJlY2lzaW9uIG1lZGl1bXAgZmxvYXQ7XG5cbiAgICB1bmlmb3JtIHNhbXBsZXIyRCB0ZXg7XG4gICAgdW5pZm9ybSBmbG9hdCBzY2FsZTtcbiAgICB1bmlmb3JtIGZsb2F0IG9mZnNldDtcbiAgICB1bmlmb3JtIGJvb2wgdHJhbnNwb3NlO1xuICAgIHVuaWZvcm0gYm9vbCBmbGlwWDtcbiAgICB1bmlmb3JtIGJvb2wgZmxpcFk7XG4gICAgdW5pZm9ybSBpbnQgY2hhbm5lbHM7XG5cbiAgICB2YXJ5aW5nIHZlYzIgcG9zO1xuXG4gICAgdmVjNCBjb2xvcm1hcChmbG9hdCB4KSB7XG4gICAgICAgIGZsb2F0IHIgPSBjbGFtcCg4LjAgLyAzLjAgKiB4LCAwLjAsIDEuMCk7XG4gICAgICAgIGZsb2F0IGcgPSBjbGFtcCg4LjAgLyAzLjAgKiB4IC0gMS4wLCAwLjAsIDEuMCk7XG4gICAgICAgIGZsb2F0IGIgPSBjbGFtcCg0LjAgKiB4IC0gMy4wLCAwLjAsIDEuMCk7XG4gICAgICAgIHJldHVybiB2ZWM0KHIsIGcsIGIsIDEuMCk7XG4gICAgfVxuXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgICB2ZWMyIHAgPSBwb3M7XG4gICAgICAgIGlmKGZsaXBYKSBwLnggPSAxLjAgLSBwLng7XG4gICAgICAgIGlmKGZsaXBZKSBwLnkgPSAxLjAgLSBwLnk7XG4gICAgICAgIGlmKHRyYW5zcG9zZSkgcCA9IHAueXg7XG4gICAgICAgIGlmKGNoYW5uZWxzID09IDQpe1xuICAgICAgICAgICAgZ2xfRnJhZ0NvbG9yID0gdmVjNCh2ZWM0KG9mZnNldCwgb2Zmc2V0LCBvZmZzZXQsIG9mZnNldCkgXG4gICAgICAgICAgICAgICAgKyBzY2FsZSAqIHRleHR1cmUyRCh0ZXgsIHApKTtcbiAgICAgICAgfWVsc2UgaWYoY2hhbm5lbHMgPT0gMyl7XG4gICAgICAgICAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KHZlYzMob2Zmc2V0LCBvZmZzZXQsIG9mZnNldCkgXG4gICAgICAgICAgICAgICAgKyBzY2FsZSAqIHRleHR1cmUyRCh0ZXgsIHApLnJnYiwgMSk7XG4gICAgICAgIH1lbHNlIGlmKGNoYW5uZWxzID09IDIpe1xuICAgICAgICAgICAgZ2xfRnJhZ0NvbG9yID0gdmVjNCh2ZWMyKG9mZnNldCwgb2Zmc2V0KSBcbiAgICAgICAgICAgICAgICArIHNjYWxlICogdGV4dHVyZTJEKHRleCwgcCkucmcsIDAsIDEpO1xuICAgICAgICB9ZWxzZSBpZihjaGFubmVscyA9PSAxKXtcbiAgICAgICAgICAgIGdsX0ZyYWdDb2xvciA9IGNvbG9ybWFwKG9mZnNldCArIHNjYWxlICogdGV4dHVyZTJEKHRleCwgcCkucik7XG4gICAgICAgIH1cbiAgICB9XG5gXG5cbmV4cG9ydCBkZWZhdWx0IGZ1bmN0aW9uIHNob3dUZXh0dXJlKGdsLCB0ZXgsIG9wdCA9IHt9KXtcbiAgICBpZighZ2wuX3Nob3dQcm9ncmFtKXtcbiAgICAgICAgZ2wuX3Nob3dQcm9ncmFtID0gY3JlYXRlU2hhZGVyUHJvZ3JhbShnbCwgU0hPV19URVhUVVJFX1ZFUlRFWCwgU0hPV19URVhUVVJFX0ZSQUdNRU5UKTtcbiAgICAgICAgZ2wudXNlUHJvZ3JhbShnbC5fc2hvd1Byb2dyYW0pO1xuICAgICAgICBiaW5kQXR0cmlidXRlQnVmZmVyKGdsLCBnbC5fc2hvd1Byb2dyYW0pO1xuICAgICAgICBnbC51bmlmb3JtMWkoZ2wuZ2V0VW5pZm9ybUxvY2F0aW9uKGdsLl9zaG93UHJvZ3JhbSwgJ3RleCcpLCAwKTtcbiAgICB9XG4gICAgXG5cbiAgICBpZihnbC5jYW52YXMgJiYgZ2wuY2FudmFzLl90ZkF1dG8pe1xuICAgICAgICBnbC5jYW52YXMuc3R5bGUuZGlzcGxheSA9ICdibG9jaydcbiAgICAgICAgZ2wuY2FudmFzLnN0eWxlLnBvc2l0aW9uID0gJ2Fic29sdXRlJ1xuICAgICAgICBnbC5jYW52YXMuc3R5bGUudG9wID0gMDtcbiAgICAgICAgZ2wuY2FudmFzLnN0eWxlLmxlZnQgPSAwO1xuICAgICAgICBnbC5jYW52YXMuc3R5bGUud2lkdGggPSBNYXRoLm1pbihpbm5lckhlaWdodCwgaW5uZXJXaWR0aCkgKyAncHgnXG4gICAgICAgIGdsLmNhbnZhcy5zdHlsZS5oZWlnaHQgPSBNYXRoLm1pbihpbm5lckhlaWdodCwgaW5uZXJXaWR0aCkgKyAncHgnXG4gICAgfVxuXG4gICAgZ2wudXNlUHJvZ3JhbShnbC5fc2hvd1Byb2dyYW0pO1xuICAgIGdsLmFjdGl2ZVRleHR1cmUoZ2wuVEVYVFVSRTApO1xuICAgIGdsLmJpbmRUZXh0dXJlKGdsLlRFWFRVUkVfMkQsIHRleCk7XG4gICAgZ2wudW5pZm9ybTFmKGdsLmdldFVuaWZvcm1Mb2NhdGlvbihnbC5fc2hvd1Byb2dyYW0sICdzY2FsZScpLCBvcHQuc2NhbGUgfHwgMSlcbiAgICBnbC51bmlmb3JtMWYoZ2wuZ2V0VW5pZm9ybUxvY2F0aW9uKGdsLl9zaG93UHJvZ3JhbSwgJ29mZnNldCcpLCBvcHQub2Zmc2V0IHx8IDApXG4gICAgZ2wudW5pZm9ybTFpKGdsLmdldFVuaWZvcm1Mb2NhdGlvbihnbC5fc2hvd1Byb2dyYW0sICd0cmFuc3Bvc2UnKSwgb3B0LnRyYW5zcG9zZSA/IDEgOiAwKVxuICAgIGdsLnVuaWZvcm0xaShnbC5nZXRVbmlmb3JtTG9jYXRpb24oZ2wuX3Nob3dQcm9ncmFtLCAnZmxpcFgnKSwgb3B0LmZsaXBYID8gMSA6IDApXG4gICAgZ2wudW5pZm9ybTFpKGdsLmdldFVuaWZvcm1Mb2NhdGlvbihnbC5fc2hvd1Byb2dyYW0sICdmbGlwWScpLCBvcHQuZmxpcFkgPyAxIDogMClcbiAgICBnbC51bmlmb3JtMWkoZ2wuZ2V0VW5pZm9ybUxvY2F0aW9uKGdsLl9zaG93UHJvZ3JhbSwgJ2NoYW5uZWxzJyksIG9wdC5jaGFubmVscyB8fCAzKTtcblxuICAgIGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgbnVsbCk7XG4gICAgZ2wudmlld3BvcnQoMCwgMCwgZ2wuZHJhd2luZ0J1ZmZlcldpZHRoLCBnbC5kcmF3aW5nQnVmZmVySGVpZ2h0KTtcbiAgICBnbC5kcmF3QXJyYXlzKGdsLlRSSUFOR0xFX1NUUklQLCAwLCA0KTtcblxufVxuIiwiZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZUdMKGNhbnZhcyl7XG4gICAgaWYoIWNhbnZhcyl7XG4gICAgICAgIGNhbnZhcyA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2NhbnZhcycpO1xuICAgICAgICBjYW52YXMud2lkdGggPSA1MTJcbiAgICAgICAgY2FudmFzLmhlaWdodCA9IDUxMlxuICAgICAgICBjYW52YXMuc3R5bGUuZGlzcGxheSA9ICdub25lJztcbiAgICAgICAgY2FudmFzLl90ZkF1dG8gPSB0cnVlO1xuICAgICAgICBkb2N1bWVudC5ib2R5LmFwcGVuZENoaWxkKGNhbnZhcylcbiAgICB9XG4gICAgdmFyIGdsID0gY2FudmFzLmdldENvbnRleHQoXCJ3ZWJnbFwiLCB7IGFudGlhbGlhczogZmFsc2UgfSkgXG4gICAgICAgICAgfHwgY2FudmFzLmdldENvbnRleHQoXCJleHBlcmltZW50YWwtd2ViZ2xcIiwgeyBhbnRpYWxpYXM6IGZhbHNlIH0pO1xuICAgIGlmICghZ2wpIGFsZXJ0KCdDb3VsZCBub3QgaW5pdGlhbGl6ZSBXZWJHTCwgdHJ5IGFub3RoZXIgYnJvd3NlcicpO1xuICAgIHJldHVybiBnbDtcbn1cbiJdfQ==
