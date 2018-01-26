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

		console.log(testView);
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
			log += net.weights_version + ",";
			log += accuracy + ",";
			log += times.requested + ",";
			log += times.received + ",";
			log += times.loaded + ",";
			log += times.trained + "\n";
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJjbGllbnQvY2xpZW50LmpzIiwibGliL0xheWVycy5qcyIsImxpYi9Nb2RlbC5qcyIsImxpYi9sYXllcnMvQWN0aXZhdGlvbnMuanMiLCJsaWIvbGF5ZXJzL0Nyb3NzRW50cm9weS5qcyIsImxpYi9sYXllcnMvRGVuc2UuanMiLCJsaWIvbGF5ZXJzL01TRS5qcyIsImxpYi9sYXllcnMvT3V0cHV0LmpzIiwibGliL2xheWVycy9Tb2Z0bWF4LmpzIiwibGliL3V0aWwvZ2VuZXJhdGVXZWlnaHRzLmpzIiwibm9kZV9tb2R1bGVzL2ZpeGVkLXdpZHRoLWZsb2F0L2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL2lvdGEtYXJyYXkvaW90YS5qcyIsIm5vZGVfbW9kdWxlcy9pcy1idWZmZXIvaW5kZXguanMiLCJub2RlX21vZHVsZXMvbmRhcnJheS1zaG93L2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL25kYXJyYXkvbmRhcnJheS5qcyIsIm5vZGVfbW9kdWxlcy9zcHJpbnRmL2xpYi9zcHJpbnRmLmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC8xLTQvYWN0aXZhdGlvbi9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9mb3JtYXQvMS00L2NvZGVjL2ZpeG51bS9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9mb3JtYXQvMS00L2NvZGVjL3NvZnRmbG9hdC9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9mb3JtYXQvMS00L2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC8xLTQvcGFjay9zdHJpZGUvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvZm9ybWF0LzEtNC9wYWNrL3RpbGUvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvZm9ybWF0LzQtNC9hY3RpdmF0aW9uL2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC80LTQvY29kZWMvbGlucXVhbnQvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvZm9ybWF0LzQtNC9jb2RlYy9yYXcvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvZm9ybWF0LzQtNC9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9mb3JtYXQvNC00L3BhY2svc3RyaWRlL2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC80LTQvcGFjay90aWxlL2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL2Zvcm1hdC9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9pbmRleC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9ydW50aW1lL2NoZWNrLmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL3J1bnRpbWUvZnJhZy5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9ydW50aW1lL2luZGV4LmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL3J1bnRpbWUvcHJvZ3JhbS5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9ydW50aW1lL3RpbWVyLmpzIiwibm9kZV9tb2R1bGVzL3RlbnNvcmZpcmUvc3JjL3J1bnRpbWUvdG5zbC5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy90ZW5zb3IvYmFzZS5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy90ZW5zb3IvZmVhdHVyZS5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy90ZW5zb3IvaGVscGVycy5qcyIsIm5vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy90ZW5zb3IvaW5kZXguanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvdGVuc29yL3Nob3cuanMiLCJub2RlX21vZHVsZXMvdGVuc29yZmlyZS9zcmMvdXRpbC5qcyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7O0FDQUEsSUFBSSxRQUFRLFFBQVosQUFBWSxBQUFRO0lBQ25CLEtBQUssUUFETixBQUNNLEFBQVE7SUFDYixLQUFLLEdBRk4sQUFFTSxBQUFHOztBQUVULFNBQUEsQUFBUyxJQUFULEFBQWEsTUFBYixBQUFtQixjQUFuQixBQUFpQyxVQUFVLEFBQzFDO0tBQUksSUFBSSxJQUFSLEFBQVEsQUFBSSxBQUNaO0dBQUEsQUFBRSxxQkFBcUIsWUFBWSxBQUNsQztNQUFJLEVBQUEsQUFBRSxlQUFlLGVBQWpCLEFBQWdDLFFBQVEsRUFBQSxBQUFFLFdBQTlDLEFBQXlELEtBQUssQUFDN0Q7WUFBUyxFQUFULEFBQVcsQUFDWDtBQUNEO0FBSkQsQUFLQTtHQUFBLEFBQUUsS0FBRixBQUFPLE9BQVAsQUFBYyxBQUNkO0dBQUEsQUFBRSxlQUFGLEFBQWlCLEFBQ2pCO0dBQUEsQUFBRSxBQUNGOzs7QUFFRCxTQUFBLEFBQVMsSUFBVCxBQUFhLE1BQWIsQUFBbUIsYUFBbkIsQUFBZ0MsTUFBaEMsQUFBc0MsVUFBVSxBQUMvQztLQUFJLElBQUksSUFBUixBQUFRLEFBQUksQUFDWjtHQUFBLEFBQUUscUJBQXFCLFlBQVksQUFDbEM7TUFBSSxFQUFBLEFBQUUsZUFBZSxlQUFqQixBQUFnQyxRQUFRLEVBQUEsQUFBRSxXQUE5QyxBQUF5RCxLQUFLLEFBQzdEO09BQUEsQUFBSSxVQUFVLFNBQVMsRUFBVCxBQUFXLEFBQ3pCO0FBQ0Q7QUFKRCxBQUtBO0dBQUEsQUFBRSxLQUFGLEFBQU8sT0FBUCxBQUFjLEFBQ2Q7S0FBQSxBQUFJLFVBQVUsRUFBQSxBQUFFLGVBQUYsQUFBaUIsQUFDL0I7R0FBQSxBQUFFLGlCQUFGLEFBQW1CLGdCQUFuQixBQUFtQyxBQUNuQztHQUFBLEFBQUUsS0FBRixBQUFPLEFBQ1A7OztBQUVELFNBQUEsQUFBUyxLQUFULEFBQWMsTUFBZCxBQUFvQixhQUFwQixBQUFpQyxNQUFNLEFBQ3RDO0tBQUksSUFBSSxJQUFSLEFBQVEsQUFBSSxBQUNaO0dBQUEsQUFBRSxxQkFBcUIsWUFBWSxBQUNsQztNQUFJLEVBQUEsQUFBRSxlQUFlLGVBQWpCLEFBQWdDLFFBQVEsRUFBQSxBQUFFLFdBQTlDLEFBQXlELEtBQUssQUFDN0Q7QUFDQTtBQUNEO0FBSkQsQUFLQTtHQUFBLEFBQUUsS0FBRixBQUFPLFFBQVAsQUFBZSxBQUNmO0tBQUksZ0JBQUosQUFBb0IsV0FDbkIsRUFBQSxBQUFFLGlCQUFGLEFBQW1CLGdCQUFuQixBQUFtQyxBQUNwQztLQUFJLFNBQUosQUFBYSxXQUNaLEVBQUEsQUFBRSxLQURILEFBQ0MsQUFBTyxXQUVQLEVBQUEsQUFBRSxBQUNIOzs7QUFFRDtBQUNBO0FBQ0E7QUFDQTs7O0FBR0E7Ozs7Ozs7Ozs7QUFVQSxDQUFDLFNBQUEsQUFBUyxPQUFPLEFBQ2hCO0tBQUksTUFBSixBQUFVO0tBQVYsQUFDQztLQURELEFBRUM7S0FGRCxBQUdDO0tBQ0E7YUFBUSxBQUNJLE1BQU0sQUFDakI7WUFGTyxBQUVHLE1BQU8sQUFDakI7VUFITyxBQUdDLE1BQVEsQUFDaEI7V0FKTyxBQUlFLE1BQVEsQUFDakI7V0FMTyxBQUtFLEtBVFgsQUFJUyxBQUtTLEFBR2xCO0FBUlMsQUFDUDs7U0FPTSxNQUFBLEFBQU0sSUFBZCxBQUFRLEFBQVUsQUFFbEI7O1VBQUEsQUFBUyxPQUFULEFBQWdCLGFBQWEsQUFFNUI7O01BQUksWUFBWSxJQUFBLEFBQUksYUFBSixBQUFpQixhQUFqQixBQUE4QixHQUE5QixBQUFpQyxHQUFqRCxBQUFnQixBQUFvQztNQUNuRCxXQUFXLElBQUEsQUFBSSxhQURoQixBQUNZLEFBQWlCO01BRDdCLEFBRUM7TUFGRCxBQUdDO01BSEQsQUFJQztNQUpELEFBS0M7TUFMRCxBQU1DO01BTkQsQUFPQyxBQUVEOztVQUFBLEFBQVEsSUFBUixBQUFZLEFBQ1o7UUFBQSxBQUFNLFdBQVcsT0FBQSxBQUFPLFlBQXhCLEFBQWlCLEFBQW1CLEFBRXBDOztTQUFPLElBQUEsQUFBSSxhQUFKLEFBQWlCLGFBQXhCLEFBQU8sQUFBOEIsQUFHckM7O01BQUksYUFBSixBQUFpQixHQUFHLEFBQUU7QUFDckI7Z0JBQUEsQUFBYSxBQUNiO09BQUksTUFBSixBQUFVLEFBQ1Y7YUFBVSxLQUFBLEFBQUssU0FBTCxBQUFjLEdBQXhCLEFBQVUsQUFBaUIsQUFDM0I7U0FBTSxLQUFBLEFBQUssS0FBSyxJQUFBLEFBQUksT0FBSixBQUFXLEdBQVgsQUFBYyxNQUpYLEFBSW5CLEFBQWdCLEFBQW9CLElBQUksQUFDeEM7VUFBTyxFQUFQLEFBQVMsQUFDVDs7T0FDSSxLQUFBLEFBQUssU0FBTCxBQUFjLEdBRFYsQUFDSixBQUFpQixBQUNwQjtPQUFHLEtBQUEsQUFBSyxTQUZULEFBQVEsQUFFSixBQUFjLEFBR2xCO0FBTFEsQUFDUDs7U0FJRCxBQUFNLEtBQU4sQUFBVyxBQUVYO0FBYkQsU0FhTyxBQUFFO0FBQ1I7QUFDQTtTQUFNLEtBQUEsQUFBSyxLQUFLLElBQUEsQUFBSSxPQUFKLEFBQVcsR0FBWCxBQUFjLE1BRnhCLEFBRU4sQUFBZ0IsQUFBb0IsSUFBSSxBQUN4Qzs7T0FDSSxLQUFBLEFBQUssU0FBTCxBQUFjLEdBQUcsRUFEYixBQUNKLEFBQW1CLEFBQ3RCO09BQUcsS0FBQSxBQUFLLFNBRlQsQUFBUSxBQUVKLEFBQWMsQUFFbEI7QUFKUSxBQUNQO0FBS0Y7O0FBQ0E7UUFBQSxBQUFNLFNBQVMsT0FBQSxBQUFPLFlBQXRCLEFBQWUsQUFBbUIsQUFDbEM7UUFBQSxBQUFNLE1BQU0sSUFBWixBQUFnQixlQUFlLElBQS9CLEFBQW1DLFlBQVksTUFBL0MsQUFBcUQsR0FBRyxNQUF4RCxBQUE4RCxHQUFHLFVBQUEsQUFBUyxTQUFULEFBQWtCLFVBQVUsQUFDNUY7T0FBSSxJQUFKLEFBQVE7T0FBRyxNQUFYLEFBQWlCO09BQUksSUFBSSxJQUFBLEFBQUksYUFBN0IsQUFBeUIsQUFBaUIsQUFDMUM7U0FBQSxBQUFNLFVBQVUsT0FBQSxBQUFPLFlBQXZCLEFBQWdCLEFBQW1CLEFBQ25DO0FBQ0E7QUFDQTtPQUFJLGVBQWUsSUFBbkIsQUFBdUIsSUFBdkIsQUFBMkIsZUFBM0IsQUFBMEMsU0FBMUMsQUFBbUQsQUFDbkQ7T0FBSSxPQUFBLEFBQU8sWUFBWCxBQUFJLEFBQW1CLEFBQ3ZCO1VBQU8sSUFBQSxBQUFJLGtCQUFYLEFBQTZCLEFBQzdCO1VBQU8sV0FBUCxBQUFrQixBQUNsQjtVQUFPLE1BQUEsQUFBTSxZQUFiLEFBQXlCLEFBQ3pCO1VBQU8sTUFBQSxBQUFNLFdBQWIsQUFBd0IsQUFDeEI7VUFBTyxNQUFBLEFBQU0sU0FBYixBQUFzQixBQUN0QjtVQUFPLE1BQUEsQUFBTSxVQUFiLEFBQXVCLEFBQ3ZCO0FBQ0E7T0FBSSxXQUFXLElBQWYsQUFBbUIsSUFBbkIsQUFBdUIsUUFBdkIsQUFBK0IsQUFDL0I7U0FBQSxBQUFNLFlBQU4sQUFBa0IsQUFDbEI7T0FBQSxBQUFJLEFBQ0o7QUFqQkQsQUFrQkE7QUFFRDs7QUFFQTs7QUFDQTtLQUFBLEFBQUksV0FBSixBQUFlLG9CQUFvQixVQUFBLEFBQVMsV0FBVyxBQUN0RDtRQUFNLEtBQUEsQUFBSyxNQUFYLEFBQU0sQUFBVyxBQUVqQjs7VUFBUSxJQUFBLEFBQUksTUFBSixBQUFVLEtBQWxCLEFBQVEsQUFBZSxBQUN2QjtTQUFBLEFBQU8saUJBQWlCLFlBQVcsQUFDbEM7UUFBSyxhQUFhLElBQWxCLEFBQXNCLElBQXRCLEFBQTBCLEFBQzFCO0FBRkQsQUFHQTtRQUFBLEFBQU0sWUFBWSxPQUFBLEFBQU8sWUFBekIsQUFBa0IsQUFBbUIsQUFDckM7TUFBSSxlQUFlLElBQW5CLEFBQXVCLElBQXZCLEFBQTJCLGVBQTNCLEFBQTBDLEFBQzFDO0FBVEQsQUFVQTtBQXpGRDs7Ozs7QUM3REEsSUFBSSxTQUFTLFFBQWIsQUFBYSxBQUFRO0lBQ3BCLFFBQVEsUUFEVCxBQUNTLEFBQVE7O0FBRWpCLE9BQUEsQUFBTyxVQUFVLFVBQUEsQUFBUyxZQUFULEFBQXFCLFdBQVcsQUFDaEQ7O1dBQ1UsTUFBQSxBQUFNLFlBRFQsQUFDRyxBQUFrQixBQUMzQjtZQUFVLE9BQUEsQUFBTyxZQUZsQixBQUFPLEFBRUksQUFBbUIsQUFFOUI7QUFKTyxBQUNOO0FBRkY7Ozs7O0FDSEEsSUFBSSxTQUFTLFFBQWIsQUFBYSxBQUFROztBQUVyQixJQUFJLFFBQVEsU0FBUixBQUFRLE1BQUEsQUFBUyxPQUFULEFBQWdCLFFBQVEsQUFDbkM7TUFBQSxBQUFLLFNBQVMsSUFBQSxBQUFJLE1BQU0sTUFBQSxBQUFNLE9BQTlCLEFBQWMsQUFBdUIsQUFDckM7TUFBQSxBQUFLLE9BQUwsQUFBWSxBQUNaO01BQUEsQUFBSyxPQUFMLEFBQVksQUFDWjtNQUFBLEFBQUssUUFBTCxBQUFhLEFBQ2I7TUFBQSxBQUFLLEtBQUwsQUFBVSxBQUVWOztBQUNBO0FBUkQ7QUFTQSxNQUFBLEFBQU0sVUFBTixBQUFnQixNQUFNLFVBQUEsQUFBUyxPQUFPLEFBQ3JDO0tBQUksU0FBSixBQUFhO0tBQ1osSUFBSSxDQURMLEFBQ00sQUFDTjtRQUFPLEVBQUEsQUFBRSxJQUFJLEtBQUEsQUFBSyxPQUFsQixBQUF5QixRQUN4QjtXQUFTLEtBQUEsQUFBSyxPQUFMLEFBQVksR0FBWixBQUFlLElBRHpCLEFBQ0MsQUFBUyxBQUFtQjtBQUM3QjtBQUxEO0FBTUEsTUFBQSxBQUFNLFVBQU4sQUFBZ0IsVUFBVSxVQUFBLEFBQVMsUUFBUSxBQUMxQztBQUNBO0FBQ0E7S0FBSSxJQUFJLENBQVIsQUFBUyxBQUNUO1FBQU8sRUFBQSxBQUFFLElBQUksS0FBQSxBQUFLLE9BQWxCLEFBQXlCLFFBQVEsQUFDaEM7V0FBUyxLQUFBLEFBQUssT0FBTCxBQUFZLEdBQVosQUFBZSxJQUF4QixBQUFTLEFBQW1CLEFBQzVCO0FBQ0E7QUFDRDtRQUFBLEFBQU8sQUFDUDtBQVREO0FBVUEsTUFBQSxBQUFNLFVBQU4sQUFBZ0IsV0FBVyxVQUFBLEFBQVMsUUFBVCxBQUFpQixPQUFPLEFBQ2xEO0FBQ0E7QUFDQTtLQUFJLElBQUksS0FBQSxBQUFLLE9BQUwsQUFBWSxTQUFwQixBQUE2QixBQUM3QjtRQUFPLE1BQVAsQUFBYSxHQUFHLEFBQ2Y7V0FBUyxLQUFBLEFBQUssT0FBTCxBQUFZLEdBQVosQUFBZSxNQUFmLEFBQXFCLFFBQTlCLEFBQVMsQUFBNkIsQUFDdEM7QUFDQTtBQUNEO0FBUkQ7O0FBVUEsTUFBQSxBQUFNLFVBQU4sQUFBZ0IsV0FBVyxVQUFBLEFBQVMsT0FBVCxBQUFnQixRQUFoQixBQUF3QixVQUFVLEFBQzVEO0tBQUksU0FBSixBQUFhO0tBQ1osWUFBWSxLQUFBLEFBQUssT0FBTyxLQUFBLEFBQUssT0FBTCxBQUFZLFNBRHJDLEFBQ2EsQUFBaUMsQUFDOUM7VUFBUyxLQUFBLEFBQUssUUFBZCxBQUFTLEFBQWEsQUFFdEI7O0FBQ0E7VUFBUyxVQUFBLEFBQVUsTUFBbkIsQUFBUyxBQUFnQixBQUN6QjtLQUFJLE9BQUEsQUFBTyxhQUFYLEFBQXdCLFlBQVksU0FBUyxVQUFULEFBQW1CLEFBRXZEO0FBVEQ7O0FBV0EsTUFBQSxBQUFNLFVBQU4sQUFBZ0IsUUFBUSxVQUFBLEFBQVMsT0FBVCxBQUFnQixZQUFoQixBQUE0QixPQUE1QixBQUFtQyxRQUFuQyxBQUEyQyxVQUFVLEFBQzVFO0tBQUEsQUFBSTtLQUNILElBREQsQUFDSztLQUNKLFlBQVksS0FBQSxBQUFLLE9BQU8sS0FBQSxBQUFLLE9BQUwsQUFBWSxTQUZyQyxBQUVhLEFBQWlDLEFBQzlDO1FBQU8sTUFBUCxBQUFhLFlBQVksQUFDeEI7V0FBQSxBQUFTLEFBQ1Q7V0FBUyxLQUFBLEFBQUssUUFBZCxBQUFTLEFBQWEsQUFFdEI7O0FBQ0E7QUFDQTtXQUFTLFVBQUEsQUFBVSxNQUFuQixBQUFTLEFBQWdCLEFBQ3pCO09BQUEsQUFBSyxPQUFPLFVBQVosQUFBc0IsQUFDdEI7VUFBQSxBQUFRLElBQUksZUFBZSxVQUEzQixBQUFxQyxBQUVyQzs7T0FBQSxBQUFLLFNBQUwsQUFBYyxRQUFkLEFBQXNCLEFBRXRCOztBQUNBO01BQUksT0FBTyxLQUFQLEFBQVksbUJBQWhCLEFBQW1DLFlBQVksS0FBQSxBQUFLLGVBQUwsQUFBb0IsTUFBcEIsQUFBMEIsQUFFekU7O0FBQ0E7QUFDRDtLQUFJLE9BQUEsQUFBTyxhQUFYLEFBQXdCLFlBQVksU0FBUyxLQUFULEFBQVMsQUFBSyxRQUFRLEtBQXRCLEFBQTJCLEFBQy9EO0FBdEJEO0FBdUJBLE1BQUEsQUFBTSxVQUFOLEFBQWdCLE9BQU8sWUFBVyxBQUNqQztBQUNBO0tBQUksVUFBVSxJQUFBLEFBQUksYUFBYSxLQUEvQixBQUFjLEFBQXNCLEFBRXBDOztLQUFJLElBQUksQ0FBUixBQUFTO0tBQ1IsSUFERCxBQUNLLEFBQ0w7QUFDQTtRQUFPLEVBQUEsQUFBRSxJQUFLLEtBQUEsQUFBSyxPQUFMLEFBQVksU0FBMUIsQUFBbUMsR0FBSSxBQUN0QztVQUFBLEFBQVEsSUFBSyxLQUFBLEFBQUssT0FBTCxBQUFZLEdBQXpCLEFBQWEsQUFBZSxRQUE1QixBQUFvQyxBQUNwQztPQUFLLEtBQUEsQUFBSyxPQUFMLEFBQVksR0FBakIsQUFBb0IsQUFDcEI7QUFDRDtBQUNBO1FBQU8sUUFBUCxBQUFlLEFBQ2Y7QUFiRDtBQWNBLE1BQUEsQUFBTSxVQUFOLEFBQWdCLE9BQU8sVUFBQSxBQUFTLFFBQVEsQUFDdkM7QUFDQTtLQUFJLFNBQUosQUFBYTtLQUFiLEFBQ0M7S0FDQSxJQUFJLENBRkwsQUFFTSxBQUdOOztNQUFBLEFBQUssT0FBTCxBQUFZLEFBQ1o7S0FBSSxVQUFBLEFBQVUsUUFBUSxFQUFFLGtCQUF4QixBQUFzQixBQUFvQixlQUFlLEFBQ3hEO1dBQVMsSUFBQSxBQUFJLGFBQWIsQUFBUyxBQUFpQixBQUMxQjtBQUNEO1FBQU8sRUFBQSxBQUFFLElBQUssS0FBQSxBQUFLLE9BQUwsQUFBWSxTQUExQixBQUFtQyxHQUFJLEFBQ3RDO1VBQVEsS0FBQSxBQUFLLE1BQUwsQUFBVyxPQUFuQixBQUFRLEFBQWtCLEFBQzFCO1VBQVEsSUFBSSxPQUFPLE1BQVgsQUFBSSxBQUFhLE1BQWpCLEFBQXVCLE9BQS9CLEFBQVEsQUFBOEIsQUFDdEM7T0FBQSxBQUFLLFFBQVEsTUFBYixBQUFtQixBQUNuQjtNQUFJLFVBQUosQUFBYyxNQUNiLFNBQVMsTUFBQSxBQUFNLEtBQU4sQUFBVyxRQURyQixBQUNDLEFBQVMsQUFBbUIsYUFDeEIsTUFBQSxBQUFNLEFBQ1g7T0FBQSxBQUFLLE9BQUwsQUFBWSxLQUFaLEFBQWlCLEFBQ2pCO0FBQ0Q7QUFDQTtTQUFRLEtBQUEsQUFBSyxNQUFMLEFBQVcsT0FBbkIsQUFBUSxBQUFrQixBQUMxQjtTQUFRLElBQUksT0FBTyxNQUFYLEFBQUksQUFBYSxNQUFqQixBQUF1QixPQUEvQixBQUFRLEFBQThCLEFBQ3RDO01BQUEsQUFBSyxPQUFMLEFBQVksS0FBWixBQUFpQixBQUVqQjtBQXpCRDs7QUEyQkEsT0FBQSxBQUFPLFVBQVUsVUFBQSxBQUFTLFlBQVQsQUFBcUIsV0FBVyxBQUNoRDtVQUFTLE9BQUEsQUFBTyxZQUFoQixBQUFTLEFBQW1CLEFBQzVCO1FBQUEsQUFBTyxBQUNQO0FBSEQ7Ozs7O0FDaEhBLE9BQUEsQUFBTzs7WUFDTSxBQUlYO0FBQ0E7QUFDQTtBQUNBO1VBUFcsQUFVWDtXQVZXLEFBYVg7YUFiVyxBQWdCWDtVQWhCVyxBQW1CWDtjQW5CVyxBQXNCWDthQXZCZSxBQUNKLEFBOEJaO0FBOUJZLEFBQ1g7O1lBNkJXLEFBSVg7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtVQVhXLEFBa0JYO1dBbEJXLEFBeUJYO2FBekJXLEFBNEJYO1VBNUJXLEFBK0JYO2NBL0JXLEFBa0NYO2FBakVGLEFBQWlCLEFBK0JKO0FBQUEsQUFDWDtBQWhDZSxBQUNoQjs7Ozs7QUNERCxJQUFJLFVBQVUsUUFBZCxBQUFjLEFBQVE7SUFBdEIsQUFDQztJQURELEFBRUM7SUFFQSxPQUpEO0lBV0MsT0FYRDs7QUEyQkEsU0FBQSxBQUFTLGVBQWUsQUFDdkI7QUFDQTtNQUFBLEFBQUssT0FBTCxBQUFZLEFBRVo7O0FBQ0E7TUFBQSxBQUFLLFFBQUwsQUFBYSxBQUViOztNQUFBLEFBQUssT0FBTyxJQUFJLEdBQUosQUFBTyxhQUFQLEFBQW9CLElBQUksQ0FBcEMsQUFBWSxBQUF3QixBQUFDLEFBQ3JDO01BQUEsQUFBSyxTQUFMLEFBQWMsQUFDZDs7QUFDRCxhQUFBLEFBQWEsVUFBYixBQUF1QixTQUFTLFVBQUEsQUFBUyxRQUFULEFBQWlCLFFBQVEsQUFDeEQ7S0FBSSxrQkFBSixBQUFzQixjQUNyQixTQUFTLElBQUksR0FBSixBQUFPLE9BQVAsQUFBYyxJQUFJLFFBQUEsQUFBUyxRQUFRLE9BQTVDLEFBQVMsQUFBa0IsQUFBd0IsQUFFcEQ7O0FBRUE7O01BQUEsQUFBSyxTQUFTLElBQUksR0FBSixBQUFPLGFBQVAsQUFBb0IsSUFBSSxPQUF0QyxBQUFjLEFBQStCLEFBQzdDO01BQUEsQUFBSyxPQUFMLEFBQVksSUFBSSxLQUFoQixBQUFxQixNQUFNLEVBQUUsR0FBRixBQUFLLFFBQVEsR0FBeEMsQUFBMkIsQUFBZ0IsQUFDM0M7QUFFQTs7TUFBQSxBQUFLLEtBQUwsQUFBVSxJQUFJLEtBQWQsQUFBbUIsT0FBTyxFQUFFLEdBQUYsQUFBSyxRQUFRLEdBQXZDLEFBQTBCLEFBQWdCLEFBQzFDO0FBRUE7O1FBQU8sS0FBUCxBQUFZLEFBQ1o7QUFkRDs7QUFnQkEsT0FBQSxBQUFPLFVBQVUsVUFBQSxBQUFTLFlBQVQsQUFBcUIsV0FBVyxBQUNoRDtNQUFBLEFBQUssQUFDTDtNQUFBLEFBQUssQUFDTDtRQUFBLEFBQU8sQUFDUDtBQUpEOzs7OztBQ3JEQSxJQUFJLFVBQVUsUUFBZCxBQUFjLEFBQVE7SUFBdEIsQUFDQztJQURELEFBRUM7SUFFQSxRQUFRLFFBSlQsQUFJUyxBQUFRO0lBQ2hCLGtCQUFrQixRQUxuQixBQUttQixBQUFRO0lBRTFCLGdCQVBEO0lBc0JDLGtCQXRCRDtJQWlDQyxpQkFqQ0Q7SUE4Q0MsbUJBOUNEO0lBeURDLFVBekREO0lBMEVDLGFBQWEsU0FBYixBQUFhLFdBQUEsQUFBQyxvQkFBRDs0SUFBQSxBQUtSLHFCQUxRO0FBMUVkO0lBbUZDLFdBQVcsU0FBWCxBQUFXLFNBQUEsQUFBQyxvQkFBRDtpT0FBQSxBQU9OLHFCQVBNO0FBbkZaOztBQWtHQSxTQUFBLEFBQVMsTUFBVCxBQUFlLE9BQWYsQUFBc0IsT0FBTyxBQUM1QjtNQUFBLEFBQUssSUFBTCxBQUFTLEFBQ1Q7QUFDQTtNQUFBLEFBQUssVUFBVSxNQUFBLEFBQU0sT0FBTixBQUFhLGdCQUE1QixBQUE0QyxBQUU1Qzs7TUFBQSxBQUFLLGFBQWEsV0FBVyxNQUFBLEFBQU0sV0FBVyxNQUE5QyxBQUFrQixBQUFXLEFBQXVCLEFBRXBEOztBQUNBO01BQUEsQUFBSyxXQUFXLE1BQUEsQUFBTSxPQUFOLEFBQWEsaUJBQTdCLEFBQThDLEFBQzlDO01BQUEsQUFBSyxXQUFXLFNBQVMsTUFBQSxBQUFNLFdBQVcsTUFBMUMsQUFBZ0IsQUFBUyxBQUF1QixBQUNoRDtBQUNBO01BQUEsQUFBSyxTQUFMLEFBQWMsQUFFZDs7TUFBQSxBQUFLLFFBQVEsTUFBYixBQUFtQixBQUNuQjtNQUFBLEFBQUssUUFBTCxBQUFhLEFBQ2I7TUFBQSxBQUFLLFNBQUwsQUFBYyxBQUNkO01BQUEsQUFBSyxpQkFBTCxBQUFzQixBQUN0QjtNQUFBLEFBQUssVUFBTCxBQUFlLEFBQ2Y7TUFBQSxBQUFLLE9BQU8sTUFBWixBQUFrQixBQUNsQjtNQUFBLEFBQUssT0FBTyxLQUFBLEFBQUssTUFBTCxBQUFXLEtBQUssS0FBQSxBQUFLLE1BQXJCLEFBQWdCLEFBQVcsTUFBTSxLQUFBLEFBQUssT0FBTyxLQUFBLEFBQUssTUFBakIsQUFBWSxBQUFXLEtBQXBFLEFBQVksQUFBNkQsQUFFekU7O0FBQ0QsTUFBQSxBQUFNLFVBQU4sQUFBZ0IsT0FBTyxVQUFBLEFBQVMsT0FBVCxBQUFnQixRQUFRLEFBQzlDO0tBQUksU0FBUyxLQUFiLEFBQWtCLEFBQ2xCO0FBQ0E7TUFBQSxBQUFLLFVBQVUsSUFBSSxHQUFKLEFBQU8sY0FBUCxBQUFxQixJQUFJLFFBQVMsTUFBQSxBQUFNLFNBQU4sQUFBZSxRQUFRLFNBQWhDLEFBQVMsQUFBZ0MsU0FBUyxDQUFDLEtBQUEsQUFBSyxNQUFOLEFBQUMsQUFBVyxJQUFJLEtBQUEsQUFBSyxNQUFMLEFBQVcsTUFBTSxLQUFBLEFBQUssT0FBTCxBQUFZLElBQXZJLEFBQWUsQUFBeUIsQUFBa0QsQUFBZ0IsQUFBaUMsQUFDM0k7V0FBQSxBQUFVLEFBQ1Y7UUFBQSxBQUFPLEFBQ1A7QUFORDtBQU9BLE1BQUEsQUFBTSxVQUFOLEFBQWdCLGdCQUFnQixZQUFXLEFBQzFDO01BQUEsQUFBSyxjQUFjLEdBQUosQUFBTyxjQUFQLEFBQXFCLFlBRWxDLGdCQUFnQixLQUFoQixBQUFxQixPQUFRLEtBQUEsQUFBSyxPQUFPLEtBQUEsQUFBSyxNQUFqQixBQUFZLEFBQVcsS0FEckQsQUFDQyxBQUF5RCxJQUFLLEFBQzlEO0VBQUMsS0FBQSxBQUFLLE1BQU4sQUFBQyxBQUFXLElBQUksS0FBQSxBQUFLLE1BQUwsQUFBVyxNQUFNLEtBQUEsQUFBSyxPQUFMLEFBQVksSUFGOUMsQUFFQyxBQUFnQixBQUFpQyxJQUhuRCxBQUFlLEFBQ2QsQUFFc0QsQUFHdkQ7QUFMQyxFQURjO0FBRGhCO0FBUUEsTUFBQSxBQUFNLFVBQU4sQUFBZ0IsT0FBTyxZQUFXLEFBQ2pDO1FBQU8sS0FBQSxBQUFLLFFBQUwsQUFBYSxPQUFwQixBQUEyQixBQUMzQjtBQUZEO0FBR0EsTUFBQSxBQUFNLFVBQU4sQUFBZ0IsTUFBTSxVQUFBLEFBQVMsT0FBTyxBQUNyQztLQUFJLGlCQUFKLEFBQXFCLGNBQWMsQUFDbEM7T0FBQSxBQUFLLFFBQVEsSUFBSSxHQUFKLEFBQU8sT0FBUCxBQUFjLElBQUksUUFBQSxBQUFTLE9BQU8sQ0FBRSxLQUFBLEFBQUssTUFBUCxBQUFFLEFBQVcsSUFBSyxNQUFBLEFBQU0sU0FBUyxLQUFBLEFBQUssTUFBckIsQUFBZ0IsQUFBVyxNQUEzRixBQUFhLEFBQWtCLEFBQWdCLEFBQW1ELEFBQ2xHO0FBRkQsUUFFTyxLQUFBLEFBQUssUUFBTCxBQUFhLEFBQ3BCO0FBQ0E7QUFDQTtBQUVBOztNQUFBLEFBQUssaUJBQWlCLElBQUksR0FBSixBQUFPLGFBQVAsQUFBb0IsSUFBSSxDQUFFLEtBQUEsQUFBSyxNQUFQLEFBQUUsQUFBVyxJQUFJLEtBQUEsQUFBSyxNQUFMLEFBQVcsTUFBMUUsQUFBc0IsQUFBd0IsQUFBaUIsQUFBaUIsQUFDaEY7TUFBQSxBQUFLLGVBQUwsQUFBb0IsSUFBSSxLQUF4QixBQUE2QixTQUFTLEVBQUMsR0FBRyxLQUFKLEFBQVMsU0FBUyxHQUFHLEtBQTNELEFBQXNDLEFBQTBCLEFBRWhFOztBQUVBOztNQUFBLEFBQUssU0FBUyxJQUFJLEdBQUosQUFBTyxhQUFQLEFBQW9CLElBQUksQ0FBRSxLQUFBLEFBQUssTUFBUCxBQUFFLEFBQVcsSUFBSSxLQUFBLEFBQUssTUFBTCxBQUFXLE1BQWxFLEFBQWMsQUFBd0IsQUFBaUIsQUFBaUIsQUFDeEU7TUFBQSxBQUFLLE9BQUwsQUFBWSxJQUFJLEtBQWhCLEFBQXFCLFlBQVksRUFBQyxHQUFHLEtBQXJDLEFBQWlDLEFBQVMsQUFFMUM7O0FBQ0E7UUFBTyxLQUFQLEFBQVksQUFDWjtBQWxCRDtBQW1CQSxNQUFBLEFBQU0sVUFBTixBQUFnQixRQUFRLFVBQUEsQUFBUyxPQUFULEFBQWdCLGVBQWUsQUFDdEQ7TUFBQSxBQUFLLFVBQVUsSUFBSSxHQUFKLEFBQU8sYUFBUCxBQUFvQixJQUFJLEtBQUEsQUFBSyxNQUE1QyxBQUFlLEFBQW1DLEFBQ2xEO01BQUEsQUFBSyxRQUFRLElBQUksR0FBSixBQUFPLGFBQVAsQUFBb0IsSUFBSSxLQUFBLEFBQUssT0FBMUMsQUFBYSxBQUFvQyxBQUVqRDs7QUFDQTtBQUVBOztBQUNBO01BQUEsQUFBSyxNQUFMLEFBQVcsSUFBSSxLQUFmLEFBQW9CLFVBQVUsRUFBQyxHQUFELEFBQUksT0FBTyxHQUFHLEtBQWQsQUFBbUIsUUFBUSxHQUFHLEtBQTVELEFBQThCLEFBQW1DLEFBQ2pFO0FBRUE7O0FBQ0E7TUFBQSxBQUFLLFFBQUwsQUFBYSxJQUFJLEtBQWpCLEFBQXNCLFVBQVUsRUFBQyxHQUFHLEtBQUosQUFBUyxPQUFPLEdBQUcsS0FBbkQsQUFBZ0MsQUFBd0IsQUFFeEQ7O0FBQ0E7TUFBQSxBQUFLLFFBQUwsQUFBYSxJQUFJLEtBQWpCLEFBQXNCLFFBQVEsRUFBQyxHQUFHLEtBQUosQUFBUyxTQUFTLEdBQUcsS0FBckIsQUFBMEIsT0FBTyxHQUFHLEtBQXBDLEFBQXlDLE9BQU8sR0FBOUUsQUFBOEIsQUFBbUQsQUFDakY7QUFFQTs7UUFBTyxLQUFQLEFBQVksQUFDWjtBQW5CRDs7QUFxQkEsT0FBQSxBQUFPLFVBQVUsVUFBQSxBQUFTLFlBQVQsQUFBcUIsV0FBVyxBQUNoRDtNQUFBLEFBQUssQUFDTDtNQUFBLEFBQUssQUFDTDtRQUFBLEFBQU8sQUFDUDtBQUpEOzs7OztBQ2xMQSxJQUFJLFVBQVUsUUFBZCxBQUFjLEFBQVE7SUFBdEIsQUFDQztJQURELEFBRUM7SUFFQSxPQUpEO0lBWUMsV0FaRDtJQXdCQyxPQXhCRDs7QUF1Q0EsU0FBQSxBQUFTLE1BQU0sQUFDZDtBQUNBO01BQUEsQUFBSyxPQUFMLEFBQVksQUFFWjs7QUFDQTtNQUFBLEFBQUssUUFBTCxBQUFhLEFBRWI7O01BQUEsQUFBSyxXQUFMLEFBQWdCLEFBRWhCOztNQUFBLEFBQUssT0FBTyxJQUFJLEdBQUosQUFBTyxhQUFQLEFBQW9CLElBQUksQ0FBcEMsQUFBWSxBQUF3QixBQUFDLEFBQ3JDO01BQUEsQUFBSyxTQUFMLEFBQWMsQUFDZDs7QUFDRCxJQUFBLEFBQUksVUFBSixBQUFjLFNBQVMsVUFBQSxBQUFTLFFBQVQsQUFBaUIsUUFBUSxBQUMvQztLQUFJLGtCQUFKLEFBQXNCLGNBQ3JCLFNBQVMsSUFBSSxHQUFKLEFBQU8sT0FBUCxBQUFjLElBQUksUUFBQSxBQUFTLFFBQVEsT0FBNUMsQUFBUyxBQUFrQixBQUF3QixBQUVwRDs7QUFFQTs7TUFBQSxBQUFLLFNBQVMsSUFBSSxHQUFKLEFBQU8sYUFBUCxBQUFvQixJQUFJLE9BQXRDLEFBQWMsQUFBK0IsQUFDN0M7TUFBQSxBQUFLLE9BQUwsQUFBWSxJQUFJLEtBQWhCLEFBQXFCLE1BQU0sRUFBRSxHQUFGLEFBQUssUUFBUSxHQUF4QyxBQUEyQixBQUFnQixBQUMzQztBQUVBOztNQUFBLEFBQUssS0FBTCxBQUFVLElBQUksS0FBZCxBQUFtQixPQUFPLEVBQUUsR0FBRyxLQUEvQixBQUEwQixBQUFVLEFBRXBDOztRQUFPLEtBQVAsQUFBWSxBQUNaO0FBYkQ7O0FBZUEsT0FBQSxBQUFPLFVBQVUsVUFBQSxBQUFTLFlBQVQsQUFBcUIsV0FBVyxBQUNoRDtNQUFBLEFBQUssQUFDTDtNQUFBLEFBQUssQUFDTDtRQUFBLEFBQU8sQUFDUDtBQUpEOzs7OztBQ2xFQSxJQUFJLFVBQVUsUUFBZCxBQUFjLEFBQVE7SUFBdEIsQUFDQztJQURELEFBRUM7SUFDQSxVQUFVLFFBSFgsQUFHVyxBQUFRO0lBQ2xCLE1BQU0sUUFKUCxBQUlPLEFBQVE7SUFDZCxlQUFlLFFBTGhCLEFBS2dCLEFBQVE7SUFFdkIsU0FQRDs7QUFrQkEsU0FBQSxBQUFTLE9BQVQsQUFBZ0IsT0FBaEIsQUFBdUIsT0FBTzthQUM3Qjs7TUFBQSxBQUFLLFNBQUwsQUFBYyxBQUNkO0tBQUksTUFBQSxBQUFNLGVBQU4sQUFBcUIsYUFBYSxNQUFBLEFBQU0sU0FBNUMsQUFBcUQsWUFBWSxBQUNoRTtPQUFBLEFBQUssU0FBUyxJQUFBLEFBQUksUUFBSixBQUFZLE9BQTFCLEFBQWMsQUFBbUIsQUFDakM7T0FBQSxBQUFLLE1BQU0sVUFBQSxBQUFDLE9BQVUsQUFDckI7U0FBQSxBQUFLLFVBQVUsTUFBQSxBQUFLLE9BQUwsQUFBWSxJQUEzQixBQUFlLEFBQWdCLEFBQy9CO1VBQU8sTUFBUCxBQUFZLEFBQ1o7QUFIRCxBQUlBO0FBTkQsUUFNTyxBQUNOO1VBQVEsTUFBUixBQUFjLEFBQ2I7UUFBQSxBQUFLLEFBQ0o7U0FBQSxBQUFLLFNBQVMsSUFBZCxBQUFjLEFBQUksQUFDbEI7QUFDRDtRQUFBLEFBQUssQUFDSjtTQUFBLEFBQUssU0FBUyxJQUFkLEFBQWMsQUFBSSxBQUNsQjtBQU5GLEFBUUE7O09BQUEsQUFBSyxNQUFNLEtBQUEsQUFBSyxJQUFMLEFBQVMsS0FBcEIsQUFBVyxBQUFjLEFBQ3pCO09BQUEsQUFBSyxRQUFRLEtBQUEsQUFBSyxNQUFMLEFBQVcsS0FBeEIsQUFBYSxBQUFnQixBQUM3QjtBQUVEOztNQUFBLEFBQUssVUFBTCxBQUFlLEFBQ2Y7TUFBQSxBQUFLLFdBQUwsQUFBZ0IsQUFDaEI7O0FBQ0QsT0FBQSxBQUFPLFVBQVAsQUFBaUIsTUFBTSxVQUFBLEFBQVMsT0FBTyxBQUN0QztNQUFBLEFBQUssVUFBTCxBQUFlLEFBQ2Y7UUFBQSxBQUFPLEFBQ1A7QUFIRDtBQUlBLE9BQUEsQUFBTyxVQUFQLEFBQWlCLFFBQVEsVUFBQSxBQUFTLFVBQVUsQUFDM0M7QUFDQTtLQUFJLG9CQUFKLEFBQXdCLGNBQ3ZCLFdBQVcsSUFBSSxHQUFKLEFBQU8sT0FBUCxBQUFjLElBQUksUUFBQSxBQUFTLFVBQVUsS0FBQSxBQUFLLFFBQXJELEFBQVcsQUFBa0IsQUFBZ0MsQUFHOUQ7O0FBRUE7O01BQUEsQUFBSyxnQkFBZ0IsSUFBSSxHQUFKLEFBQU8sYUFBUCxBQUFvQixJQUFJLENBQUEsQUFBQyxHQUFHLEtBQUEsQUFBSyxRQUFMLEFBQWEsTUFBOUQsQUFBcUIsQUFBd0IsQUFBSSxBQUFtQixBQUNwRTtNQUFBLEFBQUssWUFBWSxJQUFJLEdBQUosQUFBTyxhQUFQLEFBQW9CLElBQUksQ0FBekMsQUFBaUIsQUFBd0IsQUFBQyxBQUMxQztNQUFBLEFBQUssY0FBTCxBQUFtQixJQUFJLEtBQUEsQUFBSyxPQUE1QixBQUFtQyxVQUFVLEVBQUUsR0FBRyxLQUFMLEFBQVUsU0FBUyxHQUFoRSxBQUE2QyxBQUFzQixBQUNuRTtNQUFBLEFBQUssVUFBTCxBQUFlLElBQWYsQUFBbUIsUUFBUSxFQUFFLEdBQUcsS0FBaEMsQUFBMkIsQUFBVSxBQUNyQztNQUFBLEFBQUssV0FBVyxLQUFBLEFBQUssVUFBTCxBQUFlLE9BQWYsQUFBc0IsS0FBdEMsQUFBZ0IsQUFBMkIsQUFFM0M7O1FBQU8sS0FBQSxBQUFLLE9BQUwsQUFBWSxPQUFPLEtBQW5CLEFBQXdCLFNBQS9CLEFBQU8sQUFBaUMsQUFDeEM7QUFmRDs7QUFrQkEsT0FBQSxBQUFPLFVBQVUsVUFBQSxBQUFTLFlBQVQsQUFBcUIsV0FBVyxBQUVoRDs7TUFBQSxBQUFLLEFBQ0w7TUFBQSxBQUFLLEFBRUw7O1dBQVUsUUFBQSxBQUFRLFlBQWxCLEFBQVUsQUFBb0IsQUFDOUI7T0FBTSxJQUFBLEFBQUksWUFBVixBQUFNLEFBQWdCLEFBQ3RCO2dCQUFlLGFBQUEsQUFBYSxZQUE1QixBQUFlLEFBQXlCLEFBRXhDOztRQUFBLEFBQU8sQUFDUDtBQVZEOzs7OztBQ2hFQSxJQUFJLFVBQVUsUUFBZCxBQUFjLEFBQVE7SUFBdEIsQUFDQztJQURELEFBRUM7SUFFQSxNQUpEO0lBY0MsT0FkRDtJQXNCQyxXQXRCRDs7O0FBa0NDO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOzs7QUFHQSxPQXZERDs7QUF1RUEsU0FBQSxBQUFTLFFBQVQsQUFBaUIsT0FBakIsQUFBd0IsT0FBTyxBQUM5QjtNQUFBLEFBQUssSUFBTCxBQUFTLEFBRVQ7O01BQUEsQUFBSyxhQUFMLEFBQWtCLEFBRWxCOztNQUFBLEFBQUssV0FBTCxBQUFnQixBQUNoQjtNQUFBLEFBQUssUUFBTCxBQUFhLEFBRWI7O01BQUEsQUFBSyxXQUFMLEFBQWdCLEFBR2hCOztNQUFBLEFBQUssUUFBTCxBQUFhLEFBQ2I7TUFBQSxBQUFLLFNBQUwsQUFBYyxBQUNkO01BQUEsQUFBSyxPQUFPLElBQUksR0FBSixBQUFPLGFBQVAsQUFBb0IsSUFBSSxDQUFwQyxBQUFZLEFBQXdCLEFBQUMsQUFDckM7O0FBQ0QsUUFBQSxBQUFRLFVBQVIsQUFBa0IsTUFBTSxVQUFBLEFBQVMsT0FBTyxBQUV2Qzs7TUFBQSxBQUFLLFFBQUwsQUFBYSxBQUNiO01BQUEsQUFBSyxTQUFTLElBQUksR0FBSixBQUFPLGFBQVAsQUFBb0IsSUFBSSxLQUFBLEFBQUssTUFBM0MsQUFBYyxBQUFtQyxBQUVqRDs7TUFBQSxBQUFLLE9BQUwsQUFBWSxJQUFJLEtBQWhCLEFBQXFCLFlBQVksRUFBQyxHQUFHLEtBQXJDLEFBQWlDLEFBQVMsQUFFMUM7O0FBQ0E7UUFBTyxLQUFQLEFBQVksQUFDWjtBQVREO0FBVUEsUUFBQSxBQUFRLFVBQVIsQUFBa0IsU0FBUyxVQUFBLEFBQVMsUUFBVCxBQUFpQixVQUFVLEFBQ3JEO01BQUEsQUFBSyxVQUFVLElBQUksR0FBSixBQUFPLGFBQVAsQUFBb0IsSUFBSSxLQUFBLEFBQUssTUFBNUMsQUFBZSxBQUFtQyxBQUVsRDs7S0FBSSxvQkFBSixBQUF3QixjQUN2QixXQUFXLElBQUksR0FBSixBQUFPLE9BQVAsQUFBYyxJQUFJLFFBQUEsQUFBUSxVQUFVLEtBQUEsQUFBSyxNQUFwRCxBQUFXLEFBQWtCLEFBQTZCLEFBRTNEOztBQUNBO01BQUEsQUFBSyxRQUFMLEFBQWEsSUFBSSxLQUFqQixBQUFzQixVQUFVLEVBQUMsR0FBRCxBQUFJLFFBQVEsR0FBNUMsQUFBZ0MsQUFBZSxBQUUvQzs7QUFDQTtNQUFBLEFBQUssS0FBTCxBQUFVLElBQUksS0FBZCxBQUFtQixPQUFPLEVBQUUsR0FBRyxLQUEvQixBQUEwQixBQUFVLEFBQ3BDO0FBRUE7O0FBRUE7O1FBQU8sS0FBUCxBQUFZLEFBQ1o7QUFoQkQ7O0FBa0JBLE9BQUEsQUFBTyxVQUFVLFVBQUEsQUFBUyxZQUFULEFBQXFCLFdBQVcsQUFDaEQ7TUFBQSxBQUFLLEFBQ0w7TUFBQSxBQUFLLEFBQ0w7UUFBQSxBQUFPLEFBQ1A7QUFKRDs7Ozs7QUNsSEE7O0FBQ0EsU0FBQSxBQUFTLE9BQVQsQUFBZ0IsTUFBaEIsQUFBc0I7UUFDZCxRQUFQLEFBQWUsQUFDZjtVQUFTLFVBQVQsQUFBbUIsQUFDaEI7S0FBSSxJQUFKLEFBQVE7S0FBRyxJQUFYLEFBQWUsQUFDZjtRQUFNLE1BQU4sQUFBWSxHQUFHO01BQUksS0FBbkIsQUFBZSxBQUFJLEFBQUs7QUFKRSxFQUFBLEFBQzdCLENBR3FDLEFBQ2xDO1FBQU0sTUFBTixBQUFZLEdBQUc7TUFBSSxLQUFuQixBQUFlLEFBQUksQUFBSztBQUxFLEdBTTFCLEFBQ0E7UUFBUSxLQUFBLEFBQUssS0FBTSxDQUFBLEFBQUMsTUFBTSxLQUFBLEFBQUssSUFBdkIsQUFBa0IsQUFBVSxNQUFRLEtBQUEsQUFBSyxJQUFLLE1BQU0sS0FBTixBQUFXLEtBQTFELEFBQXFDLEFBQTBCLEtBQS9ELEFBQXNFLFNBQTdFLEFBQXNGLEFBQ3pGOzs7QUFFRCxPQUFBLEFBQU8sVUFBVSxTQUFBLEFBQVMsZ0JBQVQsQUFBeUIsT0FBekIsQUFBZ0MsTUFBTSxBQUN0RDtLQUFJLFNBQVMsSUFBQSxBQUFJLGFBQWEsTUFBQSxBQUFNLEtBQUssTUFBWCxBQUFXLEFBQU0sS0FBL0MsQUFBYSxBQUF1QyxBQUNwRDtTQUFBLEFBQVEsSUFBSSwyQkFBMkIsT0FBdkMsQUFBOEMsQUFDOUM7S0FBSSxJQUFJLENBQVIsQUFBUyxBQUNUO1FBQU8sRUFBQSxBQUFFLElBQUksT0FBYixBQUFvQixRQUFRLEFBQzNCO1NBQUEsQUFBTyxLQUFLLE9BQUEsQUFBTyxHQUFHLEtBQUEsQUFBSyxLQUFLLElBQUksTUFBcEMsQUFBWSxBQUFVLEFBQWMsQUFBTSxBQUMxQztBQUNEO0FBQ0E7UUFBQSxBQUFPLEFBQ1A7QUFURDs7O0FDWEE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUNyRkE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUNWQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUNyQkE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FDdERBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FDdlZBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FB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K0IsQ0FBL0IsRUFBa0MsSUFBbEMsRUFBdUM7O0FBRTdDLE1BQUssQ0FBTCxJQUFVLEtBQUssS0FBTCxDQUFXLE1BQU0sS0FBSyxHQUFMLENBQVMsQ0FBVCxFQUFZLEtBQUssR0FBTCxDQUFTLENBQVQsRUFBWSxDQUFDLElBQUksS0FBSyxLQUFMLENBQVcsQ0FBWCxDQUFMLEtBQXFCLEtBQUssS0FBTCxDQUFXLENBQVgsSUFBZ0IsS0FBSyxLQUFMLENBQVcsQ0FBWCxDQUFyQyxDQUFaLENBQVosQ0FBakIsQ0FBVjtBQUNBLE1BQUssQ0FBTCxJQUFVLEtBQUssS0FBTCxDQUFXLE1BQU0sS0FBSyxHQUFMLENBQVMsQ0FBVCxFQUFZLEtBQUssR0FBTCxDQUFTLENBQVQsRUFBWSxDQUFDLElBQUksS0FBSyxLQUFMLENBQVcsQ0FBWCxDQUFMLEtBQXFCLEtBQUssS0FBTCxDQUFXLENBQVgsSUFBZ0IsS0FBSyxLQUFMLENBQVcsQ0FBWCxDQUFyQyxDQUFaLENBQVosQ0FBakIsQ0FBVjtBQUNBLE1BQUssQ0FBTCxJQUFVLEtBQUssS0FBTCxDQUFXLE1BQU0sS0FBSyxHQUFMLENBQVMsQ0FBVCxFQUFZLEtBQUssR0FBTCxDQUFTLENBQVQsRUFBWSxDQUFDLElBQUksS0FBSyxLQUFMLENBQVcsQ0FBWCxDQUFMLEtBQXFCLEtBQUssS0FBTCxDQUFXLENBQVgsSUFBZ0IsS0FBSyxLQUFMLENBQVcsQ0FBWCxDQUFyQyxDQUFaLENBQVosQ0FBakIsQ0FBVjtBQUNBLE1BQUssQ0FBTCxJQUFVLEtBQUssS0FBTCxDQUFXLE1BQU0sS0FBSyxHQUFMLENBQVMsQ0FBVCxFQUFZLEtBQUssR0FBTCxDQUFTLENBQVQsRUFBWSxDQUFDLElBQUksS0FBSyxLQUFMLENBQVcsQ0FBWCxDQUFMLEtBQXFCLEtBQUssS0FBTCxDQUFXLENBQVgsSUFBZ0IsS0FBSyxLQUFMLENBQVcsQ0FBWCxDQUFyQyxDQUFaLENBQVosQ0FBakIsQ0FBVjtBQUNBO0FBQ0E7O0FBR00sU0FBUyxNQUFULENBQWdCLElBQWhCLEVBQXNCLENBQXRCLEVBQXlC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QUcsa0JBQUgsQ0FBc0IsR0FBRyxZQUF6QixFQUF1QyxPQUF2QyxDQUFiLEVBQThELElBQUksS0FBSixHQUFZLENBQVosR0FBZ0IsQ0FBOUU7QUFDQSxPQUFHLFNBQUgsQ0FBYSxHQUFHLGtCQUFILENBQXNCLEdBQUcsWUFBekIsRUFBdUMsVUFBdkMsQ0FBYixFQUFpRSxJQUFJLFFBQUosSUFBZ0IsQ0FBakY7O0FBRUEsT0FBRyxlQUFILENBQW1CLEdBQUcsV0FBdEIsRUFBbUMsSUFBbkM7QUFDQSxPQUFHLFFBQUgsQ0FBWSxDQUFaLEVBQWUsQ0FBZixFQUFrQixHQUFHLGtCQUFyQixFQUF5QyxHQUFHLG1CQUE1QztBQUNBLE9BQUcsVUFBSCxDQUFjLEdBQUcsY0FBakIsRUFBaUMsQ0FBakMsRUFBb0MsQ0FBcEM7QUFFSDs7Ozs7Ozs7UUNuRmUsUSxHQUFBLFE7QUFBVCxTQUFTLFFBQVQsQ0FBa0IsTUFBbEIsRUFBeUI7QUFDNUIsUUFBRyxDQUFDLE1BQUosRUFBVztBQUNQLGlCQUFTLFNBQVMsYUFBVCxDQUF1QixRQUF2QixDQUFUO0FBQ0EsZUFBTyxLQUFQLEdBQWUsR0FBZjtBQUNBLGVBQU8sTUFBUCxHQUFnQixHQUFoQjtBQUNBLGVBQU8sS0FBUCxDQUFhLE9BQWIsR0FBdUIsTUFBdkI7QUFDQSxlQUFPLE9BQVAsR0FBaUIsSUFBakI7QUFDQSxpQkFBUyxJQUFULENBQWMsV0FBZCxDQUEwQixNQUExQjtBQUNIO0FBQ0QsUUFBSSxLQUFLLE9BQU8sVUFBUCxDQUFrQixPQUFsQixFQUEyQixFQUFFLFdBQVcsS0FBYixFQUEzQixLQUNBLE9BQU8sVUFBUCxDQUFrQixvQkFBbEIsRUFBd0MsRUFBRSxXQUFXLEtBQWIsRUFBeEMsQ0FEVDtBQUVBLFFBQUksQ0FBQyxFQUFMLEVBQVMsTUFBTSxpREFBTjtBQUNULFdBQU8sRUFBUDtBQUNIIiwiZmlsZSI6ImdlbmVyYXRlZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzQ29udGVudCI6WyIoZnVuY3Rpb24gZSh0LG4scil7ZnVuY3Rpb24gcyhvLHUpe2lmKCFuW29dKXtpZighdFtvXSl7dmFyIGE9dHlwZW9mIHJlcXVpcmU9PVwiZnVuY3Rpb25cIiYmcmVxdWlyZTtpZighdSYmYSlyZXR1cm4gYShvLCEwKTtpZihpKXJldHVybiBpKG8sITApO3ZhciBmPW5ldyBFcnJvcihcIkNhbm5vdCBmaW5kIG1vZHVsZSAnXCIrbytcIidcIik7dGhyb3cgZi5jb2RlPVwiTU9EVUxFX05PVF9GT1VORFwiLGZ9dmFyIGw9bltvXT17ZXhwb3J0czp7fX07dFtvXVswXS5jYWxsKGwuZXhwb3J0cyxmdW5jdGlvbihlKXt2YXIgbj10W29dWzFdW2VdO3JldHVybiBzKG4/bjplKX0sbCxsLmV4cG9ydHMsZSx0LG4scil9cmV0dXJuIG5bb10uZXhwb3J0c312YXIgaT10eXBlb2YgcmVxdWlyZT09XCJmdW5jdGlvblwiJiZyZXF1aXJlO2Zvcih2YXIgbz0wO288ci5sZW5ndGg7bysrKXMocltvXSk7cmV0dXJuIHN9KSIsInZhciBNb2RlbCA9IHJlcXVpcmUoXCIuLi9saWIvTW9kZWxcIiksXG5cdFRGID0gcmVxdWlyZShcIi4uL25vZGVfbW9kdWxlcy90ZW5zb3JmaXJlL3NyYy9pbmRleFwiKSxcblx0R0wgPSBURi5jcmVhdGVHTCgpO1xuXG5mdW5jdGlvbiBHRVQocGF0aCwgcmVzcG9uc2VUeXBlLCBjYWxsYmFjaykge1xuXHR2YXIgciA9IG5ldyBYTUxIdHRwUmVxdWVzdCgpO1xuXHRyLm9ucmVhZHlzdGF0ZWNoYW5nZSA9IGZ1bmN0aW9uICgpIHtcblx0XHRpZiAoci5yZWFkeVN0YXRlID09PSBYTUxIdHRwUmVxdWVzdC5ET05FICYmIHIuc3RhdHVzID09PSAyMDApIHtcblx0XHRcdGNhbGxiYWNrKHIucmVzcG9uc2UpO1xuXHRcdH1cblx0fTtcblx0ci5vcGVuKFwiR0VUXCIsIHBhdGgpO1xuXHRyLnJlc3BvbnNlVHlwZSA9IHJlc3BvbnNlVHlwZTtcblx0ci5zZW5kKCk7XG59XG5cbmZ1bmN0aW9uIFBVVChwYXRoLCBjb250ZW50VHlwZSwgYm9keSwgY2FsbGJhY2spIHtcblx0dmFyIHIgPSBuZXcgWE1MSHR0cFJlcXVlc3QoKTtcblx0ci5vbnJlYWR5c3RhdGVjaGFuZ2UgPSBmdW5jdGlvbiAoKSB7XG5cdFx0aWYgKHIucmVhZHlTdGF0ZSA9PT0gWE1MSHR0cFJlcXVlc3QuRE9ORSAmJiByLnN0YXR1cyA9PT0gMjAwKSB7XG5cdFx0XHRpZiAoY2FsbGJhY2spIGNhbGxiYWNrKHIucmVzcG9uc2UpO1xuXHRcdH1cblx0fVxuXHRyLm9wZW4oXCJQVVRcIiwgcGF0aCk7XG5cdGlmIChjYWxsYmFjaykgci5yZXNwb25zZVR5cGUgPSBjb250ZW50VHlwZTtcblx0ci5zZXRSZXF1ZXN0SGVhZGVyKFwiQ29udGVudC1UeXBlXCIsIGNvbnRlbnRUeXBlKTtcblx0ci5zZW5kKGJvZHkpO1xufVxuXG5mdW5jdGlvbiBQT1NUKHBhdGgsIGNvbnRlbnRUeXBlLCBib2R5KSB7XG5cdHZhciByID0gbmV3IFhNTEh0dHBSZXF1ZXN0KCk7XG5cdHIub25yZWFkeXN0YXRlY2hhbmdlID0gZnVuY3Rpb24gKCkge1xuXHRcdGlmIChyLnJlYWR5U3RhdGUgPT09IFhNTEh0dHBSZXF1ZXN0LkRPTkUgJiYgci5zdGF0dXMgIT09IDIwMCkge1xuXHRcdFx0Ly8gVE9ETyAtIHJlc2VuZCBvciBzYXZlIHRvIGxvY2FsP1xuXHRcdH1cblx0fVxuXHRyLm9wZW4oXCJQT1NUXCIsIHBhdGgpO1xuXHRpZiAoY29udGVudFR5cGUgIT09IHVuZGVmaW5lZClcblx0XHRyLnNldFJlcXVlc3RIZWFkZXIoXCJDb250ZW50LVR5cGVcIiwgY29udGVudFR5cGUpO1xuXHRpZiAoYm9keSAhPT0gdW5kZWZpbmVkKVxuXHRcdHIuc2VuZChib2R5KTtcblx0ZWxzZVxuXHRcdHIuc2VuZCgpO1xufVxuXG4vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vL1xuLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy9cbi8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vXG4vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vLy8vL1xuXG5cbi8qXG5cblx0MS4gR2V0IG1vZGVsIGZyb20gc2VydmVyXG5cdDIuIEdldCB3ZWlnaHRzIGZyb20gc2VydmVyXG5cdDMuIEdldCBkYXRhIGZyb20gc2VydmVyXG5cdDQuIFRyYWluIGFuZCByZXR1cm4gdXBkYXRlc1xuXG5cbiovXG5cbihmdW5jdGlvbiBtYWluKCkge1xuXHR2YXIgcnVuID0gdHJ1ZSxcblx0XHRuZXQsXG5cdFx0bW9kZWwsXG5cdFx0aXRlcmF0aW9ucyxcblx0XHR0aW1lcyA9IHtcblx0XHRcdHJlcXVlc3RlZDogbnVsbCxcdC8vIG1vZGVsIHdlaWdodHMgYW5kIGRhdGEgcmVxdWVzdCBzZW50XG5cdFx0XHRyZWNlaXZlZDogbnVsbCxcdFx0Ly8gbW9kZWwgZGF0YSAmIHdlaWdodHMgcmVjZWl2ZWQgZnJvbSBzZXJ2ZXJcblx0XHRcdGxvYWRlZDogbnVsbCwgXHRcdC8vIG1vZGVsIHdlaWdodHMgbG9hZGVkXG5cdFx0XHR0cmFpbmVkOiBudWxsLCBcdFx0Ly8gbW9kZWwgZmluaXNoZWQgdHJhaW5pbmdcblx0XHRcdHVwZGF0ZWQ6IG51bGwgXHRcdC8vIG1vZGVsIHVwZGF0ZXMgc2VudFxuXHRcdH07XG5cblx0TW9kZWwgPSBNb2RlbChURiwgR0wpO1xuXG5cdGZ1bmN0aW9uIHVwZGF0ZShhcnJheWJ1ZmZlcikge1xuXG5cdFx0dmFyIGl0ZXJhdGlvbiA9IG5ldyBGbG9hdDMyQXJyYXkoYXJyYXlidWZmZXIsIDAsIDEpWzBdLFxuXHRcdFx0dGVzdFZpZXcgPSBuZXcgRmxvYXQzMkFycmF5KGFycmF5YnVmZmVyKSxcblx0XHRcdHZpZXcsXG5cdFx0XHR3ZWlnaHRzLFxuXHRcdFx0ZGF0YSxcblx0XHRcdGxlbixcblx0XHRcdGksXG5cdFx0XHRiYXRjaDtcblxuXHRcdGNvbnNvbGUubG9nKHRlc3RWaWV3KTtcblx0XHR0aW1lcy5yZWNlaXZlZCA9IHdpbmRvdy5wZXJmb3JtYW5jZS5ub3coKTtcblxuXHRcdHZpZXcgPSBuZXcgRmxvYXQzMkFycmF5KGFycmF5YnVmZmVyLCA0KTtcblxuXG5cdFx0aWYgKGl0ZXJhdGlvbiA+PSAwKSB7IC8vIGluY2x1ZGVzIG5ldyB3ZWlnaHRzIGFuZCBkYXRhXG5cdFx0XHRpdGVyYXRpb25zID0gaXRlcmF0aW9uO1xuXHRcdFx0aSA9IG1vZGVsLnNpemU7XG5cdFx0XHR3ZWlnaHRzID0gdmlldy5zdWJhcnJheSgwLCBpKTtcblx0XHRcdGxlbiA9IHZpZXdbaV0gKiBuZXQubGF5ZXJzWzBdLnNoYXBlWzFdOyAvLyBmaXJzdCBmbG9hdCBpcyBudW1iZXIgb2Ygc2FtcGxlcyBpbiB0aGlzIGJhdGNoXG5cdFx0XHRsZW4gKz0gKytpO1xuXHRcdFx0YmF0Y2ggPSB7XG5cdFx0XHRcdHg6IHZpZXcuc3ViYXJyYXkoaSwgbGVuKSxcblx0XHRcdFx0eTogdmlldy5zdWJhcnJheShsZW4pXG5cdFx0XHR9O1xuXG5cdFx0XHRtb2RlbC5sb2FkKHdlaWdodHMpO1xuXG5cdFx0fSBlbHNlIHsgLy8gd2VpZ2h0cyBhcmUgZnJlc2gsIHNvIGRhdGEgb25seVxuXHRcdFx0aXRlcmF0aW9ucysrO1xuXHRcdFx0bGVuID0gdmlld1swXSAqIG5ldC5sYXllcnNbMF0uc2hhcGVbMV07IC8vIGZpcnN0IGZsb2F0IGlzIG51bWJlciBvZiBzYW1wbGVzIGluIHRoaXMgYmF0Y2hcblx0XHRcdGJhdGNoID0ge1xuXHRcdFx0XHR4OiB2aWV3LnN1YmFycmF5KDEsICsrbGVuKSxcblx0XHRcdFx0eTogdmlldy5zdWJhcnJheShsZW4pXG5cdFx0XHR9O1xuXHRcdH1cblxuXHRcdC8vIFRSQUlOXG5cdFx0dGltZXMubG9hZGVkID0gd2luZG93LnBlcmZvcm1hbmNlLm5vdygpO1xuXHRcdG1vZGVsLnRyYWluKG5ldC5sZWFybmluZ19yYXRlLCBuZXQuaXRlcmF0aW9ucywgYmF0Y2gueCwgYmF0Y2gueSwgZnVuY3Rpb24od2VpZ2h0cywgYWNjdXJhY3kpIHtcblx0XHRcdHZhciByID0gMCwgbG9nID0gXCJcIiwgdyA9IG5ldyBGbG9hdDMyQXJyYXkod2VpZ2h0cyk7XG5cdFx0XHR0aW1lcy50cmFpbmVkID0gd2luZG93LnBlcmZvcm1hbmNlLm5vdygpO1xuXHRcdFx0Ly9jb25zb2xlLmxvZyhcIlRpbWUgdG8gdHJhaW46IFwiICsgKGRlbHRhIC8gMTAwMCkgKyBcIiBzZWNvbmRzXCIpO1xuXHRcdFx0Ly8gcG9zdCByZXN1bHRzIHRvIHNlcnZlclxuXHRcdFx0UFVUKFwiLi93ZWlnaHRzL1wiICsgbmV0LmlkLCBcImFycmF5YnVmZmVyXCIsIHdlaWdodHMsIHVwZGF0ZSk7XG5cdFx0XHRyID0gd2luZG93LnBlcmZvcm1hbmNlLm5vdygpO1xuXHRcdFx0bG9nICs9IG5ldC53ZWlnaHRzX3ZlcnNpb24gKyBcIixcIjtcblx0XHRcdGxvZyArPSBhY2N1cmFjeSArIFwiLFwiO1xuXHRcdFx0bG9nICs9IHRpbWVzLnJlcXVlc3RlZCArIFwiLFwiO1xuXHRcdFx0bG9nICs9IHRpbWVzLnJlY2VpdmVkICsgXCIsXCI7XG5cdFx0XHRsb2cgKz0gdGltZXMubG9hZGVkICsgXCIsXCI7XG5cdFx0XHRsb2cgKz0gdGltZXMudHJhaW5lZCArIFwiXFxuXCI7XG5cdFx0XHQvLyBzZW5kIHRpbWUgYW5kIHRyYWluaW5nIGxvZyB0byBzZXJ2ZXJcblx0XHRcdFBVVChcIi4vbG9nL1wiICsgbmV0LmlkLCBcInRleHRcIiwgbG9nKTtcblx0XHRcdHRpbWVzLnJlcXVlc3RlZCA9IHI7XG5cdFx0XHRuZXQud2VpZ2h0c192ZXJzaW9uKys7XG5cdFx0fSk7XG5cdH1cblxuXHQvL3ZhciBzZXJ2ZXIgPSBpbygpO1xuXG5cdC8vIHJlcXVlc3QgbW9kZWwgdG8gdHJhaW5cblx0R0VUKFwiLi9tb2RlbFwiLCBcImFwcGxpY2F0aW9uL2pzb25cIiwgZnVuY3Rpb24oanNvbk1vZGVsKSB7XG5cdFx0bmV0ID0gSlNPTi5wYXJzZShqc29uTW9kZWwpO1xuXG5cdFx0bW9kZWwgPSBuZXcgTW9kZWwobmV0LCBudWxsKTtcblx0XHR3aW5kb3cub25iZWZvcmV1bmxvYWQgPSBmdW5jdGlvbigpIHtcblx0XHRcdFBPU1QoXCIuL2Nsb3NlL1wiICsgbmV0LmlkLCBcInN0cmluZ1wiKVxuXHRcdH07XG5cdFx0dGltZXMucmVxdWVzdGVkID0gd2luZG93LnBlcmZvcm1hbmNlLm5vdygpO1xuXHRcdEdFVChcIi4vd2VpZ2h0cy9cIiArIG5ldC5pZCwgXCJhcnJheWJ1ZmZlclwiLCB1cGRhdGUpO1xuXHR9KTtcbn0pKCk7IiwidmFyIE91dHB1dCA9IHJlcXVpcmUoJy4vbGF5ZXJzL091dHB1dCcpLFxuXHREZW5zZSA9IHJlcXVpcmUoJy4vbGF5ZXJzL0RlbnNlJyk7XG5cbm1vZHVsZS5leHBvcnRzID0gZnVuY3Rpb24odGVuc29yZmlyZSwgZ2xDb250ZXh0KSB7XG5cdHJldHVybiB7XG5cdFx0XCJkZW5zZVwiOiBEZW5zZSh0ZW5zb3JmaXJlLCBnbENvbnRleHQpLFxuXHRcdFwib3V0cHV0XCI6IE91dHB1dCh0ZW5zb3JmaXJlLCBnbENvbnRleHQpXG5cdH07XG59OyIsInZhciBMYXllcnMgPSByZXF1aXJlKFwiLi9MYXllcnNcIik7XG5cbnZhciBNb2RlbCA9IGZ1bmN0aW9uKG1vZGVsLCBsYXllcnMpIHtcblx0dGhpcy5sYXllcnMgPSBuZXcgQXJyYXkobW9kZWwubGF5ZXJzLmxlbmd0aCk7XG5cdHRoaXMubG9zcyA9IDAuMDtcblx0dGhpcy5zaXplID0gMC4wO1xuXHR0aGlzLm1vZGVsID0gbW9kZWw7XG5cdHRoaXMubG9hZChsYXllcnMpO1xuXG5cdC8vY29uc29sZS5sb2coSlNPTi5zdHJpbmdpZnkodGhpcy5sYXllcnNbMF0uc2F2ZSgpKSk7XG59O1xuTW9kZWwucHJvdG90eXBlLnJ1biA9IGZ1bmN0aW9uKGlucHV0KSB7XG5cdHZhciBvdXRwdXQgPSBpbnB1dCxcblx0XHRsID0gLTE7XG5cdHdoaWxlICgrK2wgPCB0aGlzLmxheWVycy5sZW5ndGgpXG5cdFx0b3V0cHV0ID0gdGhpcy5sYXllcnNbbF0ucnVuKG91dHB1dCk7XG59O1xuTW9kZWwucHJvdG90eXBlLmZvcndhcmQgPSBmdW5jdGlvbihvdXRwdXQpIHtcblx0Ly9jb25zb2xlLndhcm4oXCJDYWxjdWxvbi0gRm9yd2FyZCBwYXNzXFxuXCIpO1xuXHQvLyBmb3J3YXJkIHByb3BvZ2F0aW9uXG5cdHZhciBsID0gLTE7XG5cdHdoaWxlICgrK2wgPCB0aGlzLmxheWVycy5sZW5ndGgpIHtcblx0XHRvdXRwdXQgPSB0aGlzLmxheWVyc1tsXS5ydW4ob3V0cHV0KTtcblx0XHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIG91dHB1dCBcIiArIGwgKyBcIjogXCIgKyBvdXRwdXQucmVhZCgpLmRhdGEpO1xuXHR9XG5cdHJldHVybiBvdXRwdXQ7XG59O1xuTW9kZWwucHJvdG90eXBlLmJhY2t3YXJkID0gZnVuY3Rpb24ob3V0cHV0LCBsZWFybikge1xuXHQvL2NvbnNvbGUud2FybihcIkNhbGN1bG9uLSBCYWNrd2FyZCBwYXNzXCIpO1xuXHQvLyBiYWNrd2FyZCBwcm9wb2dhdGlvblxuXHR2YXIgbCA9IHRoaXMubGF5ZXJzLmxlbmd0aCAtIDE7XG5cdHdoaWxlIChsLS0gPiAwKSB7XG5cdFx0b3V0cHV0ID0gdGhpcy5sYXllcnNbbF0udHJhaW4ob3V0cHV0LCBsZWFybik7XG5cdFx0Ly9jb25zb2xlLmxvZyhvdXRwdXQucmVhZCgpLmRhdGEpO1xuXHR9XG59O1xuXG5Nb2RlbC5wcm90b3R5cGUudmFsaWRhdGUgPSBmdW5jdGlvbihpbnB1dCwgZXhwZWN0LCBjYWxsYmFjaykge1xuXHR2YXIgb3V0cHV0ID0gaW5wdXQsXG5cdFx0bG9zc0xheWVyID0gdGhpcy5sYXllcnNbdGhpcy5sYXllcnMubGVuZ3RoIC0gMV07XG5cdG91dHB1dCA9IHRoaXMuZm9yd2FyZChvdXRwdXQpO1xuXG5cdC8vIGNhbGN1bGF0ZSBsb3NzXG5cdG91dHB1dCA9IGxvc3NMYXllci50cmFpbihleHBlY3QpO1xuXHRpZiAodHlwZW9mIGNhbGxiYWNrID09PSBcImZ1bmN0aW9uXCIpIGNhbGxiYWNrKGxvc3NMYXllci5hY2N1cmFjeSlcblxufVxuXG5Nb2RlbC5wcm90b3R5cGUudHJhaW4gPSBmdW5jdGlvbihsZWFybiwgaXRlcmF0aW9ucywgaW5wdXQsIGV4cGVjdCwgY2FsbGJhY2spIHtcblx0dmFyIG91dHB1dCxcblx0XHRlID0gMCxcblx0XHRsb3NzTGF5ZXIgPSB0aGlzLmxheWVyc1t0aGlzLmxheWVycy5sZW5ndGggLSAxXTtcblx0d2hpbGUgKGUrKyA8IGl0ZXJhdGlvbnMpIHtcblx0XHRvdXRwdXQgPSBpbnB1dDtcblx0XHRvdXRwdXQgPSB0aGlzLmZvcndhcmQob3V0cHV0KTtcblxuXHRcdC8vY29uc29sZS5sb2coXCJDYWxjdWxvbi0gb3V0cHV0OiBcIiArIG91dHB1dC5yZWFkKCkuZGF0YSk7XG5cdFx0Ly8gY2FsY3VsYXRlIGxvc3Ncblx0XHRvdXRwdXQgPSBsb3NzTGF5ZXIudHJhaW4oZXhwZWN0KTtcblx0XHR0aGlzLmxvc3MgPSBsb3NzTGF5ZXIuYWNjdXJhY3k7XG5cdFx0Y29uc29sZS5sb2coXCJBY2N1cmFjeTogXCIgKyBsb3NzTGF5ZXIuYWNjdXJhY3kpO1xuXG5cdFx0dGhpcy5iYWNrd2FyZChvdXRwdXQsIGxlYXJuKTtcblxuXHRcdC8vIGNoYW5jZSB0byBzZW5kIG91dCBkYXRhIGZyb20gbW9kZWwgKG1ldGFkYXRhIGFuZCBsb2cgZGF0YSlcblx0XHRpZiAodHlwZW9mIHRoaXMuYWZ0ZXJJdGVyYXRpb24gPT09IFwiZnVuY3Rpb25cIikgdGhpcy5hZnRlckl0ZXJhdGlvbih0aGlzLCBlKTtcblxuXHRcdC8vY29uc29sZS53YXJuKFwiQ2FsY3Vsb24tIEl0ZXJhdGlvbjogXCIgKyBlICsgXCIsIExvc3M6IFwiICsgdGhpcy5sb3NzKTtcblx0fVxuXHRpZiAodHlwZW9mIGNhbGxiYWNrID09PSBcImZ1bmN0aW9uXCIpIGNhbGxiYWNrKHRoaXMuc2F2ZSgpLCB0aGlzLmxvc3MpO1xufVxuTW9kZWwucHJvdG90eXBlLnNhdmUgPSBmdW5jdGlvbigpIHtcblx0Ly8gVHlwZWRBcnJheSB0byBob2xkIHdlaWdodHMsIGJpYXMsIGV0Yy4gZnJvbSBldmVyeSBsYXllciBvZiBtb2RlbFxuXHR2YXIgd2VpZ2h0cyA9IG5ldyBGbG9hdDMyQXJyYXkodGhpcy5zaXplKTtcblx0XG5cdHZhciBsID0gLTEsXG5cdFx0byA9IDA7XG5cdC8vIHB1bGwgb3V0IHRyYWluZWQgd2VpZ2h0cyBmb3IgZWFjaCBsYXllclxuXHR3aGlsZSAoKytsIDwgKHRoaXMubGF5ZXJzLmxlbmd0aCAtIDEpKSB7XG5cdFx0d2VpZ2h0cy5zZXQoIHRoaXMubGF5ZXJzW2xdLnNhdmUoKSwgbyk7XG5cdFx0byArPSB0aGlzLmxheWVyc1tsXS5zaXplO1xuXHR9XG5cdC8vY29uc29sZS5sb2coXCJ3ZWlnaHRzOiBcIiArIHdlaWdodHMpO1xuXHRyZXR1cm4gd2VpZ2h0cy5idWZmZXI7XG59O1xuTW9kZWwucHJvdG90eXBlLmxvYWQgPSBmdW5jdGlvbihsYXllcnMpIHtcblx0Ly8gY29uc3RydWN0IGxheWVyc1xuXHR2YXIgb2Zmc2V0ID0gMCxcblx0XHRsYXllcixcblx0XHRsID0gLTE7XG5cblxuXHR0aGlzLnNpemUgPSAwO1xuXHRpZiAobGF5ZXJzICE9IG51bGwgJiYgIShsYXllcnMgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkpKSB7XG5cdFx0bGF5ZXJzID0gbmV3IEZsb2F0MzJBcnJheShsYXllcnMpO1xuXHR9XG5cdHdoaWxlICgrK2wgPCAodGhpcy5sYXllcnMubGVuZ3RoIC0gMSkpIHtcblx0XHRsYXllciA9IHRoaXMubW9kZWwubGF5ZXJzW2xdO1xuXHRcdGxheWVyID0gbmV3IExheWVyc1tsYXllci50eXBlXShsYXllciwgbCk7XG5cdFx0dGhpcy5zaXplICs9IGxheWVyLnNpemU7XG5cdFx0aWYgKGxheWVycyAhPSBudWxsKVxuXHRcdFx0b2Zmc2V0ID0gbGF5ZXIubG9hZChsYXllcnMsIG9mZnNldCk7XG5cdFx0ZWxzZSBsYXllci5yYW5kb21XZWlnaHRzKCk7XG5cdFx0dGhpcy5sYXllcnNbbF0gPSBsYXllcjtcblx0fVxuXHQvLyBpbml0aWFsaXplIG91dHB1dCBsYXllclxuXHRsYXllciA9IHRoaXMubW9kZWwubGF5ZXJzW2xdO1xuXHRsYXllciA9IG5ldyBMYXllcnNbbGF5ZXIudHlwZV0obGF5ZXIsIGwpO1xuXHR0aGlzLmxheWVyc1tsXSA9IGxheWVyO1xuXG59O1xuXG5tb2R1bGUuZXhwb3J0cyA9IGZ1bmN0aW9uKHRlbnNvcmZpcmUsIGdsQ29udGV4dCkge1xuXHRMYXllcnMgPSBMYXllcnModGVuc29yZmlyZSwgZ2xDb250ZXh0KTtcblx0cmV0dXJuIE1vZGVsO1xufTsiLCJtb2R1bGUuZXhwb3J0cyA9IHtcblx0QWN0aXZhdGlvbjoge1xuXHRcdFwibGluZWFyXCI6IGBcblx0XHRcdG8gPSBuO1xuXHRcdGAsXG5cdFx0Ly8gXCJiaW5hcnlcIjogYFxuXHRcdC8vIFx0aWYgKG4gPiAwLjApIHsgbyA9IDAuMDsgfSBlbHNlIHsgbyA9IDEuMDsgfVxuXHRcdC8vIGAsXG5cdFx0XCJyZWx1XCI6IGBcblx0XHRcdG8gPSBtYXgoMC4wLCBuKTtcblx0XHRgLFxuXHRcdFwibHJlbHVcIjogYFxuXHRcdFx0aWYgKG4gPj0gMC4wKSB7IG8gPSBuOyB9IGVsc2UgeyBvID0gMC4wMSAqIG47IH1cblx0XHRgLFxuXHRcdFwic2lnbW9pZFwiOiBgXG5cdFx0XHRvID0gMS4wIC8gKDEuMCArIGV4cCgwLjAgLSBuKSk7XG5cdFx0YCxcblx0XHRcInRhbmhcIjogYFxuXHRcdFx0byA9ICgyLjAgLyAoMS4wICsgZXhwKC0yLjAgKiBuKSkpIC0gMS4wO1xuXHRcdGAsXG5cdFx0XCJzb2Z0cGx1c1wiOiBgXG5cdFx0XHRvID0gbG9nKDEuMCArIGV4cChuKSk7XG5cdFx0YCxcblx0XHRcInNvZnRtYXhcIjogYFxuXHRcdFx0ZmxvYXQgayA9IDAuMDtcblx0XHRcdGZvcihpbnQgaSA9IDA7IGkgPCAjKE8uc2hhcGUpLng7IGkrKyl7XG5cdFx0XHRcdGsgKz0gZXhwKE8ucmVhZChpLCBwb3MueSkpO1xuXHRcdFx0fVxuXHRcdFx0byA9IGV4cChuKSAvIGs7XG5cdFx0YFxuXHR9LFxuXHREZXJpdmF0aXZlOiB7XG5cdFx0XCJsaW5lYXJcIjogYFxuXHRcdFx0ZCA9IDEuMDtcblx0XHRgLFxuXHRcdC8vIFwiYmluYXJ5XCI6IGBcblx0XHQvLyBcdGlmIChvID09IDAuMCkge1xuXHRcdC8vIFx0XHRkID0gMC4wO1xuXHRcdC8vIFx0fSBlbHNlIHtcblx0XHQvLyBcdFx0ZCA9IDAuMDtcblx0XHQvLyBcdH1cblx0XHQvLyBgLFxuXHRcdFwicmVsdVwiOiBgXG5cdFx0XHRpZiAobyA+PSAwLjApIHtcblx0XHRcdFx0ZCA9IDEuMDtcblx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdGQgPSAwLjA7XG5cdFx0XHR9XG5cdFx0YCxcblx0XHRcImxyZWx1XCI6IGBcblx0XHRcdGlmIChvID49IDAuMCkge1xuXHRcdFx0XHRkID0gMS4wO1xuXHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0ZCA9IDAuMDE7XG5cdFx0XHR9XG5cdFx0YCxcblx0XHRcInNpZ21vaWRcIjogYFxuXHRcdFx0ZCA9IG8gKiAoIDEuMCAtIG8gKTtcblx0XHRgLFxuXHRcdFwidGFuaFwiOiBgXG5cdFx0XHRkID0gKCA0LjAgLyBwb3coKCBleHAoLW8pICsgZXhwKG8pKSwgMi4wKSApO1xuXHRcdGAsXG5cdFx0XCJzb2Z0cGx1c1wiOiBgXG5cdFx0XHRkID0gMS4wIC0gKCAxLjAgLyBleHAobykgKTtcblx0XHRgLFxuXHRcdFwic29mdG1heFwiOiBgXG5cdFx0XHRkID0gbyAqICggMS4wIC0gbyApO1xuXHRcdGAgXG5cdH1cbn07IiwidmFyIG5kYXJyYXkgPSByZXF1aXJlKFwibmRhcnJheVwiKSxcblx0VEYsXG5cdEdMLFxuXG5cdEdyYWQgPSBgXG5cdFx0dW5pZm9ybSBUZW5zb3IgTztcblx0XHR1bmlmb3JtIFRlbnNvciBFO1xuXHRcdGZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7XG5cdFx0XHRyZXR1cm4gMC4wIC0gRS5yZWFkKHBvcykgLyBPLnJlYWQocG9zKTtcblx0XHR9XG5cdGAsXG5cdExvc3MgPSBgXG5cdFx0dW5pZm9ybSBUZW5zb3IgTztcblx0XHR1bmlmb3JtIFRlbnNvciBFO1xuXHRcdGZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7XG5cdFx0XHRmbG9hdCBsb3NzID0gMC4wO1xuXHRcdFx0Zm9yKGludCBpID0gMDsgaSA8ICMoTy5zaGFwZSkueTsgaSsrKXsgLy8gaXRlcmF0ZSBvdmVyIGVhY2ggc2FtcGxlXG5cdFx0XHRcdGZsb2F0IGwgPSAwLjA7XG5cdFx0XHRcdGZvcihpbnQgaiA9IDA7IGogPCAjKE8uc2hhcGUpLng7IGorKyl7IC8vIGl0ZXJhdGUgb3ZlciBldmVyeSBvdXRwdXQgYW5kIGNhbGN1bGF0ZSBhdmVyYWdlXG5cdFx0XHRcdFx0bCAtPSBFLnJlYWQoaiwgaSkgKiBsb2coTy5yZWFkKGosIGkpKTtcblx0XHRcdFx0fVxuXHRcdFx0XHRsb3NzID0gbCAvIGZsb2F0KCMoTy5zaGFwZSkueSk7XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gbG9zcztcblx0XHR9XG5cdGA7XG5cbmZ1bmN0aW9uIENyb3NzRW50cm9weSgpIHtcblx0Ly8gY2FsY3VsYXRlIGxvc3MgZ3JhZGllbnRzXG5cdHRoaXMuZ3JhZCA9IEdyYWQ7XG5cblx0Ly8gY2FsY3VsYXRlIGJhdGNoIGF2ZXJhZ2UgbG9zc1xuXHR0aGlzLmxvc3NGID0gTG9zcztcblxuXHR0aGlzLmxvc3MgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBbMV0pO1xuXHR0aGlzLm91dHB1dCA9IG51bGw7XG59XG5Dcm9zc0VudHJvcHkucHJvdG90eXBlLmRlbHRhcyA9IGZ1bmN0aW9uKG91dHB1dCwgZXhwZWN0KSB7XG5cdGlmIChleHBlY3QgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkpXG5cdFx0ZXhwZWN0ID0gbmV3IFRGLlRlbnNvcihHTCwgbmRhcnJheSggZXhwZWN0LCBvdXRwdXQuc2hhcGUpKTtcblxuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIGV4cGVjdGVkOiBcIiArIGV4cGVjdC5yZWFkKCkuZGF0YSk7XG5cblx0dGhpcy5vdXRwdXQgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBvdXRwdXQuc2hhcGUpO1xuXHR0aGlzLm91dHB1dC5ydW4odGhpcy5ncmFkLCB7IE86IG91dHB1dCwgRTogZXhwZWN0IH0pO1xuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIGdyYWRpZW50OiBcIiArIHRoaXMub3V0cHV0LnJlYWQoKS5kYXRhKTtcblxuXHR0aGlzLmxvc3MucnVuKHRoaXMubG9zc0YsIHsgTzogb3V0cHV0LCBFOiBleHBlY3QgfSk7XG5cdC8vY29uc29sZS5sb2codGhpcy5sb3NzLnJlYWQoKSk7XG5cblx0cmV0dXJuIHRoaXMub3V0cHV0O1xufVxuXG5tb2R1bGUuZXhwb3J0cyA9IGZ1bmN0aW9uKHRlbnNvcmZpcmUsIGdsQ29udGV4dCkge1xuXHRURiA9IHRlbnNvcmZpcmU7XG5cdEdMID0gZ2xDb250ZXh0O1xuXHRyZXR1cm4gQ3Jvc3NFbnRyb3B5O1xufTsiLCJ2YXIgbmRhcnJheSA9IHJlcXVpcmUoXCJuZGFycmF5XCIpLFxuXHRURixcblx0R0wsXG5cblx0RnVuY3MgPSByZXF1aXJlKCcuL0FjdGl2YXRpb25zJyksXG5cdGdlbmVyYXRlV2VpZ2h0cyA9IHJlcXVpcmUoJy4uL3V0aWwvZ2VuZXJhdGVXZWlnaHRzJyksXG5cblx0Rm9yd2FyZEJpYXNlZCA9IGBcblx0XHR1bmlmb3JtIFRlbnNvciBXOyAvLyBsYXllciB3ZWlnaHRzXG5cdFx0dW5pZm9ybSBUZW5zb3IgSTsgLy8gbGF5ZXIgaW5wdXRzXG5cdFx0ZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHsgLy8gZm9yIGVhY2ggdW5pdCBpbiBvdXRwdXQgKHg6IHVuaXQsIHk6IHNhbXBsZSlcblx0XHRcdFx0ZmxvYXQgbiA9IDAuMDtcblx0XHRcdFx0Zm9yKGludCBpID0gMDsgaSA8ICMoVy5zaGFwZSkueTsgaSsrKXsgLy8gZm9yIGVhY2ggd2VpZ2h0XG5cdFx0XHRcdFx0aWYgKGkgPT0gIyhXLnNoYXBlKS55IC0gMSkge1xuXHRcdFx0XHRcdFx0biArPSBXLnJlYWQocG9zLngsIGkpO1xuXHRcdFx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdFx0XHRuICs9IEkucmVhZChpLCBwb3MueSkgKiBXLnJlYWQocG9zLngsIGkpO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0fVxuXHRcdFx0XHRyZXR1cm4gbjtcblx0XHR9XG5cdGAsXG5cdEZvcndhcmRVbmJpYXNlZCA9IGAgLy8gZm9yIGVhY2ggb3V0cHV0IG5vZGVcblx0XHR1bmlmb3JtIFRlbnNvciBXOyAvLyBsYXllciB3ZWlnaHRzXG5cdFx0dW5pZm9ybSBUZW5zb3IgSTsgLy8gbGF5ZXIgaW5wdXRzXG5cdFx0ZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHtcblx0XHRcdGZsb2F0IG4gPSAwLjA7XG5cdFx0XHRmb3IoaW50IGkgPSAwOyBpIDwgIyhXLnNoYXBlKS55OyBpKyspe1xuXHRcdFx0XHRuICs9IEkucmVhZChpLCBwb3MueSkgKiBXLnJlYWQocG9zLngsIGkpO1xuXHRcdFx0fVxuXHRcdFx0cmV0dXJuIG47XG5cdFx0fVxuXHRgLFxuXHRCYWNrd2FyZEJpYXNlZCA9IGAgLy8gZm9yIGVhY2ggaW5wdXQgbm9kZVxuXHRcdHVuaWZvcm0gVGVuc29yIEU7IC8vIGxvY2FsIGVycm9yIChmcm9tIGFjdGl2YXRpb24pXG5cdFx0dW5pZm9ybSBUZW5zb3IgVzsgLy8gd2VpZ2h0c1xuXHRcdGZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7IC8vIHBvc2l0aW9uIGluIGlucHV0IGdyYWRpZW50IFRlbnNvclxuXHRcdFx0ZmxvYXQgZSA9IDAuMDsgLy8gc3VtIG91dHB1dCBlcnJvclxuXHRcdFx0Zm9yKGludCBpID0gMDsgaSA8ICMoRS5zaGFwZSkueDsgaSsrKXtcblx0XHRcdFx0aWYgKHBvcy55ICE9ICMoRS5zaGFwZSkueCkge1xuXHRcdFx0XHRcdGUgKz0gVy5yZWFkKGksIHBvcy54KSAqIEUucmVhZChpLCBwb3MueSk7XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHRcdHJldHVybiBlO1xuXHRcdH1cblx0YCxcblx0QmFja3dhcmRVbmJpYXNlZCA9IGAgLy8gZm9yIGVhY2ggaW5wdXQgbm9kZVxuXHRcdHVuaWZvcm0gVGVuc29yIEU7IC8vIGxvY2FsIGVycm9yIChmcm9tIGFjdGl2YXRpb24pXG5cdFx0dW5pZm9ybSBUZW5zb3IgVzsgLy8gd2VpZ2h0c1xuXHRcdGZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7IC8vIHBvc2l0aW9uIGluIGlucHV0IGdyYWRpZW50IFRlbnNvclxuXHRcdFx0ZmxvYXQgZSA9IDAuMDsgLy8gc3VtIG91dHB1dCBlcnJvclxuXHRcdFx0Zm9yKGludCBpID0gMDsgaSA8ICMoRS5zaGFwZSkueDsgaSsrKXtcblx0XHRcdFx0ZSArPSBXLnJlYWQoaSwgcG9zLngpICogRS5yZWFkKGksIHBvcy55KTtcblx0XHRcdH1cblx0XHRcdHJldHVybiBlO1xuXHRcdH1cblx0YCxcblx0V2VpZ2h0cyA9IGBcblx0XHR1bmlmb3JtIFRlbnNvciBFOyAvLyBsb2NhbCBlcnJvciAoZnJvbSBhY3RpdmF0aW9uKVxuXHRcdHVuaWZvcm0gVGVuc29yIFc7IC8vIHdlaWdodHNcblx0XHR1bmlmb3JtIFRlbnNvciBJOyAvLyBpbnB1dFxuXHRcdHVuaWZvcm0gZmxvYXQgbDsgLy8gbGVhcm5pbmcgcmF0ZVxuXHRcdGZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7IC8vIHBvcyBpbiB3ZWlnaHRzIFRlbnNvclxuXHRcdFx0ZmxvYXQgZSA9IDAuMDsgLy8gYXZnIG5vZGUgYmF0Y2ggZXJyb3Jcblx0XHRcdGZvcihpbnQgaSA9IDA7IGkgPCAjKEUuc2hhcGUpLnk7IGkrKyl7XG5cdFx0XHRcdGlmIChwb3MueSA9PSAjKEkuc2hhcGUpLngpIHsgLy8gaGFuZGxlIGJpYXMgbGF5ZXIgP1xuXHRcdFx0XHRcdGUgKz0gRS5yZWFkKHBvcy54LCBpKTtcblx0XHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0XHRlICs9IEUucmVhZChwb3MueCwgaSkgKiBJLnJlYWQocG9zLnksIGkpO1xuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gVy5yZWFkKHBvcykgLSAobCAqIGUpO1xuXHRcdH1cblx0YCxcblx0QWN0aXZhdGlvbiA9IChhY3RpdmF0aW9uRnVuY3Rpb24pID0+IGBcblx0XHR1bmlmb3JtIFRlbnNvciBPOyAvLyB3ZWlnaHRlZCBpbnB1dFxuXHRcdGZsb2F0IHByb2Nlc3MoaXZlYzQgcG9zKSB7XG5cdFx0XHRmbG9hdCBuID0gTy5yZWFkKHBvcyk7XG5cdFx0XHRmbG9hdCBvO1xuXHRcdFx0JHsgYWN0aXZhdGlvbkZ1bmN0aW9uIH1cblx0XHRcdHJldHVybiBvO1xuXHRcdH1cblx0YCxcblx0R3JhZGllbnQgPSAoZGVyaXZhdGl2ZUZ1bmN0aW9uKSA9PiBgXG5cdFx0dW5pZm9ybSBUZW5zb3IgRTtcdC8vIGRvd25zdHJlYW0gZXJyb3Jcblx0XHR1bmlmb3JtIFRlbnNvciBPO1x0Ly8gbGF5ZXIgb3V0cHV0XG5cdFx0dW5pZm9ybSBUZW5zb3IgSDtcdC8vIHdlaWdodGVkIGlucHV0XG5cdFx0ZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHtcblx0XHRcdGZsb2F0IGQ7XG5cdFx0XHRmbG9hdCBvID0gTy5yZWFkKHBvcyk7XG5cdFx0XHQkeyBkZXJpdmF0aXZlRnVuY3Rpb24gfVxuXHRcdFx0ZCAqPSBFLnJlYWQocG9zKTtcblx0XHRcdHJldHVybiBkO1xuXHRcdH1cblx0YDtcblxuXG5cbmZ1bmN0aW9uIERlbnNlKGxheWVyLCBpbmRleCkge1xuXHR0aGlzLmwgPSBpbmRleDtcblx0Ly8gcHJvZHVjZSBPdXRwdXQgVGVuc29yIGdpdmVuIGlucHV0LCB3ZWlnaHRzLCBhbmQgYmlhcyBUZW5zb3JzXG5cdHRoaXMuZm9yd2FyZCA9IGxheWVyLmJpYXMgPyBGb3J3YXJkQmlhc2VkIDogRm9yd2FyZFVuYmlhc2VkO1xuXG5cdHRoaXMuYWN0aXZhdGlvbiA9IEFjdGl2YXRpb24oRnVuY3MuQWN0aXZhdGlvbltsYXllci5hY3RpdmF0aW9uXSk7XG5cdFx0XHRcdFx0XHRcblx0Ly8gcHJvZHVjZSB1cHN0cmVhbSBlcnJvciBUZW5zb3IgZ2l2ZW4gZG93bnN0cmVhbSBlcnJvciwgaW5wdXQsIHdlaWdodHMsIGJpYXNcblx0dGhpcy5iYWNrd2FyZCA9IGxheWVyLmJpYXMgPyBCYWNrd2FyZEJpYXNlZCA6IEJhY2t3YXJkVW5iaWFzZWQ7XG5cdHRoaXMuZ3JhZGllbnQgPSBHcmFkaWVudChGdW5jcy5EZXJpdmF0aXZlW2xheWVyLmFjdGl2YXRpb25dKTtcblx0Ly8gYWRqdXN0IHdlaWdodHMgVGVuc29yIGdpdmVuIGVycm9yIGFuZCBpbnB1dCBUZW5zb3Jcblx0dGhpcy51cGRhdGUgPSBXZWlnaHRzO1xuXG5cdHRoaXMuc2hhcGUgPSBsYXllci5zaGFwZTtcblx0dGhpcy5pbnB1dCA9IG51bGw7XG5cdHRoaXMub3V0cHV0ID0gbnVsbDtcblx0dGhpcy53ZWlnaHRlZE91dHB1dCA9IG51bGw7XG5cdHRoaXMud2VpZ2h0cyA9IG51bGw7XG5cdHRoaXMuYmlhcyA9IGxheWVyLmJpYXM7XG5cdHRoaXMuc2l6ZSA9IHRoaXMuc2hhcGVbMF0gKiB0aGlzLnNoYXBlWzFdICsgKHRoaXMuYmlhcyA/IHRoaXMuc2hhcGVbMF0gOiAwKTtcblxufVxuRGVuc2UucHJvdG90eXBlLmxvYWQgPSBmdW5jdGlvbihhcnJheSwgb2Zmc2V0KSB7XG5cdHZhciBsZW5ndGggPSB0aGlzLnNpemU7XG5cdC8vIHJlYWQgaW4gd2VpZ2h0cyAoYW5kIGJpYXMpXG5cdHRoaXMud2VpZ2h0cyA9IG5ldyBURi5JblBsYWNlVGVuc29yKEdMLCBuZGFycmF5KCBhcnJheS5zdWJhcnJheShvZmZzZXQsIG9mZnNldCArIGxlbmd0aCksIFt0aGlzLnNoYXBlWzBdLCB0aGlzLnNoYXBlWzFdICsgKHRoaXMuYmlhcyA/IDEgOiAwKV0gKSApO1xuXHRvZmZzZXQgKz0gbGVuZ3RoO1xuXHRyZXR1cm4gb2Zmc2V0O1xufVxuRGVuc2UucHJvdG90eXBlLnJhbmRvbVdlaWdodHMgPSBmdW5jdGlvbigpIHtcblx0dGhpcy53ZWlnaHRzID0gbmV3IFRGLkluUGxhY2VUZW5zb3IoR0wsIFxuXHRcdG5kYXJyYXkoXG5cdFx0XHRnZW5lcmF0ZVdlaWdodHModGhpcy5zaGFwZSwgKHRoaXMuYmlhcyA/IHRoaXMuc2hhcGVbMF0gOiAwKSksIC8vIHZhbHVlc1xuXHRcdFx0W3RoaXMuc2hhcGVbMF0sIHRoaXMuc2hhcGVbMV0gKyAodGhpcy5iaWFzID8gMSA6IDApXSAvLyBzaGFwZVxuXHRcdClcblx0KTtcbn1cbkRlbnNlLnByb3RvdHlwZS5zYXZlID0gZnVuY3Rpb24oKSB7XG5cdHJldHVybiB0aGlzLndlaWdodHMucmVhZCgpLmRhdGE7XG59XG5EZW5zZS5wcm90b3R5cGUucnVuID0gZnVuY3Rpb24oaW5wdXQpIHtcblx0aWYgKGlucHV0IGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KSB7XG5cdFx0dGhpcy5pbnB1dCA9IG5ldyBURi5UZW5zb3IoR0wsIG5kYXJyYXkoIGlucHV0LCBbIHRoaXMuc2hhcGVbMV0sIChpbnB1dC5sZW5ndGggLyB0aGlzLnNoYXBlWzFdKSA+PiAwIF0pKTtcblx0fSBlbHNlIHRoaXMuaW5wdXQgPSBpbnB1dDtcblx0Ly9jb25zb2xlLmxvZyh0aGlzLmlucHV0LnNoYXBlKTtcblx0Ly9jb25zb2xlLmxvZyhcIkNhbGN1bG9uLSBpbnB1dCBcIiArIHRoaXMubCArIFwiOiBcIiArIHRoaXMuaW5wdXQucmVhZCgpLmRhdGEpO1xuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIHdlaWdodHMgXCIgKyB0aGlzLmwgKyBcIjogXCIgKyB0aGlzLndlaWdodHMucmVhZCgpLmRhdGEpO1xuXG5cdHRoaXMud2VpZ2h0ZWRPdXRwdXQgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBbIHRoaXMuc2hhcGVbMF0sIHRoaXMuaW5wdXQuc2hhcGVbMV0gXSk7XG5cdHRoaXMud2VpZ2h0ZWRPdXRwdXQucnVuKHRoaXMuZm9yd2FyZCwge1c6IHRoaXMud2VpZ2h0cywgSTogdGhpcy5pbnB1dH0pO1xuXG5cdC8vY29uc29sZS5sb2coXCJDYWxjdWxvbi0gd2VpZ2h0ZWRPdXRwdXQgXCIgKyB0aGlzLmwgKyBcIjogXCIgKyB0aGlzLndlaWdodGVkT3V0cHV0LnJlYWQoKS5kYXRhKTtcblxuXHR0aGlzLm91dHB1dCA9IG5ldyBURi5PdXRwdXRUZW5zb3IoR0wsIFsgdGhpcy5zaGFwZVswXSwgdGhpcy5pbnB1dC5zaGFwZVsxXSBdKTtcblx0dGhpcy5vdXRwdXQucnVuKHRoaXMuYWN0aXZhdGlvbiwge086IHRoaXMud2VpZ2h0ZWRPdXRwdXR9KTtcblxuXHQvL2NvbnNvbGUubG9nKFwib3V0cHV0IFwiICsgdGhpcy5sOiBcIiArIHRoaXMub3V0cHV0LnJlYWQoKS5kYXRhKTtcblx0cmV0dXJuIHRoaXMub3V0cHV0O1xufTtcbkRlbnNlLnByb3RvdHlwZS50cmFpbiA9IGZ1bmN0aW9uKGVycm9yLCBsZWFybmluZ19yYXRlKSB7XG5cdHRoaXMucGFydGlhbCA9IG5ldyBURi5PdXRwdXRUZW5zb3IoR0wsIHRoaXMuaW5wdXQuc2hhcGUpO1xuXHR0aGlzLmxvY2FsID0gbmV3IFRGLk91dHB1dFRlbnNvcihHTCwgdGhpcy5vdXRwdXQuc2hhcGUpO1xuXG5cdC8vY29uc29sZS5sb2coXCJDYWxjdWxvbi0gZXJyb3I6IFwiICsgZXJyb3IucmVhZCgpLmRhdGEpO1xuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIHdlaWdodHMgXCIgKyB0aGlzLmw6IFwiICsgdGhpcy53ZWlnaHRzLnJlYWQoKS5kYXRhKTtcblxuXHQvLyBjYWxjdWxhdGUgbG9jYWwgZXJyb3IgZnJvbSB3ZWlnaHRlZE91dHB1dCAoc3RyaXBzIG91dCBlcnJvciBmcm9tIGFjdGl2YXRpb24gZnVuY3Rpb24pXG5cdHRoaXMubG9jYWwucnVuKHRoaXMuZ3JhZGllbnQsIHtFOiBlcnJvciwgTzogdGhpcy5vdXRwdXQsIEg6IHRoaXMud2VpZ2h0ZWRPdXRwdXR9KTtcblx0Ly9jb25zb2xlLmxvZyhcIkNhbGN1bG9uLSBsb2NhbEU6IFwiICsgbG9jYWwucmVhZCgpLmRhdGEpO1xuXG5cdC8vIGNhbGN1bGF0ZSB1cHN0cmVhbSBlcnJvcnMgZnJvbSBpbnB1dFxuXHR0aGlzLnBhcnRpYWwucnVuKHRoaXMuYmFja3dhcmQsIHtFOiB0aGlzLmxvY2FsLCBXOiB0aGlzLndlaWdodHN9KTtcblxuXHQvLyB0cmFpbiB3ZWlnaHRzIGJhc2VkIG9uIGxvY2FsIGVycm9yXG5cdHRoaXMud2VpZ2h0cy5ydW4odGhpcy51cGRhdGUsIHtXOiB0aGlzLndlaWdodHMsIEU6IHRoaXMubG9jYWwsIEk6IHRoaXMuaW5wdXQsIGw6IGxlYXJuaW5nX3JhdGV9KTtcblx0Ly9jb25zb2xlLmxvZyhcIkNhbGN1bG9uLSB1cGRhdGVkIFwiICsgdGhpcy5sOiBcIiArIHRoaXMud2VpZ2h0cy5yZWFkKCkuZGF0YSk7XG5cblx0cmV0dXJuIHRoaXMucGFydGlhbDtcbn07XG5cbm1vZHVsZS5leHBvcnRzID0gZnVuY3Rpb24odGVuc29yZmlyZSwgZ2xDb250ZXh0KSB7XG5cdFRGID0gdGVuc29yZmlyZTtcblx0R0wgPSBnbENvbnRleHQ7XG5cdHJldHVybiBEZW5zZTtcbn07IiwidmFyIG5kYXJyYXkgPSByZXF1aXJlKFwibmRhcnJheVwiKSxcblx0VEYsXG5cdEdMLFxuXG5cdEdyYWQgXHQ9IGBcblx0XHR1bmlmb3JtIFRlbnNvciBPO1xuXHRcdHVuaWZvcm0gVGVuc29yIEU7XG5cdFx0ZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHtcblx0XHRcdHJldHVybiBPLnJlYWQocG9zKSAtIEUucmVhZChwb3MpO1xuXHRcdH1cblx0YCxcblxuXHRBY2N1cmFjeSA9IGBcblx0XHR1bmlmb3JtIFRlbnNvciBPO1xuXHRcdHVuaWZvcm0gVGVuc29yIEU7XG5cdFx0ZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHtcblx0XHRcdGZsb2F0IHMgPSAwLjA7XG5cdFx0XHRmb3IgKGludCBpID0gMDsgaSA8ICMoTy5zaGFwZSkueDsgaSsrKSB7IC8vIGl0ZXJhdGUgb3ZlciBldmVyeSBvdXRwdXRcblx0XHRcdFx0cyArPSBwb3coKEUucmVhZChpLCBwb3MueCkgLSBPLnJlYWQoaSwgcG9zLngpKSwgMi4wKTtcblx0XHRcdH1cblx0XHRcdHJldHVybiAxLjAgLSBjbGFtcChzIC8gZmxvYXQoIyhPLnNoYXBlKS54KSwgMC4wLCAxLjApO1xuXHRcdH1cblx0YCxcblxuXHRMb3NzIFx0PSBgXG5cdFx0dW5pZm9ybSBUZW5zb3IgRztcblx0XHRmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykge1xuXHRcdFx0ZmxvYXQgbG9zcyA9IDAuMDtcblx0XHRcdGZvcihpbnQgaSA9IDA7IGkgPCAjKEcuc2hhcGUpLnk7IGkrKyl7IC8vIGl0ZXJhdGUgb3ZlciBlYWNoIHNhbXBsZVxuXHRcdFx0XHRmbG9hdCBsID0gMC4wO1xuXHRcdFx0XHRmb3IoaW50IGogPSAwOyBqIDwgIyhHLnNoYXBlKS54OyBqKyspeyAvLyBpdGVyYXRlIG92ZXIgZXZlcnkgb3V0cHV0IGFuZCBjYWxjdWxhdGUgYXZlcmFnZVxuXHRcdFx0XHRcdGwgKz0gcG93KGZsb2F0KEcucmVhZChqLCBpKSksIDIuMCk7XG5cdFx0XHRcdH1cblx0XHRcdFx0bG9zcyArPSBsIC8gZmxvYXQoIyhHLnNoYXBlKS55KTtcblx0XHRcdH1cblx0XHRcdHJldHVybiBsb3NzO1xuXHRcdH1cblx0YDtcblxuZnVuY3Rpb24gTVNFKCkge1xuXHQvLyBjYWxjdWxhdGUgbG9zcyBncmFkaWVudHNcblx0dGhpcy5ncmFkID0gR3JhZDtcblxuXHQvLyBjYWxjdWxhdGUgYmF0Y2ggYXZlcmFnZSBsb3NzXG5cdHRoaXMubG9zc0YgPSBMb3NzO1xuXG5cdHRoaXMuQWNjdXJhY3kgPSBBY2N1cmFjeTtcblxuXHR0aGlzLmxvc3MgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBbMV0pO1xuXHR0aGlzLm91dHB1dCA9IG51bGw7XG59XG5NU0UucHJvdG90eXBlLmRlbHRhcyA9IGZ1bmN0aW9uKG91dHB1dCwgZXhwZWN0KSB7XG5cdGlmIChleHBlY3QgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkpXG5cdFx0ZXhwZWN0ID0gbmV3IFRGLlRlbnNvcihHTCwgbmRhcnJheSggZXhwZWN0LCBvdXRwdXQuc2hhcGUpKTtcblxuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIGV4cGVjdGVkOiBcIiArIGV4cGVjdC5yZWFkKCkuZGF0YSk7XG5cblx0dGhpcy5vdXRwdXQgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBvdXRwdXQuc2hhcGUpO1xuXHR0aGlzLm91dHB1dC5ydW4odGhpcy5ncmFkLCB7IE86IG91dHB1dCwgRTogZXhwZWN0IH0pO1xuXHQvL2NvbnNvbGUubG9nKFwiQ2FsY3Vsb24tIGdyYWRpZW50OiBcIiArIHRoaXMub3V0cHV0LnJlYWQoKS5kYXRhKTtcblxuXHR0aGlzLmxvc3MucnVuKHRoaXMubG9zc0YsIHsgRzogdGhpcy5vdXRwdXQgfSk7XG5cblx0cmV0dXJuIHRoaXMub3V0cHV0O1xufVxuXG5tb2R1bGUuZXhwb3J0cyA9IGZ1bmN0aW9uKHRlbnNvcmZpcmUsIGdsQ29udGV4dCkge1xuXHRURiA9IHRlbnNvcmZpcmU7XG5cdEdMID0gZ2xDb250ZXh0O1xuXHRyZXR1cm4gTVNFO1xufTsiLCJ2YXIgbmRhcnJheSA9IHJlcXVpcmUoXCJuZGFycmF5XCIpLFxuXHRURixcblx0R0wsXG5cdFNvZnRtYXggPSByZXF1aXJlKCcuL1NvZnRtYXgnKSxcblx0TVNFID0gcmVxdWlyZSgnLi9NU0UnKSxcblx0Q3Jvc3NFbnRyb3B5ID0gcmVxdWlyZSgnLi9Dcm9zc0VudHJvcHknKSxcblxuXHRBY2NTdW1cdD0gYFxuXHRcdHVuaWZvcm0gVGVuc29yIEE7XG5cdFx0ZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHtcblx0XHRcdGZsb2F0IGFjYyA9IDAuMDtcblx0XHRcdGZvciAoaW50IGkgPSAwOyBpIDwgIyhBLnNoYXBlKS55OyBpKyspeyAvLyBpdGVyYXRlIG92ZXIgZWFjaCBzYW1wbGVcblx0XHRcdFx0YWNjICs9IEEucmVhZChwb3MueCwgaSk7XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gYWNjIC8gZmxvYXQoIyhBLnNoYXBlKS55KTtcblx0XHR9XG5cdGA7XG5cbmZ1bmN0aW9uIE91dHB1dChsYXllciwgaW5kZXgpIHtcblx0dGhpcy5vdXRwdXQgPSBudWxsO1xuXHRpZiAobGF5ZXIuYWN0aXZhdGlvbiA9PT0gXCJzb2Z0bWF4XCIgJiYgbGF5ZXIubG9zcyA9PT0gXCJ4ZW50cm9weVwiKSB7XG5cdFx0dGhpcy5vdXRwdXQgPSBuZXcgU29mdG1heChsYXllciwgaW5kZXgpO1xuXHRcdHRoaXMucnVuID0gKGlucHV0KSA9PiB7XG5cdFx0XHR0aGlzLl9vdXRwdXQgPSB0aGlzLm91dHB1dC5ydW4oaW5wdXQpO1xuXHRcdFx0cmV0dXJuIHRoaXMuX291dHB1dDtcblx0XHR9O1xuXHR9IGVsc2Uge1xuXHRcdHN3aXRjaCAobGF5ZXIubG9zcykge1xuXHRcdFx0Y2FzZSBcInhlbnRyb3B5XCI6XG5cdFx0XHRcdHRoaXMub3V0cHV0ID0gbmV3IENyb3NzRW50cm9weSgpO1xuXHRcdFx0XHRicmVhaztcblx0XHRcdGNhc2UgXCJtc2VcIjpcblx0XHRcdFx0dGhpcy5vdXRwdXQgPSBuZXcgTVNFKCk7XG5cdFx0XHRcdGJyZWFrO1xuXHRcdH1cblx0XHR0aGlzLnJ1biA9IHRoaXMucnVuLmJpbmQodGhpcyk7XG5cdFx0dGhpcy50cmFpbiA9IHRoaXMudHJhaW4uYmluZCh0aGlzKTtcblx0fVxuXG5cdHRoaXMuX291dHB1dCA9IG51bGw7XG5cdHRoaXMuYWNjdXJhY3kgPSAwO1xufVxuT3V0cHV0LnByb3RvdHlwZS5ydW4gPSBmdW5jdGlvbihpbnB1dCkge1xuXHR0aGlzLl9vdXRwdXQgPSBpbnB1dDtcblx0cmV0dXJuIGlucHV0O1xufTtcbk91dHB1dC5wcm90b3R5cGUudHJhaW4gPSBmdW5jdGlvbihleHBlY3RlZCkge1xuXHQvL2NvbnNvbGUubG9nKFwiRXhwZWN0ZWQ6IFwiICsgZXhwZWN0ZWQpO1xuXHRpZiAoZXhwZWN0ZWQgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkpXG5cdFx0ZXhwZWN0ZWQgPSBuZXcgVEYuVGVuc29yKEdMLCBuZGFycmF5KCBleHBlY3RlZCwgdGhpcy5fb3V0cHV0LnNoYXBlKSk7XG5cblx0XG5cdC8vY29uc29sZS5sb2coXCIgIE91dHB1dDogXCIgKyB0aGlzLl9vdXRwdXQucmVhZCgpLmRhdGEpO1xuXG5cdHRoaXMuYmF0Y2hBY2N1cmFjeSA9IG5ldyBURi5PdXRwdXRUZW5zb3IoR0wsIFsxLCB0aGlzLl9vdXRwdXQuc2hhcGVbMV1dKTtcblx0dGhpcy5fYWNjdXJhY3kgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCBbMV0pO1xuXHR0aGlzLmJhdGNoQWNjdXJhY3kucnVuKHRoaXMub3V0cHV0LkFjY3VyYWN5LCB7IE86IHRoaXMuX291dHB1dCwgRTogZXhwZWN0ZWQgfSk7XG5cdHRoaXMuX2FjY3VyYWN5LnJ1bihBY2NTdW0sIHsgQTogdGhpcy5iYXRjaEFjY3VyYWN5IH0pO1xuXHR0aGlzLmFjY3VyYWN5ID0gdGhpcy5fYWNjdXJhY3kucmVhZCgpLmRhdGFbMF07XG5cblx0cmV0dXJuIHRoaXMub3V0cHV0LmRlbHRhcyh0aGlzLl9vdXRwdXQsIGV4cGVjdGVkKTtcbn07XG5cblxubW9kdWxlLmV4cG9ydHMgPSBmdW5jdGlvbih0ZW5zb3JmaXJlLCBnbENvbnRleHQpIHtcblxuXHRURiA9IHRlbnNvcmZpcmU7XG5cdEdMID0gZ2xDb250ZXh0O1xuXG5cdFNvZnRtYXggPSBTb2Z0bWF4KHRlbnNvcmZpcmUsIGdsQ29udGV4dCk7XG5cdE1TRSA9IE1TRSh0ZW5zb3JmaXJlLCBnbENvbnRleHQpO1xuXHRDcm9zc0VudHJvcHkgPSBDcm9zc0VudHJvcHkodGVuc29yZmlyZSwgZ2xDb250ZXh0KTtcblxuXHRyZXR1cm4gT3V0cHV0O1xufTsiLCJ2YXIgbmRhcnJheSA9IHJlcXVpcmUoXCJuZGFycmF5XCIpLFxuXHRURixcblx0R0wsXG5cblx0QWN0ID0gYFxuXHRcdHVuaWZvcm0gVGVuc29yIEk7IC8vIGlucHV0XG5cdFx0ZmxvYXQgcHJvY2VzcyhpdmVjNCBwb3MpIHtcblx0XHRcdGZsb2F0IGsgPSAwLjA7XG5cdFx0XHRmb3IoaW50IGkgPSAwOyBpIDwgIyhJLnNoYXBlKS54OyBpKyspe1xuXHRcdFx0XHRrICs9IGV4cChJLnJlYWQoaSwgcG9zLnkpKTtcblx0XHRcdH1cblx0XHRcdHJldHVybiBleHAoSS5yZWFkKHBvcykpIC8gaztcblx0XHR9XG5cdGAsXG5cdEdyYWQgPSBgXG5cdFx0dW5pZm9ybSBUZW5zb3IgTzsgLy8gU29mdG1heCBvdXRwdXRcblx0XHR1bmlmb3JtIFRlbnNvciBFOyAvLyBleHBlY3RlZCBvdXRwdXRcblx0XHRmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykge1xuXHRcdFx0cmV0dXJuIE8ucmVhZChwb3MpIC0gRS5yZWFkKHBvcyk7XG5cdFx0fVxuXHRgLFxuXG5cdEFjY3VyYWN5ID0gYFxuXHRcdHVuaWZvcm0gVGVuc29yIE87XG5cdFx0dW5pZm9ybSBUZW5zb3IgRTtcblx0XHRmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykge1xuXHRcdFx0ZmxvYXQgcyA9IDAuMDtcblx0XHRcdGZvciAoaW50IGkgPSAwOyBpIDwgIyhPLnNoYXBlKS54OyBpKyspIHsgLy8gaXRlcmF0ZSBvdmVyIGV2ZXJ5IG91dHB1dFxuXHRcdFx0XHRzICs9IHBvdygoRS5yZWFkKGksIHBvcy54KSAtIE8ucmVhZChpLCBwb3MueCkpLCAyLjApO1xuXHRcdFx0fVxuXHRcdFx0cmV0dXJuIDEuMCAtIGNsYW1wKHMgLyBmbG9hdCgjKE8uc2hhcGUpLngpLCAwLjAsIDEuMCk7XG5cdFx0fVxuXHRgLFxuXG5cdC8vIEFjY3VyYWN5ID0gYFxuXHQvLyBcdHVuaWZvcm0gVGVuc29yIE87XG5cdC8vIFx0dW5pZm9ybSBUZW5zb3IgRTtcblx0Ly8gXHRmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykge1xuXHQvLyBcdFx0aW50IGwgPSAjKE8uc2hhcGUpLng7XG5cdC8vIFx0XHRmbG9hdCB2ID0gLTEwMDAwMC4wO1xuXHQvLyBcdFx0Zm9yIChpbnQgaSA9IDA7IGkgPCAjKE8uc2hhcGUpLng7IGkrKykgeyAvLyBpdGVyYXRlIG92ZXIgZXZlcnkgb3V0cHV0XG5cdC8vIFx0XHRcdGlmIChPLnJlYWQoaSwgcG9zLngpID4gdikge1xuXHQvLyBcdFx0XHRcdHYgPSBPLnJlYWQoaSwgcG9zLnkpOyAvKiBnZXQgbGFyZ2VzdCBjYXRlZ29yeSBpbiBvdXRwdXQgKi9cblx0Ly8gXHRcdFx0XHRsID0gaTsgLyogc2F2ZSBpbmRleCBvZiBsYXJnZXN0IHZhbHVlICovXG5cdC8vIFx0XHRcdH1cblx0Ly8gXHRcdH1cblx0Ly8gXHRcdGlmIChFLnJlYWQobCwgcG9zLnkpID4gMC45KSB7XG5cdC8vIFx0XHRcdHJldHVybiAxLjA7XG5cdC8vIFx0XHR9IGVsc2Uge1xuXHQvLyBcdFx0XHRyZXR1cm4gMC4wO1xuXHQvLyBcdFx0fVxuXHQvLyBcdH1cblx0Ly8gYCxcblxuXG5cdExvc3MgPSBgXG5cdFx0dW5pZm9ybSBUZW5zb3IgRztcblx0XHRmbG9hdCBwcm9jZXNzKGl2ZWM0IHBvcykge1xuXHRcdFx0ZmxvYXQgbG9zcyA9IDAuMDtcblx0XHRcdGZvcihpbnQgaSA9IDA7IGkgPCAjKEcuc2hhcGUpLnk7IGkrKyl7IC8qIGl0ZXJhdGUgb3ZlciBlYWNoIHNhbXBsZSAqL1xuXHRcdFx0XHRmbG9hdCBsID0gMC4wO1xuXHRcdFx0XHRmb3IoaW50IGogPSAwOyBqIDwgIyhHLnNoYXBlKS54OyBqKyspeyAvKiBpdGVyYXRlIG92ZXIgZXZlcnkgb3V0cHV0IGFuZCBjYWxjdWxhdGUgYXZlcmFnZSAqL1xuXHRcdFx0XHRcdGwgKz0gcG93KGZsb2F0KEcucmVhZChqLCBpKSksIDIuMCkgLyBmbG9hdCgjKEcuc2hhcGUpLngpO1xuXHRcdFx0XHR9XG5cdFx0XHRcdGxvc3MgKz0gbCAvIGZsb2F0KCMoRy5zaGFwZSkueSk7XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gbG9zcztcblx0XHR9XG5cdGA7XG5cblxuZnVuY3Rpb24gU29mdG1heChsYXllciwgaW5kZXgpIHtcblx0dGhpcy5sID0gaW5kZXg7XG5cblx0dGhpcy5hY3RpdmF0aW9uID0gQWN0O1xuXG5cdHRoaXMuZ3JhZGllbnQgPSBHcmFkO1xuXHR0aGlzLmxvc3NGID0gTG9zcztcdFxuXG5cdHRoaXMuQWNjdXJhY3kgPSBBY2N1cmFjeTtcblxuXG5cdHRoaXMuaW5wdXQgPSBudWxsO1xuXHR0aGlzLm91dHB1dCA9IG51bGw7XG5cdHRoaXMubG9zcyA9IG5ldyBURi5PdXRwdXRUZW5zb3IoR0wsIFsxXSk7XG59XG5Tb2Z0bWF4LnByb3RvdHlwZS5ydW4gPSBmdW5jdGlvbihpbnB1dCkge1xuXG5cdHRoaXMuaW5wdXQgPSBpbnB1dDtcblx0dGhpcy5vdXRwdXQgPSBuZXcgVEYuT3V0cHV0VGVuc29yKEdMLCB0aGlzLmlucHV0LnNoYXBlKTtcblxuXHR0aGlzLm91dHB1dC5ydW4odGhpcy5hY3RpdmF0aW9uLCB7STogdGhpcy5pbnB1dH0pO1xuXG5cdC8vY29uc29sZS5sb2coXCJvdXRwdXQgXCIgKyB0aGlzLmwgKyBcIjogXCIgKyB0aGlzLm91dHB1dC5yZWFkKCkuZGF0YSk7XG5cdHJldHVybiB0aGlzLm91dHB1dDtcbn1cblNvZnRtYXgucHJvdG90eXBlLmRlbHRhcyA9IGZ1bmN0aW9uKG91dHB1dCwgZXhwZWN0ZWQpIHtcblx0dGhpcy5wYXJ0aWFsID0gbmV3IFRGLk91dHB1dFRlbnNvcihHTCwgdGhpcy5pbnB1dC5zaGFwZSk7XG5cblx0aWYgKGV4cGVjdGVkIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KVxuXHRcdGV4cGVjdGVkID0gbmV3IFRGLlRlbnNvcihHTCwgbmRhcnJheShleHBlY3RlZCwgdGhpcy5pbnB1dC5zaGFwZSkpO1xuXG5cdC8vIGNhbGN1bGF0ZSB1cHN0cmVhbSBlcnJvcnNcblx0dGhpcy5wYXJ0aWFsLnJ1bih0aGlzLmdyYWRpZW50LCB7Tzogb3V0cHV0LCBFOiBleHBlY3RlZH0pO1xuXG5cdC8vIGNhbGN1bGF0ZSBiYXRjaCB0cmFpbmluZyBsb3NzXG5cdHRoaXMubG9zcy5ydW4odGhpcy5sb3NzRiwgeyBHOiB0aGlzLnBhcnRpYWwgfSk7XG5cdC8vY29uc29sZS5sb2codGhpcy5sb3NzLnJlYWQoKSk7XG5cblx0Ly8gY29uc29sZS5sb2cob3V0cHV0LnJlYWQoKS5kYXRhKTtcblxuXHRyZXR1cm4gdGhpcy5wYXJ0aWFsO1xufVxuXG5tb2R1bGUuZXhwb3J0cyA9IGZ1bmN0aW9uKHRlbnNvcmZpcmUsIGdsQ29udGV4dCkge1xuXHRURiA9IHRlbnNvcmZpcmU7XG5cdEdMID0gZ2xDb250ZXh0O1xuXHRyZXR1cm4gU29mdG1heDtcbn0iLCIvLyBTdGFuZGFyZCBOb3JtYWwgdmFyaWF0ZSB1c2luZyBCb3gtTXVsbGVyIHRyYW5zZm9ybS5cbmZ1bmN0aW9uIHJhbmRvbShtZWFuLCBzdGREZXYpIHtcblx0bWVhbiA9IG1lYW4gfHwgMDtcblx0c3RkRGV2ID0gc3RkRGV2IHx8IDE7XG4gICAgdmFyIHUgPSAwLCB2ID0gMDtcbiAgICB3aGlsZSh1ID09PSAwKSB1ID0gTWF0aC5yYW5kb20oKTsgLy9Db252ZXJ0aW5nIFswLDEpIHRvICgwLDEpXG4gICAgd2hpbGUodiA9PT0gMCkgdiA9IE1hdGgucmFuZG9tKCk7XG4gICAgLy9yZXR1cm4gMC40O1xuICAgIHJldHVybiAoTWF0aC5zcXJ0KCAtMi4wICogTWF0aC5sb2coIHUgKSApICogTWF0aC5jb3MoIDIuMCAqIE1hdGguUEkgKiB2ICkpICogc3RkRGV2ICsgbWVhbjtcbn1cblxubW9kdWxlLmV4cG9ydHMgPSBmdW5jdGlvbiBnZW5lcmF0ZVdlaWdodHMoc2hhcGUsIGJpYXMpIHtcblx0dmFyIHJlc3VsdCA9IG5ldyBGbG9hdDMyQXJyYXkoc2hhcGVbMF0gKiBzaGFwZVsxXSArIGJpYXMpO1xuXHRjb25zb2xlLmxvZyhcIkxheWVyIHdlaWdodHMgKyBiaWFzOiBcIiArIHJlc3VsdC5sZW5ndGgpO1xuXHR2YXIgbCA9IC0xO1xuXHR3aGlsZSAoKytsIDwgcmVzdWx0Lmxlbmd0aCkge1xuXHRcdHJlc3VsdFtsXSA9IHJhbmRvbSgwLCBNYXRoLnNxcnQoMiAvIHNoYXBlWzFdKSk7XG5cdH1cblx0Ly9jb25zb2xlLmxvZyhyZXN1bHRbMF0pO1xuXHRyZXR1cm4gcmVzdWx0O1xufSIsInZhciBzcHJpbnRmID0gcmVxdWlyZSgnc3ByaW50ZicpO1xubW9kdWxlLmV4cG9ydHMgPSBmb3JtYXQ7XG5cbmZ1bmN0aW9uIGZvcm1hdCAoeCwgYnl0ZXMpIHtcbiAgICBpZiAoYnl0ZXMgPT09IHVuZGVmaW5lZCkgYnl0ZXMgPSA4O1xuICAgIHZhciByZm10ID0gJyUnICsgYnl0ZXMgKyAnLicgKyBieXRlcyArICdzJztcbiAgICBcbiAgICBpZiAoYnl0ZXMgPD0gMCkgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICBpZiAoaXNOYU4oeCkpIHJldHVybiBzcHJpbnRmKHJmbXQsICdOYU4nKTtcbiAgICBpZiAoeCA9PT0gSW5maW5pdHkpIHtcbiAgICAgICAgaWYgKGJ5dGVzID09PSAxKSByZXR1cm4gdW5kZWZpbmVkO1xuICAgICAgICByZXR1cm4gc3ByaW50ZihyZm10LCBieXRlcyA+PSA5ID8gJ0luZmluaXR5JyA6ICcgSW5mJykuc2xpY2UoMCwgYnl0ZXMpO1xuICAgIH1cbiAgICBpZiAoeCA9PT0gLUluZmluaXR5KSB7XG4gICAgICAgIGlmIChieXRlcyA9PT0gMSkgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICAgICAgcmV0dXJuIHNwcmludGYocmZtdCwgYnl0ZXMgPj0gOSA/ICctSW5maW5pdHknIDogJy1JbmYnKS5zbGljZSgwLCBieXRlcyk7XG4gICAgfVxuICAgIHJldHVybiBwYWNrZih4LCBieXRlcyk7XG59O1xuXG5mdW5jdGlvbiBzY2kgKHgsIGJ5dGVzKSB7XG4gICAgdmFyIG4gPSBNYXRoLm1heCgxLCBsb2cxMGYoTWF0aC5hYnMoeCkpKTtcbiAgICB2YXIgc3ogPSBsb2cxMGYoTWF0aC5hYnMobikpO1xuICAgIFxuICAgIHZhciBiID0gTWF0aC5wb3coMTAsYnl0ZXMrMSk7XG4gICAgaWYgKE1hdGguYWJzKHgpIDwgMSkge1xuICAgICAgICB4ID0gTWF0aC5yb3VuZCh4ICogYikgLyBiO1xuICAgIH1cbiAgICBlbHNlIHtcbiAgICAgICAgdmFyIHRuID0gTWF0aC5wb3coMTAsIG4gKyAxKTtcbiAgICAgICAgeCA9IE1hdGgucm91bmQoeCAvIHRuICogYikgLyBiICogdG47XG4gICAgfVxuICAgIFxuICAgIHZhciBzO1xuICAgIGlmIChieXRlcyAtIHN6IC0gNiA9PT0gLTEpIHtcbiAgICAgICAgeCA9IE1hdGgucm91bmQoeCAvIE1hdGgucG93KDEwLCBuKSk7XG4gICAgICAgIHggPSB4ICogTWF0aC5wb3coMTAsIG4pO1xuICAgICAgICBzID0gc3ByaW50ZignJTFlJywgeCkucmVwbGFjZSgvXFwuW15lXSsvLCAnJyk7XG4gICAgfVxuICAgIGVsc2UgaWYgKGJ5dGVzIC0gc3ogLSA2IDwgMCkgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICBlbHNlIHtcbiAgICAgICAgcyA9IHNwcmludGYoJyUuJyArIChieXRlcyAtIHN6IC0gNikgKyAnZScsIHgpO1xuICAgIH1cbiAgICBpZiAoeCA+IDApIHMgPSAnICcgKyBzO1xuICAgIHJldHVybiBwYWQocywgYnl0ZXMpO1xufVxuXG5mdW5jdGlvbiBwYWQgKHMsIGJ5dGVzKSB7XG4gICAgcmV0dXJuIEFycmF5KE1hdGgubWF4KDAsIGJ5dGVzIC0gcy5sZW5ndGggKyAxKSkuam9pbignICcpICsgcztcbn1cblxuZnVuY3Rpb24gbG9nMTBmIChuKSB7XG4gICAgcmV0dXJuIE1hdGguZmxvb3IoTWF0aC5sb2cobikgLyBNYXRoLmxvZygxMCkpO1xufVxuXG5mdW5jdGlvbiBwYWNrZiAoeCwgYnl0ZXMpIHtcbiAgICB2YXIgbGJ5dGVzID0gTWF0aC5tYXgoMSwgTWF0aC5mbG9vcigoYnl0ZXMgLSAyKSAvIDIpKTtcbiAgICB2YXIgcmJ5dGVzID0gYnl0ZXMgLSBsYnl0ZXMgLSAyO1xuICAgIFxuICAgIGlmICh4ID09PSAwICYmIGJ5dGVzIDwgNCkge1xuICAgICAgICByZXR1cm4gcGFkKCcwJywgYnl0ZXMpO1xuICAgIH1cbiAgICBlbHNlIGlmICh4ID09PSAwKSB7XG4gICAgICAgIHJldHVybiBwYWQoJzAuJyArIEFycmF5KHJieXRlcysxKS5qb2luKCcwJyksIGJ5dGVzKTtcbiAgICB9XG4gICAgXG4gICAgaWYgKHJieXRlcyA8PSAwKSB7XG4gICAgICAgIHZhciBzID0gc3ByaW50ZignJScgKyBsYnl0ZXMgKyAnZicsIHgpO1xuICAgICAgICBpZiAoeCA+PSAwKSBzID0gJyAnICsgcztcbiAgICAgICAgaWYgKHMubGVuZ3RoID4gYnl0ZXMpIHJldHVybiB1bmRlZmluZWQ7XG4gICAgICAgIHJldHVybiBwYWQocywgYnl0ZXMpO1xuICAgIH1cbiAgICBpZiAoTWF0aC5hYnMoeCkgPCBNYXRoLnBvdygxMCwxLXJieXRlcykpIHJldHVybiBzY2koeCwgYnl0ZXMpO1xuICAgIFxuICAgIHZhciBiID0gTWF0aC5wb3coMTAsYnl0ZXMtMyk7XG4gICAgdmFyIHRuID0gTWF0aC5wb3coMTAsIGxvZzEwZihNYXRoLmFicyh4KSkpO1xuICAgIHZhciB4ciA9IE1hdGgucm91bmQoeCAvIHRuICogYikgLyBiICogdG47XG4gICAgXG4gICAgdmFyIHMgPSBzcHJpbnRmKCclJyArIGxieXRlcyArICcuJyArIHJieXRlcyArICdmJywgeHIpO1xuICAgIGlmICh4ciA+IDApIHMgPSAnICcgKyBzO1xuICAgIHMgPSBzLnNsaWNlKDAsIGJ5dGVzKTtcbiAgICB2YXIgciA9IHMuc3BsaXQoJy4nKVsxXTtcbiAgICBpZiAoIXIgfHwgci5sZW5ndGggPCAxKSByZXR1cm4gc2NpKHhyLCBieXRlcyk7XG4gICAgcmV0dXJuIHBhZChzLCBieXRlcykuc2xpY2UoMCwgYnl0ZXMpO1xufVxuIiwiXCJ1c2Ugc3RyaWN0XCJcblxuZnVuY3Rpb24gaW90YShuKSB7XG4gIHZhciByZXN1bHQgPSBuZXcgQXJyYXkobilcbiAgZm9yKHZhciBpPTA7IGk8bjsgKytpKSB7XG4gICAgcmVzdWx0W2ldID0gaVxuICB9XG4gIHJldHVybiByZXN1bHRcbn1cblxubW9kdWxlLmV4cG9ydHMgPSBpb3RhIiwiLyohXG4gKiBEZXRlcm1pbmUgaWYgYW4gb2JqZWN0IGlzIGEgQnVmZmVyXG4gKlxuICogQGF1dGhvciAgIEZlcm9zcyBBYm91a2hhZGlqZWggPGZlcm9zc0BmZXJvc3Mub3JnPiA8aHR0cDovL2Zlcm9zcy5vcmc+XG4gKiBAbGljZW5zZSAgTUlUXG4gKi9cblxuLy8gVGhlIF9pc0J1ZmZlciBjaGVjayBpcyBmb3IgU2FmYXJpIDUtNyBzdXBwb3J0LCBiZWNhdXNlIGl0J3MgbWlzc2luZ1xuLy8gT2JqZWN0LnByb3RvdHlwZS5jb25zdHJ1Y3Rvci4gUmVtb3ZlIHRoaXMgZXZlbnR1YWxseVxubW9kdWxlLmV4cG9ydHMgPSBmdW5jdGlvbiAob2JqKSB7XG4gIHJldHVybiBvYmogIT0gbnVsbCAmJiAoaXNCdWZmZXIob2JqKSB8fCBpc1Nsb3dCdWZmZXIob2JqKSB8fCAhIW9iai5faXNCdWZmZXIpXG59XG5cbmZ1bmN0aW9uIGlzQnVmZmVyIChvYmopIHtcbiAgcmV0dXJuICEhb2JqLmNvbnN0cnVjdG9yICYmIHR5cGVvZiBvYmouY29uc3RydWN0b3IuaXNCdWZmZXIgPT09ICdmdW5jdGlvbicgJiYgb2JqLmNvbnN0cnVjdG9yLmlzQnVmZmVyKG9iailcbn1cblxuLy8gRm9yIE5vZGUgdjAuMTAgc3VwcG9ydC4gUmVtb3ZlIHRoaXMgZXZlbnR1YWxseS5cbmZ1bmN0aW9uIGlzU2xvd0J1ZmZlciAob2JqKSB7XG4gIHJldHVybiB0eXBlb2Ygb2JqLnJlYWRGbG9hdExFID09PSAnZnVuY3Rpb24nICYmIHR5cGVvZiBvYmouc2xpY2UgPT09ICdmdW5jdGlvbicgJiYgaXNCdWZmZXIob2JqLnNsaWNlKDAsIDApKVxufVxuIiwidmFyIHNob3dmID0gcmVxdWlyZSgnZml4ZWQtd2lkdGgtZmxvYXQnKTtcbnZhciBuZGFycmF5ID0gcmVxdWlyZSgnbmRhcnJheScpO1xuXG5tb2R1bGUuZXhwb3J0cyA9IGZ1bmN0aW9uIChtLCBvcHRzKSB7XG4gICAgaWYgKCFvcHRzKSBvcHRzID0ge307XG4gICAgaWYgKHR5cGVvZiBvcHRzID09PSAnbnVtYmVyJykgb3B0cyA9IHsgd2lkdGg6IG9wdHMgfTtcbiAgICBpZiAoIW9wdHMud2lkdGgpIG9wdHMud2lkdGggPSA4O1xuXG4gICAgaWYgKG0uZGltZW5zaW9uID09PSB1bmRlZmluZWQpIHtcbiAgICAgICAgbSA9IG5kYXJyYXkobSk7XG4gICAgfVxuXG4gICAgaWYgKG0uZGltZW5zaW9uID09PSAxKSByZXR1cm4gZDEobSwgb3B0cyk7XG4gICAgaWYgKG0uZGltZW5zaW9uID09PSAyKSByZXR1cm4gZDIobSwgb3B0cyk7XG4gICAgaWYgKG0uZGltZW5zaW9uID09PSAzKSByZXR1cm4gZDMobSwgb3B0cyk7XG4gICAgaWYgKG0uZGltZW5zaW9uID09PSA0KSByZXR1cm4gZDQobSwgb3B0cyk7XG59O1xuXG5mdW5jdGlvbiBkMSAobSwgb3B0cykge1xuICAgIHZhciB0ZXJtcyA9IFtdO1xuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgbS5zaGFwZVswXTsgaSsrKSB7XG4gICAgICAgIHRlcm1zLnB1c2goc2hvd2YobS5nZXQoaSksIG9wdHMud2lkdGgpKTtcbiAgICB9XG4gICAgcmV0dXJuIHRlcm1zLmpvaW4oJyAnKTtcbn1cblxuZnVuY3Rpb24gZDIgKG0sIG9wdHMpIHtcbiAgICB2YXIgcm93cyA9IFtdO1xuICAgIGZvciAodmFyIHkgPSAwOyB5IDwgbS5zaGFwZVswXTsgeSsrKSB7XG4gICAgICAgIHJvd3MucHVzaChkMShtLnBpY2soeSwgbnVsbCksIG9wdHMpKTtcbiAgICB9XG4gICAgcmV0dXJuIHJvd3Muam9pbignXFxuJyk7XG59XG5cbmZ1bmN0aW9uIGQzIChtLCBvcHRzKSB7XG4gICAgdmFyIHJvd3MgPSBbXTtcbiAgICBmb3IgKHZhciB6ID0gMDsgeiA8IG0uc2hhcGVbMF07IHorKykge1xuICAgICAgICByb3dzLnB1c2goZDIobS5waWNrKHosIG51bGwsIG51bGwpLCBvcHRzKSwgJycpO1xuICAgIH1cbiAgICByZXR1cm4gcm93cy5qb2luKCdcXG4nKTtcbn1cblxuZnVuY3Rpb24gZDQgKG0sIG9wdHMpIHtcbiAgICB2YXIgcm93cyA9IFtdLCBsZW4gPSAzXG4gICAgZm9yICh2YXIgdyA9IDA7IHcgPCBtLnNoYXBlWzBdOyB3KyspIHtcbiAgICAgICAgdmFyIHIgPSBkMyhtLnBpY2sodywgbnVsbCwgbnVsbCwgbnVsbCksIG9wdHMpXG4gICAgICAgIHJvd3MucHVzaChyKTtcbiAgICAgICAgdmFyIGxpbmVzID0gci5zcGxpdCgnXFxuJyk7XG4gICAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgbGluZXMubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICAgIGxlbiA9IE1hdGgubWF4KGxlbiwgbGluZXNbaV0ubGVuZ3RoKTtcbiAgICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gcm93cy5qb2luKCdcXG4nICsgQXJyYXkobGVuKzEpLmpvaW4oJy0nKSArICdcXG5cXG4nKTtcbn1cbiIsInZhciBpb3RhID0gcmVxdWlyZShcImlvdGEtYXJyYXlcIilcbnZhciBpc0J1ZmZlciA9IHJlcXVpcmUoXCJpcy1idWZmZXJcIilcblxudmFyIGhhc1R5cGVkQXJyYXlzICA9ICgodHlwZW9mIEZsb2F0NjRBcnJheSkgIT09IFwidW5kZWZpbmVkXCIpXG5cbmZ1bmN0aW9uIGNvbXBhcmUxc3QoYSwgYikge1xuICByZXR1cm4gYVswXSAtIGJbMF1cbn1cblxuZnVuY3Rpb24gb3JkZXIoKSB7XG4gIHZhciBzdHJpZGUgPSB0aGlzLnN0cmlkZVxuICB2YXIgdGVybXMgPSBuZXcgQXJyYXkoc3RyaWRlLmxlbmd0aClcbiAgdmFyIGlcbiAgZm9yKGk9MDsgaTx0ZXJtcy5sZW5ndGg7ICsraSkge1xuICAgIHRlcm1zW2ldID0gW01hdGguYWJzKHN0cmlkZVtpXSksIGldXG4gIH1cbiAgdGVybXMuc29ydChjb21wYXJlMXN0KVxuICB2YXIgcmVzdWx0ID0gbmV3IEFycmF5KHRlcm1zLmxlbmd0aClcbiAgZm9yKGk9MDsgaTxyZXN1bHQubGVuZ3RoOyArK2kpIHtcbiAgICByZXN1bHRbaV0gPSB0ZXJtc1tpXVsxXVxuICB9XG4gIHJldHVybiByZXN1bHRcbn1cblxuZnVuY3Rpb24gY29tcGlsZUNvbnN0cnVjdG9yKGR0eXBlLCBkaW1lbnNpb24pIHtcbiAgdmFyIGNsYXNzTmFtZSA9IFtcIlZpZXdcIiwgZGltZW5zaW9uLCBcImRcIiwgZHR5cGVdLmpvaW4oXCJcIilcbiAgaWYoZGltZW5zaW9uIDwgMCkge1xuICAgIGNsYXNzTmFtZSA9IFwiVmlld19OaWxcIiArIGR0eXBlXG4gIH1cbiAgdmFyIHVzZUdldHRlcnMgPSAoZHR5cGUgPT09IFwiZ2VuZXJpY1wiKVxuXG4gIGlmKGRpbWVuc2lvbiA9PT0gLTEpIHtcbiAgICAvL1NwZWNpYWwgY2FzZSBmb3IgdHJpdmlhbCBhcnJheXNcbiAgICB2YXIgY29kZSA9XG4gICAgICBcImZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIihhKXt0aGlzLmRhdGE9YTt9O1xcXG52YXIgcHJvdG89XCIrY2xhc3NOYW1lK1wiLnByb3RvdHlwZTtcXFxucHJvdG8uZHR5cGU9J1wiK2R0eXBlK1wiJztcXFxucHJvdG8uaW5kZXg9ZnVuY3Rpb24oKXtyZXR1cm4gLTF9O1xcXG5wcm90by5zaXplPTA7XFxcbnByb3RvLmRpbWVuc2lvbj0tMTtcXFxucHJvdG8uc2hhcGU9cHJvdG8uc3RyaWRlPXByb3RvLm9yZGVyPVtdO1xcXG5wcm90by5sbz1wcm90by5oaT1wcm90by50cmFuc3Bvc2U9cHJvdG8uc3RlcD1cXFxuZnVuY3Rpb24oKXtyZXR1cm4gbmV3IFwiK2NsYXNzTmFtZStcIih0aGlzLmRhdGEpO307XFxcbnByb3RvLmdldD1wcm90by5zZXQ9ZnVuY3Rpb24oKXt9O1xcXG5wcm90by5waWNrPWZ1bmN0aW9uKCl7cmV0dXJuIG51bGx9O1xcXG5yZXR1cm4gZnVuY3Rpb24gY29uc3RydWN0X1wiK2NsYXNzTmFtZStcIihhKXtyZXR1cm4gbmV3IFwiK2NsYXNzTmFtZStcIihhKTt9XCJcbiAgICB2YXIgcHJvY2VkdXJlID0gbmV3IEZ1bmN0aW9uKGNvZGUpXG4gICAgcmV0dXJuIHByb2NlZHVyZSgpXG4gIH0gZWxzZSBpZihkaW1lbnNpb24gPT09IDApIHtcbiAgICAvL1NwZWNpYWwgY2FzZSBmb3IgMGQgYXJyYXlzXG4gICAgdmFyIGNvZGUgPVxuICAgICAgXCJmdW5jdGlvbiBcIitjbGFzc05hbWUrXCIoYSxkKSB7XFxcbnRoaXMuZGF0YSA9IGE7XFxcbnRoaXMub2Zmc2V0ID0gZFxcXG59O1xcXG52YXIgcHJvdG89XCIrY2xhc3NOYW1lK1wiLnByb3RvdHlwZTtcXFxucHJvdG8uZHR5cGU9J1wiK2R0eXBlK1wiJztcXFxucHJvdG8uaW5kZXg9ZnVuY3Rpb24oKXtyZXR1cm4gdGhpcy5vZmZzZXR9O1xcXG5wcm90by5kaW1lbnNpb249MDtcXFxucHJvdG8uc2l6ZT0xO1xcXG5wcm90by5zaGFwZT1cXFxucHJvdG8uc3RyaWRlPVxcXG5wcm90by5vcmRlcj1bXTtcXFxucHJvdG8ubG89XFxcbnByb3RvLmhpPVxcXG5wcm90by50cmFuc3Bvc2U9XFxcbnByb3RvLnN0ZXA9ZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX2NvcHkoKSB7XFxcbnJldHVybiBuZXcgXCIrY2xhc3NOYW1lK1wiKHRoaXMuZGF0YSx0aGlzLm9mZnNldClcXFxufTtcXFxucHJvdG8ucGljaz1mdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfcGljaygpe1xcXG5yZXR1cm4gVHJpdmlhbEFycmF5KHRoaXMuZGF0YSk7XFxcbn07XFxcbnByb3RvLnZhbHVlT2Y9cHJvdG8uZ2V0PWZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl9nZXQoKXtcXFxucmV0dXJuIFwiKyh1c2VHZXR0ZXJzID8gXCJ0aGlzLmRhdGEuZ2V0KHRoaXMub2Zmc2V0KVwiIDogXCJ0aGlzLmRhdGFbdGhpcy5vZmZzZXRdXCIpK1xuXCJ9O1xcXG5wcm90by5zZXQ9ZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiX3NldCh2KXtcXFxucmV0dXJuIFwiKyh1c2VHZXR0ZXJzID8gXCJ0aGlzLmRhdGEuc2V0KHRoaXMub2Zmc2V0LHYpXCIgOiBcInRoaXMuZGF0YVt0aGlzLm9mZnNldF09dlwiKStcIlxcXG59O1xcXG5yZXR1cm4gZnVuY3Rpb24gY29uc3RydWN0X1wiK2NsYXNzTmFtZStcIihhLGIsYyxkKXtyZXR1cm4gbmV3IFwiK2NsYXNzTmFtZStcIihhLGQpfVwiXG4gICAgdmFyIHByb2NlZHVyZSA9IG5ldyBGdW5jdGlvbihcIlRyaXZpYWxBcnJheVwiLCBjb2RlKVxuICAgIHJldHVybiBwcm9jZWR1cmUoQ0FDSEVEX0NPTlNUUlVDVE9SU1tkdHlwZV1bMF0pXG4gIH1cblxuICB2YXIgY29kZSA9IFtcIid1c2Ugc3RyaWN0J1wiXVxuXG4gIC8vQ3JlYXRlIGNvbnN0cnVjdG9yIGZvciB2aWV3XG4gIHZhciBpbmRpY2VzID0gaW90YShkaW1lbnNpb24pXG4gIHZhciBhcmdzID0gaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkgeyByZXR1cm4gXCJpXCIraSB9KVxuICB2YXIgaW5kZXhfc3RyID0gXCJ0aGlzLm9mZnNldCtcIiArIGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHtcbiAgICAgICAgcmV0dXJuIFwidGhpcy5zdHJpZGVbXCIgKyBpICsgXCJdKmlcIiArIGlcbiAgICAgIH0pLmpvaW4oXCIrXCIpXG4gIHZhciBzaGFwZUFyZyA9IGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHtcbiAgICAgIHJldHVybiBcImJcIitpXG4gICAgfSkuam9pbihcIixcIilcbiAgdmFyIHN0cmlkZUFyZyA9IGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHtcbiAgICAgIHJldHVybiBcImNcIitpXG4gICAgfSkuam9pbihcIixcIilcbiAgY29kZS5wdXNoKFxuICAgIFwiZnVuY3Rpb24gXCIrY2xhc3NOYW1lK1wiKGEsXCIgKyBzaGFwZUFyZyArIFwiLFwiICsgc3RyaWRlQXJnICsgXCIsZCl7dGhpcy5kYXRhPWFcIixcbiAgICAgIFwidGhpcy5zaGFwZT1bXCIgKyBzaGFwZUFyZyArIFwiXVwiLFxuICAgICAgXCJ0aGlzLnN0cmlkZT1bXCIgKyBzdHJpZGVBcmcgKyBcIl1cIixcbiAgICAgIFwidGhpcy5vZmZzZXQ9ZHwwfVwiLFxuICAgIFwidmFyIHByb3RvPVwiK2NsYXNzTmFtZStcIi5wcm90b3R5cGVcIixcbiAgICBcInByb3RvLmR0eXBlPSdcIitkdHlwZStcIidcIixcbiAgICBcInByb3RvLmRpbWVuc2lvbj1cIitkaW1lbnNpb24pXG5cbiAgLy92aWV3LnNpemU6XG4gIGNvZGUucHVzaChcIk9iamVjdC5kZWZpbmVQcm9wZXJ0eShwcm90bywnc2l6ZScse2dldDpmdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfc2l6ZSgpe1xcXG5yZXR1cm4gXCIraW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkgeyByZXR1cm4gXCJ0aGlzLnNoYXBlW1wiK2krXCJdXCIgfSkuam9pbihcIipcIiksXG5cIn19KVwiKVxuXG4gIC8vdmlldy5vcmRlcjpcbiAgaWYoZGltZW5zaW9uID09PSAxKSB7XG4gICAgY29kZS5wdXNoKFwicHJvdG8ub3JkZXI9WzBdXCIpXG4gIH0gZWxzZSB7XG4gICAgY29kZS5wdXNoKFwiT2JqZWN0LmRlZmluZVByb3BlcnR5KHByb3RvLCdvcmRlcicse2dldDpcIilcbiAgICBpZihkaW1lbnNpb24gPCA0KSB7XG4gICAgICBjb2RlLnB1c2goXCJmdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfb3JkZXIoKXtcIilcbiAgICAgIGlmKGRpbWVuc2lvbiA9PT0gMikge1xuICAgICAgICBjb2RlLnB1c2goXCJyZXR1cm4gKE1hdGguYWJzKHRoaXMuc3RyaWRlWzBdKT5NYXRoLmFicyh0aGlzLnN0cmlkZVsxXSkpP1sxLDBdOlswLDFdfX0pXCIpXG4gICAgICB9IGVsc2UgaWYoZGltZW5zaW9uID09PSAzKSB7XG4gICAgICAgIGNvZGUucHVzaChcblwidmFyIHMwPU1hdGguYWJzKHRoaXMuc3RyaWRlWzBdKSxzMT1NYXRoLmFicyh0aGlzLnN0cmlkZVsxXSksczI9TWF0aC5hYnModGhpcy5zdHJpZGVbMl0pO1xcXG5pZihzMD5zMSl7XFxcbmlmKHMxPnMyKXtcXFxucmV0dXJuIFsyLDEsMF07XFxcbn1lbHNlIGlmKHMwPnMyKXtcXFxucmV0dXJuIFsxLDIsMF07XFxcbn1lbHNle1xcXG5yZXR1cm4gWzEsMCwyXTtcXFxufVxcXG59ZWxzZSBpZihzMD5zMil7XFxcbnJldHVybiBbMiwwLDFdO1xcXG59ZWxzZSBpZihzMj5zMSl7XFxcbnJldHVybiBbMCwxLDJdO1xcXG59ZWxzZXtcXFxucmV0dXJuIFswLDIsMV07XFxcbn19fSlcIilcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgY29kZS5wdXNoKFwiT1JERVJ9KVwiKVxuICAgIH1cbiAgfVxuXG4gIC8vdmlldy5zZXQoaTAsIC4uLiwgdik6XG4gIGNvZGUucHVzaChcblwicHJvdG8uc2V0PWZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl9zZXQoXCIrYXJncy5qb2luKFwiLFwiKStcIix2KXtcIilcbiAgaWYodXNlR2V0dGVycykge1xuICAgIGNvZGUucHVzaChcInJldHVybiB0aGlzLmRhdGEuc2V0KFwiK2luZGV4X3N0citcIix2KX1cIilcbiAgfSBlbHNlIHtcbiAgICBjb2RlLnB1c2goXCJyZXR1cm4gdGhpcy5kYXRhW1wiK2luZGV4X3N0citcIl09dn1cIilcbiAgfVxuXG4gIC8vdmlldy5nZXQoaTAsIC4uLik6XG4gIGNvZGUucHVzaChcInByb3RvLmdldD1mdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfZ2V0KFwiK2FyZ3Muam9pbihcIixcIikrXCIpe1wiKVxuICBpZih1c2VHZXR0ZXJzKSB7XG4gICAgY29kZS5wdXNoKFwicmV0dXJuIHRoaXMuZGF0YS5nZXQoXCIraW5kZXhfc3RyK1wiKX1cIilcbiAgfSBlbHNlIHtcbiAgICBjb2RlLnB1c2goXCJyZXR1cm4gdGhpcy5kYXRhW1wiK2luZGV4X3N0citcIl19XCIpXG4gIH1cblxuICAvL3ZpZXcuaW5kZXg6XG4gIGNvZGUucHVzaChcbiAgICBcInByb3RvLmluZGV4PWZ1bmN0aW9uIFwiK2NsYXNzTmFtZStcIl9pbmRleChcIiwgYXJncy5qb2luKCksIFwiKXtyZXR1cm4gXCIraW5kZXhfc3RyK1wifVwiKVxuXG4gIC8vdmlldy5oaSgpOlxuICBjb2RlLnB1c2goXCJwcm90by5oaT1mdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfaGkoXCIrYXJncy5qb2luKFwiLFwiKStcIil7cmV0dXJuIG5ldyBcIitjbGFzc05hbWUrXCIodGhpcy5kYXRhLFwiK1xuICAgIGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHtcbiAgICAgIHJldHVybiBbXCIodHlwZW9mIGlcIixpLFwiIT09J251bWJlcid8fGlcIixpLFwiPDApP3RoaXMuc2hhcGVbXCIsIGksIFwiXTppXCIsIGksXCJ8MFwiXS5qb2luKFwiXCIpXG4gICAgfSkuam9pbihcIixcIikrXCIsXCIrXG4gICAgaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkge1xuICAgICAgcmV0dXJuIFwidGhpcy5zdHJpZGVbXCIraSArIFwiXVwiXG4gICAgfSkuam9pbihcIixcIikrXCIsdGhpcy5vZmZzZXQpfVwiKVxuXG4gIC8vdmlldy5sbygpOlxuICB2YXIgYV92YXJzID0gaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkgeyByZXR1cm4gXCJhXCIraStcIj10aGlzLnNoYXBlW1wiK2krXCJdXCIgfSlcbiAgdmFyIGNfdmFycyA9IGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHsgcmV0dXJuIFwiY1wiK2krXCI9dGhpcy5zdHJpZGVbXCIraStcIl1cIiB9KVxuICBjb2RlLnB1c2goXCJwcm90by5sbz1mdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfbG8oXCIrYXJncy5qb2luKFwiLFwiKStcIil7dmFyIGI9dGhpcy5vZmZzZXQsZD0wLFwiK2FfdmFycy5qb2luKFwiLFwiKStcIixcIitjX3ZhcnMuam9pbihcIixcIikpXG4gIGZvcih2YXIgaT0wOyBpPGRpbWVuc2lvbjsgKytpKSB7XG4gICAgY29kZS5wdXNoKFxuXCJpZih0eXBlb2YgaVwiK2krXCI9PT0nbnVtYmVyJyYmaVwiK2krXCI+PTApe1xcXG5kPWlcIitpK1wifDA7XFxcbmIrPWNcIitpK1wiKmQ7XFxcbmFcIitpK1wiLT1kfVwiKVxuICB9XG4gIGNvZGUucHVzaChcInJldHVybiBuZXcgXCIrY2xhc3NOYW1lK1wiKHRoaXMuZGF0YSxcIitcbiAgICBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7XG4gICAgICByZXR1cm4gXCJhXCIraVxuICAgIH0pLmpvaW4oXCIsXCIpK1wiLFwiK1xuICAgIGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHtcbiAgICAgIHJldHVybiBcImNcIitpXG4gICAgfSkuam9pbihcIixcIikrXCIsYil9XCIpXG5cbiAgLy92aWV3LnN0ZXAoKTpcbiAgY29kZS5wdXNoKFwicHJvdG8uc3RlcD1mdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfc3RlcChcIithcmdzLmpvaW4oXCIsXCIpK1wiKXt2YXIgXCIrXG4gICAgaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkge1xuICAgICAgcmV0dXJuIFwiYVwiK2krXCI9dGhpcy5zaGFwZVtcIitpK1wiXVwiXG4gICAgfSkuam9pbihcIixcIikrXCIsXCIrXG4gICAgaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkge1xuICAgICAgcmV0dXJuIFwiYlwiK2krXCI9dGhpcy5zdHJpZGVbXCIraStcIl1cIlxuICAgIH0pLmpvaW4oXCIsXCIpK1wiLGM9dGhpcy5vZmZzZXQsZD0wLGNlaWw9TWF0aC5jZWlsXCIpXG4gIGZvcih2YXIgaT0wOyBpPGRpbWVuc2lvbjsgKytpKSB7XG4gICAgY29kZS5wdXNoKFxuXCJpZih0eXBlb2YgaVwiK2krXCI9PT0nbnVtYmVyJyl7XFxcbmQ9aVwiK2krXCJ8MDtcXFxuaWYoZDwwKXtcXFxuYys9YlwiK2krXCIqKGFcIitpK1wiLTEpO1xcXG5hXCIraStcIj1jZWlsKC1hXCIraStcIi9kKVxcXG59ZWxzZXtcXFxuYVwiK2krXCI9Y2VpbChhXCIraStcIi9kKVxcXG59XFxcbmJcIitpK1wiKj1kXFxcbn1cIilcbiAgfVxuICBjb2RlLnB1c2goXCJyZXR1cm4gbmV3IFwiK2NsYXNzTmFtZStcIih0aGlzLmRhdGEsXCIrXG4gICAgaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkge1xuICAgICAgcmV0dXJuIFwiYVwiICsgaVxuICAgIH0pLmpvaW4oXCIsXCIpK1wiLFwiK1xuICAgIGluZGljZXMubWFwKGZ1bmN0aW9uKGkpIHtcbiAgICAgIHJldHVybiBcImJcIiArIGlcbiAgICB9KS5qb2luKFwiLFwiKStcIixjKX1cIilcblxuICAvL3ZpZXcudHJhbnNwb3NlKCk6XG4gIHZhciB0U2hhcGUgPSBuZXcgQXJyYXkoZGltZW5zaW9uKVxuICB2YXIgdFN0cmlkZSA9IG5ldyBBcnJheShkaW1lbnNpb24pXG4gIGZvcih2YXIgaT0wOyBpPGRpbWVuc2lvbjsgKytpKSB7XG4gICAgdFNoYXBlW2ldID0gXCJhW2lcIitpK1wiXVwiXG4gICAgdFN0cmlkZVtpXSA9IFwiYltpXCIraStcIl1cIlxuICB9XG4gIGNvZGUucHVzaChcInByb3RvLnRyYW5zcG9zZT1mdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfdHJhbnNwb3NlKFwiK2FyZ3MrXCIpe1wiK1xuICAgIGFyZ3MubWFwKGZ1bmN0aW9uKG4saWR4KSB7IHJldHVybiBuICsgXCI9KFwiICsgbiArIFwiPT09dW5kZWZpbmVkP1wiICsgaWR4ICsgXCI6XCIgKyBuICsgXCJ8MClcIn0pLmpvaW4oXCI7XCIpLFxuICAgIFwidmFyIGE9dGhpcy5zaGFwZSxiPXRoaXMuc3RyaWRlO3JldHVybiBuZXcgXCIrY2xhc3NOYW1lK1wiKHRoaXMuZGF0YSxcIit0U2hhcGUuam9pbihcIixcIikrXCIsXCIrdFN0cmlkZS5qb2luKFwiLFwiKStcIix0aGlzLm9mZnNldCl9XCIpXG5cbiAgLy92aWV3LnBpY2soKTpcbiAgY29kZS5wdXNoKFwicHJvdG8ucGljaz1mdW5jdGlvbiBcIitjbGFzc05hbWUrXCJfcGljayhcIithcmdzK1wiKXt2YXIgYT1bXSxiPVtdLGM9dGhpcy5vZmZzZXRcIilcbiAgZm9yKHZhciBpPTA7IGk8ZGltZW5zaW9uOyArK2kpIHtcbiAgICBjb2RlLnB1c2goXCJpZih0eXBlb2YgaVwiK2krXCI9PT0nbnVtYmVyJyYmaVwiK2krXCI+PTApe2M9KGMrdGhpcy5zdHJpZGVbXCIraStcIl0qaVwiK2krXCIpfDB9ZWxzZXthLnB1c2godGhpcy5zaGFwZVtcIitpK1wiXSk7Yi5wdXNoKHRoaXMuc3RyaWRlW1wiK2krXCJdKX1cIilcbiAgfVxuICBjb2RlLnB1c2goXCJ2YXIgY3Rvcj1DVE9SX0xJU1RbYS5sZW5ndGgrMV07cmV0dXJuIGN0b3IodGhpcy5kYXRhLGEsYixjKX1cIilcblxuICAvL0FkZCByZXR1cm4gc3RhdGVtZW50XG4gIGNvZGUucHVzaChcInJldHVybiBmdW5jdGlvbiBjb25zdHJ1Y3RfXCIrY2xhc3NOYW1lK1wiKGRhdGEsc2hhcGUsc3RyaWRlLG9mZnNldCl7cmV0dXJuIG5ldyBcIitjbGFzc05hbWUrXCIoZGF0YSxcIitcbiAgICBpbmRpY2VzLm1hcChmdW5jdGlvbihpKSB7XG4gICAgICByZXR1cm4gXCJzaGFwZVtcIitpK1wiXVwiXG4gICAgfSkuam9pbihcIixcIikrXCIsXCIrXG4gICAgaW5kaWNlcy5tYXAoZnVuY3Rpb24oaSkge1xuICAgICAgcmV0dXJuIFwic3RyaWRlW1wiK2krXCJdXCJcbiAgICB9KS5qb2luKFwiLFwiKStcIixvZmZzZXQpfVwiKVxuXG4gIC8vQ29tcGlsZSBwcm9jZWR1cmVcbiAgdmFyIHByb2NlZHVyZSA9IG5ldyBGdW5jdGlvbihcIkNUT1JfTElTVFwiLCBcIk9SREVSXCIsIGNvZGUuam9pbihcIlxcblwiKSlcbiAgcmV0dXJuIHByb2NlZHVyZShDQUNIRURfQ09OU1RSVUNUT1JTW2R0eXBlXSwgb3JkZXIpXG59XG5cbmZ1bmN0aW9uIGFycmF5RFR5cGUoZGF0YSkge1xuICBpZihpc0J1ZmZlcihkYXRhKSkge1xuICAgIHJldHVybiBcImJ1ZmZlclwiXG4gIH1cbiAgaWYoaGFzVHlwZWRBcnJheXMpIHtcbiAgICBzd2l0Y2goT2JqZWN0LnByb3RvdHlwZS50b1N0cmluZy5jYWxsKGRhdGEpKSB7XG4gICAgICBjYXNlIFwiW29iamVjdCBGbG9hdDY0QXJyYXldXCI6XG4gICAgICAgIHJldHVybiBcImZsb2F0NjRcIlxuICAgICAgY2FzZSBcIltvYmplY3QgRmxvYXQzMkFycmF5XVwiOlxuICAgICAgICByZXR1cm4gXCJmbG9hdDMyXCJcbiAgICAgIGNhc2UgXCJbb2JqZWN0IEludDhBcnJheV1cIjpcbiAgICAgICAgcmV0dXJuIFwiaW50OFwiXG4gICAgICBjYXNlIFwiW29iamVjdCBJbnQxNkFycmF5XVwiOlxuICAgICAgICByZXR1cm4gXCJpbnQxNlwiXG4gICAgICBjYXNlIFwiW29iamVjdCBJbnQzMkFycmF5XVwiOlxuICAgICAgICByZXR1cm4gXCJpbnQzMlwiXG4gICAgICBjYXNlIFwiW29iamVjdCBVaW50OEFycmF5XVwiOlxuICAgICAgICByZXR1cm4gXCJ1aW50OFwiXG4gICAgICBjYXNlIFwiW29iamVjdCBVaW50MTZBcnJheV1cIjpcbiAgICAgICAgcmV0dXJuIFwidWludDE2XCJcbiAgICAgIGNhc2UgXCJbb2JqZWN0IFVpbnQzMkFycmF5XVwiOlxuICAgICAgICByZXR1cm4gXCJ1aW50MzJcIlxuICAgICAgY2FzZSBcIltvYmplY3QgVWludDhDbGFtcGVkQXJyYXldXCI6XG4gICAgICAgIHJldHVybiBcInVpbnQ4X2NsYW1wZWRcIlxuICAgIH1cbiAgfVxuICBpZihBcnJheS5pc0FycmF5KGRhdGEpKSB7XG4gICAgcmV0dXJuIFwiYXJyYXlcIlxuICB9XG4gIHJldHVybiBcImdlbmVyaWNcIlxufVxuXG52YXIgQ0FDSEVEX0NPTlNUUlVDVE9SUyA9IHtcbiAgXCJmbG9hdDMyXCI6W10sXG4gIFwiZmxvYXQ2NFwiOltdLFxuICBcImludDhcIjpbXSxcbiAgXCJpbnQxNlwiOltdLFxuICBcImludDMyXCI6W10sXG4gIFwidWludDhcIjpbXSxcbiAgXCJ1aW50MTZcIjpbXSxcbiAgXCJ1aW50MzJcIjpbXSxcbiAgXCJhcnJheVwiOltdLFxuICBcInVpbnQ4X2NsYW1wZWRcIjpbXSxcbiAgXCJidWZmZXJcIjpbXSxcbiAgXCJnZW5lcmljXCI6W11cbn1cblxuOyhmdW5jdGlvbigpIHtcbiAgZm9yKHZhciBpZCBpbiBDQUNIRURfQ09OU1RSVUNUT1JTKSB7XG4gICAgQ0FDSEVEX0NPTlNUUlVDVE9SU1tpZF0ucHVzaChjb21waWxlQ29uc3RydWN0b3IoaWQsIC0xKSlcbiAgfVxufSk7XG5cbmZ1bmN0aW9uIHdyYXBwZWROREFycmF5Q3RvcihkYXRhLCBzaGFwZSwgc3RyaWRlLCBvZmZzZXQpIHtcbiAgaWYoZGF0YSA9PT0gdW5kZWZpbmVkKSB7XG4gICAgdmFyIGN0b3IgPSBDQUNIRURfQ09OU1RSVUNUT1JTLmFycmF5WzBdXG4gICAgcmV0dXJuIGN0b3IoW10pXG4gIH0gZWxzZSBpZih0eXBlb2YgZGF0YSA9PT0gXCJudW1iZXJcIikge1xuICAgIGRhdGEgPSBbZGF0YV1cbiAgfVxuICBpZihzaGFwZSA9PT0gdW5kZWZpbmVkKSB7XG4gICAgc2hhcGUgPSBbIGRhdGEubGVuZ3RoIF1cbiAgfVxuICB2YXIgZCA9IHNoYXBlLmxlbmd0aFxuICBpZihzdHJpZGUgPT09IHVuZGVmaW5lZCkge1xuICAgIHN0cmlkZSA9IG5ldyBBcnJheShkKVxuICAgIGZvcih2YXIgaT1kLTEsIHN6PTE7IGk+PTA7IC0taSkge1xuICAgICAgc3RyaWRlW2ldID0gc3pcbiAgICAgIHN6ICo9IHNoYXBlW2ldXG4gICAgfVxuICB9XG4gIGlmKG9mZnNldCA9PT0gdW5kZWZpbmVkKSB7XG4gICAgb2Zmc2V0ID0gMFxuICAgIGZvcih2YXIgaT0wOyBpPGQ7ICsraSkge1xuICAgICAgaWYoc3RyaWRlW2ldIDwgMCkge1xuICAgICAgICBvZmZzZXQgLT0gKHNoYXBlW2ldLTEpKnN0cmlkZVtpXVxuICAgICAgfVxuICAgIH1cbiAgfVxuICB2YXIgZHR5cGUgPSBhcnJheURUeXBlKGRhdGEpXG4gIHZhciBjdG9yX2xpc3QgPSBDQUNIRURfQ09OU1RSVUNUT1JTW2R0eXBlXVxuICB3aGlsZShjdG9yX2xpc3QubGVuZ3RoIDw9IGQrMSkge1xuICAgIGN0b3JfbGlzdC5wdXNoKGNvbXBpbGVDb25zdHJ1Y3RvcihkdHlwZSwgY3Rvcl9saXN0Lmxlbmd0aC0xKSlcbiAgfVxuICB2YXIgY3RvciA9IGN0b3JfbGlzdFtkKzFdXG4gIHJldHVybiBjdG9yKGRhdGEsIHNoYXBlLCBzdHJpZGUsIG9mZnNldClcbn1cblxubW9kdWxlLmV4cG9ydHMgPSB3cmFwcGVkTkRBcnJheUN0b3JcbiIsIi8qKlxuc3ByaW50ZigpIGZvciBKYXZhU2NyaXB0IDAuNy1iZXRhMVxuaHR0cDovL3d3dy5kaXZlaW50b2phdmFzY3JpcHQuY29tL3Byb2plY3RzL2phdmFzY3JpcHQtc3ByaW50ZlxuXG5Db3B5cmlnaHQgKGMpIEFsZXhhbmRydSBNYXJhc3RlYW51IDxhbGV4YWhvbGljIFthdCkgZ21haWwgKGRvdF0gY29tPlxuQWxsIHJpZ2h0cyByZXNlcnZlZC5cblxuUmVkaXN0cmlidXRpb24gYW5kIHVzZSBpbiBzb3VyY2UgYW5kIGJpbmFyeSBmb3Jtcywgd2l0aCBvciB3aXRob3V0XG5tb2RpZmljYXRpb24sIGFyZSBwZXJtaXR0ZWQgcHJvdmlkZWQgdGhhdCB0aGUgZm9sbG93aW5nIGNvbmRpdGlvbnMgYXJlIG1ldDpcbiAgICAqIFJlZGlzdHJpYnV0aW9ucyBvZiBzb3VyY2UgY29kZSBtdXN0IHJldGFpbiB0aGUgYWJvdmUgY29weXJpZ2h0XG4gICAgICBub3RpY2UsIHRoaXMgbGlzdCBvZiBjb25kaXRpb25zIGFuZCB0aGUgZm9sbG93aW5nIGRpc2NsYWltZXIuXG4gICAgKiBSZWRpc3RyaWJ1dGlvbnMgaW4gYmluYXJ5IGZvcm0gbXVzdCByZXByb2R1Y2UgdGhlIGFib3ZlIGNvcHlyaWdodFxuICAgICAgbm90aWNlLCB0aGlzIGxpc3Qgb2YgY29uZGl0aW9ucyBhbmQgdGhlIGZvbGxvd2luZyBkaXNjbGFpbWVyIGluIHRoZVxuICAgICAgZG9jdW1lbnRhdGlvbiBhbmQvb3Igb3RoZXIgbWF0ZXJpYWxzIHByb3ZpZGVkIHdpdGggdGhlIGRpc3RyaWJ1dGlvbi5cbiAgICAqIE5laXRoZXIgdGhlIG5hbWUgb2Ygc3ByaW50ZigpIGZvciBKYXZhU2NyaXB0IG5vciB0aGVcbiAgICAgIG5hbWVzIG9mIGl0cyBjb250cmlidXRvcnMgbWF5IGJlIHVzZWQgdG8gZW5kb3JzZSBvciBwcm9tb3RlIHByb2R1Y3RzXG4gICAgICBkZXJpdmVkIGZyb20gdGhpcyBzb2Z0d2FyZSB3aXRob3V0IHNwZWNpZmljIHByaW9yIHdyaXR0ZW4gcGVybWlzc2lvbi5cblxuVEhJUyBTT0ZUV0FSRSBJUyBQUk9WSURFRCBCWSBUSEUgQ09QWVJJR0hUIEhPTERFUlMgQU5EIENPTlRSSUJVVE9SUyBcIkFTIElTXCIgQU5EXG5BTlkgRVhQUkVTUyBPUiBJTVBMSUVEIFdBUlJBTlRJRVMsIElOQ0xVRElORywgQlVUIE5PVCBMSU1JVEVEIFRPLCBUSEUgSU1QTElFRFxuV0FSUkFOVElFUyBPRiBNRVJDSEFOVEFCSUxJVFkgQU5EIEZJVE5FU1MgRk9SIEEgUEFSVElDVUxBUiBQVVJQT1NFIEFSRVxuRElTQ0xBSU1FRC4gSU4gTk8gRVZFTlQgU0hBTEwgQWxleGFuZHJ1IE1hcmFzdGVhbnUgQkUgTElBQkxFIEZPUiBBTllcbkRJUkVDVCwgSU5ESVJFQ1QsIElOQ0lERU5UQUwsIFNQRUNJQUwsIEVYRU1QTEFSWSwgT1IgQ09OU0VRVUVOVElBTCBEQU1BR0VTXG4oSU5DTFVESU5HLCBCVVQgTk9UIExJTUlURUQgVE8sIFBST0NVUkVNRU5UIE9GIFNVQlNUSVRVVEUgR09PRFMgT1IgU0VSVklDRVM7XG5MT1NTIE9GIFVTRSwgREFUQSwgT1IgUFJPRklUUzsgT1IgQlVTSU5FU1MgSU5URVJSVVBUSU9OKSBIT1dFVkVSIENBVVNFRCBBTkRcbk9OIEFOWSBUSEVPUlkgT0YgTElBQklMSVRZLCBXSEVUSEVSIElOIENPTlRSQUNULCBTVFJJQ1QgTElBQklMSVRZLCBPUiBUT1JUXG4oSU5DTFVESU5HIE5FR0xJR0VOQ0UgT1IgT1RIRVJXSVNFKSBBUklTSU5HIElOIEFOWSBXQVkgT1VUIE9GIFRIRSBVU0UgT0YgVEhJU1xuU09GVFdBUkUsIEVWRU4gSUYgQURWSVNFRCBPRiBUSEUgUE9TU0lCSUxJVFkgT0YgU1VDSCBEQU1BR0UuXG5cblxuQ2hhbmdlbG9nOlxuMjAxMC4xMS4wNyAtIDAuNy1iZXRhMS1ub2RlXG4gIC0gY29udmVydGVkIGl0IHRvIGEgbm9kZS5qcyBjb21wYXRpYmxlIG1vZHVsZVxuXG4yMDEwLjA5LjA2IC0gMC43LWJldGExXG4gIC0gZmVhdHVyZXM6IHZzcHJpbnRmLCBzdXBwb3J0IGZvciBuYW1lZCBwbGFjZWhvbGRlcnNcbiAgLSBlbmhhbmNlbWVudHM6IGZvcm1hdCBjYWNoZSwgcmVkdWNlZCBnbG9iYWwgbmFtZXNwYWNlIHBvbGx1dGlvblxuXG4yMDEwLjA1LjIyIC0gMC42OlxuIC0gcmV2ZXJ0ZWQgdG8gMC40IGFuZCBmaXhlZCB0aGUgYnVnIHJlZ2FyZGluZyB0aGUgc2lnbiBvZiB0aGUgbnVtYmVyIDBcbiBOb3RlOlxuIFRoYW5rcyB0byBSYXBoYWVsIFBpZ3VsbGEgPHJhcGggKGF0XSBuM3JkIFtkb3QpIG9yZz4gKGh0dHA6Ly93d3cubjNyZC5vcmcvKVxuIHdobyB3YXJuZWQgbWUgYWJvdXQgYSBidWcgaW4gMC41LCBJIGRpc2NvdmVyZWQgdGhhdCB0aGUgbGFzdCB1cGRhdGUgd2FzXG4gYSByZWdyZXNzLiBJIGFwcG9sb2dpemUgZm9yIHRoYXQuXG5cbjIwMTAuMDUuMDkgLSAwLjU6XG4gLSBidWcgZml4OiAwIGlzIG5vdyBwcmVjZWVkZWQgd2l0aCBhICsgc2lnblxuIC0gYnVnIGZpeDogdGhlIHNpZ24gd2FzIG5vdCBhdCB0aGUgcmlnaHQgcG9zaXRpb24gb24gcGFkZGVkIHJlc3VsdHMgKEthbWFsIEFiZGFsaSlcbiAtIHN3aXRjaGVkIGZyb20gR1BMIHRvIEJTRCBsaWNlbnNlXG5cbjIwMDcuMTAuMjEgLSAwLjQ6XG4gLSB1bml0IHRlc3QgYW5kIHBhdGNoIChEYXZpZCBCYWlyZClcblxuMjAwNy4wOS4xNyAtIDAuMzpcbiAtIGJ1ZyBmaXg6IG5vIGxvbmdlciB0aHJvd3MgZXhjZXB0aW9uIG9uIGVtcHR5IHBhcmFtZW50ZXJzIChIYW5zIFB1ZmFsKVxuXG4yMDA3LjA5LjExIC0gMC4yOlxuIC0gZmVhdHVyZTogYWRkZWQgYXJndW1lbnQgc3dhcHBpbmdcblxuMjAwNy4wNC4wMyAtIDAuMTpcbiAtIGluaXRpYWwgcmVsZWFzZVxuKiovXG5cbnZhciBzcHJpbnRmID0gKGZ1bmN0aW9uKCkge1xuXHRmdW5jdGlvbiBnZXRfdHlwZSh2YXJpYWJsZSkge1xuXHRcdHJldHVybiBPYmplY3QucHJvdG90eXBlLnRvU3RyaW5nLmNhbGwodmFyaWFibGUpLnNsaWNlKDgsIC0xKS50b0xvd2VyQ2FzZSgpO1xuXHR9XG5cdGZ1bmN0aW9uIHN0cl9yZXBlYXQoaW5wdXQsIG11bHRpcGxpZXIpIHtcblx0XHRmb3IgKHZhciBvdXRwdXQgPSBbXTsgbXVsdGlwbGllciA+IDA7IG91dHB1dFstLW11bHRpcGxpZXJdID0gaW5wdXQpIHsvKiBkbyBub3RoaW5nICovfVxuXHRcdHJldHVybiBvdXRwdXQuam9pbignJyk7XG5cdH1cblxuXHR2YXIgc3RyX2Zvcm1hdCA9IGZ1bmN0aW9uKCkge1xuXHRcdGlmICghc3RyX2Zvcm1hdC5jYWNoZS5oYXNPd25Qcm9wZXJ0eShhcmd1bWVudHNbMF0pKSB7XG5cdFx0XHRzdHJfZm9ybWF0LmNhY2hlW2FyZ3VtZW50c1swXV0gPSBzdHJfZm9ybWF0LnBhcnNlKGFyZ3VtZW50c1swXSk7XG5cdFx0fVxuXHRcdHJldHVybiBzdHJfZm9ybWF0LmZvcm1hdC5jYWxsKG51bGwsIHN0cl9mb3JtYXQuY2FjaGVbYXJndW1lbnRzWzBdXSwgYXJndW1lbnRzKTtcblx0fTtcblxuXHQvLyBjb252ZXJ0IG9iamVjdCB0byBzaW1wbGUgb25lIGxpbmUgc3RyaW5nIHdpdGhvdXQgaW5kZW50YXRpb24gb3Jcblx0Ly8gbmV3bGluZXMuIE5vdGUgdGhhdCB0aGlzIGltcGxlbWVudGF0aW9uIGRvZXMgbm90IHByaW50IGFycmF5XG5cdC8vIHZhbHVlcyB0byB0aGVpciBhY3R1YWwgcGxhY2UgZm9yIHNwYXJzZSBhcnJheXMuIFxuXHQvL1xuXHQvLyBGb3IgZXhhbXBsZSBzcGFyc2UgYXJyYXkgbGlrZSB0aGlzXG5cdC8vICAgIGwgPSBbXVxuXHQvLyAgICBsWzRdID0gMVxuXHQvLyBXb3VsZCBiZSBwcmludGVkIGFzIFwiWzFdXCIgaW5zdGVhZCBvZiBcIlssICwgLCAsIDFdXCJcblx0Ly8gXG5cdC8vIElmIGFyZ3VtZW50ICdzZWVuJyBpcyBub3QgbnVsbCBhbmQgYXJyYXkgdGhlIGZ1bmN0aW9uIHdpbGwgY2hlY2sgZm9yIFxuXHQvLyBjaXJjdWxhciBvYmplY3QgcmVmZXJlbmNlcyBmcm9tIGFyZ3VtZW50LlxuXHRzdHJfZm9ybWF0Lm9iamVjdF9zdHJpbmdpZnkgPSBmdW5jdGlvbihvYmosIGRlcHRoLCBtYXhkZXB0aCwgc2Vlbikge1xuXHRcdHZhciBzdHIgPSAnJztcblx0XHRpZiAob2JqICE9IG51bGwpIHtcblx0XHRcdHN3aXRjaCggdHlwZW9mKG9iaikgKSB7XG5cdFx0XHRjYXNlICdmdW5jdGlvbic6IFxuXHRcdFx0XHRyZXR1cm4gJ1tGdW5jdGlvbicgKyAob2JqLm5hbWUgPyAnOiAnK29iai5uYW1lIDogJycpICsgJ10nO1xuXHRcdFx0ICAgIGJyZWFrO1xuXHRcdFx0Y2FzZSAnb2JqZWN0Jzpcblx0XHRcdFx0aWYgKCBvYmogaW5zdGFuY2VvZiBFcnJvcikgeyByZXR1cm4gJ1snICsgb2JqLnRvU3RyaW5nKCkgKyAnXScgfTtcblx0XHRcdFx0aWYgKGRlcHRoID49IG1heGRlcHRoKSByZXR1cm4gJ1tPYmplY3RdJ1xuXHRcdFx0XHRpZiAoc2Vlbikge1xuXHRcdFx0XHRcdC8vIGFkZCBvYmplY3QgdG8gc2VlbiBsaXN0XG5cdFx0XHRcdFx0c2VlbiA9IHNlZW4uc2xpY2UoMClcblx0XHRcdFx0XHRzZWVuLnB1c2gob2JqKTtcblx0XHRcdFx0fVxuXHRcdFx0XHRpZiAob2JqLmxlbmd0aCAhPSBudWxsKSB7IC8vYXJyYXlcblx0XHRcdFx0XHRzdHIgKz0gJ1snO1xuXHRcdFx0XHRcdHZhciBhcnIgPSBbXVxuXHRcdFx0XHRcdGZvciAodmFyIGkgaW4gb2JqKSB7XG5cdFx0XHRcdFx0XHRpZiAoc2VlbiAmJiBzZWVuLmluZGV4T2Yob2JqW2ldKSA+PSAwKSBhcnIucHVzaCgnW0NpcmN1bGFyXScpO1xuXHRcdFx0XHRcdFx0ZWxzZSBhcnIucHVzaChzdHJfZm9ybWF0Lm9iamVjdF9zdHJpbmdpZnkob2JqW2ldLCBkZXB0aCsxLCBtYXhkZXB0aCwgc2VlbikpO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0XHRzdHIgKz0gYXJyLmpvaW4oJywgJykgKyAnXSc7XG5cdFx0XHRcdH0gZWxzZSBpZiAoJ2dldE1vbnRoJyBpbiBvYmopIHsgLy8gZGF0ZVxuXHRcdFx0XHRcdHJldHVybiAnRGF0ZSgnICsgb2JqICsgJyknO1xuXHRcdFx0XHR9IGVsc2UgeyAvLyBvYmplY3Rcblx0XHRcdFx0XHRzdHIgKz0gJ3snO1xuXHRcdFx0XHRcdHZhciBhcnIgPSBbXVxuXHRcdFx0XHRcdGZvciAodmFyIGsgaW4gb2JqKSB7IFxuXHRcdFx0XHRcdFx0aWYob2JqLmhhc093blByb3BlcnR5KGspKSB7XG5cdFx0XHRcdFx0XHRcdGlmIChzZWVuICYmIHNlZW4uaW5kZXhPZihvYmpba10pID49IDApIGFyci5wdXNoKGsgKyAnOiBbQ2lyY3VsYXJdJyk7XG5cdFx0XHRcdFx0XHRcdGVsc2UgYXJyLnB1c2goayArJzogJyArc3RyX2Zvcm1hdC5vYmplY3Rfc3RyaW5naWZ5KG9ialtrXSwgZGVwdGgrMSwgbWF4ZGVwdGgsIHNlZW4pKTsgXG5cdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHRcdHN0ciArPSBhcnIuam9pbignLCAnKSArICd9Jztcblx0XHRcdFx0fVxuXHRcdFx0XHRyZXR1cm4gc3RyO1xuXHRcdFx0XHRicmVhaztcblx0XHRcdGNhc2UgJ3N0cmluZyc6XHRcdFx0XHRcblx0XHRcdFx0cmV0dXJuICdcIicgKyBvYmogKyAnXCInO1xuXHRcdFx0XHRicmVha1xuXHRcdFx0fVxuXHRcdH1cblx0XHRyZXR1cm4gJycgKyBvYmo7XG5cdH1cblxuXHRzdHJfZm9ybWF0LmZvcm1hdCA9IGZ1bmN0aW9uKHBhcnNlX3RyZWUsIGFyZ3YpIHtcblx0XHR2YXIgY3Vyc29yID0gMSwgdHJlZV9sZW5ndGggPSBwYXJzZV90cmVlLmxlbmd0aCwgbm9kZV90eXBlID0gJycsIGFyZywgb3V0cHV0ID0gW10sIGksIGssIG1hdGNoLCBwYWQsIHBhZF9jaGFyYWN0ZXIsIHBhZF9sZW5ndGg7XG5cdFx0Zm9yIChpID0gMDsgaSA8IHRyZWVfbGVuZ3RoOyBpKyspIHtcblx0XHRcdG5vZGVfdHlwZSA9IGdldF90eXBlKHBhcnNlX3RyZWVbaV0pO1xuXHRcdFx0aWYgKG5vZGVfdHlwZSA9PT0gJ3N0cmluZycpIHtcblx0XHRcdFx0b3V0cHV0LnB1c2gocGFyc2VfdHJlZVtpXSk7XG5cdFx0XHR9XG5cdFx0XHRlbHNlIGlmIChub2RlX3R5cGUgPT09ICdhcnJheScpIHtcblx0XHRcdFx0bWF0Y2ggPSBwYXJzZV90cmVlW2ldOyAvLyBjb252ZW5pZW5jZSBwdXJwb3NlcyBvbmx5XG5cdFx0XHRcdGlmIChtYXRjaFsyXSkgeyAvLyBrZXl3b3JkIGFyZ3VtZW50XG5cdFx0XHRcdFx0YXJnID0gYXJndltjdXJzb3JdO1xuXHRcdFx0XHRcdGZvciAoayA9IDA7IGsgPCBtYXRjaFsyXS5sZW5ndGg7IGsrKykge1xuXHRcdFx0XHRcdFx0aWYgKCFhcmcuaGFzT3duUHJvcGVydHkobWF0Y2hbMl1ba10pKSB7XG5cdFx0XHRcdFx0XHRcdHRocm93IG5ldyBFcnJvcihzcHJpbnRmKCdbc3ByaW50Zl0gcHJvcGVydHkgXCIlc1wiIGRvZXMgbm90IGV4aXN0JywgbWF0Y2hbMl1ba10pKTtcblx0XHRcdFx0XHRcdH1cblx0XHRcdFx0XHRcdGFyZyA9IGFyZ1ttYXRjaFsyXVtrXV07XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9XG5cdFx0XHRcdGVsc2UgaWYgKG1hdGNoWzFdKSB7IC8vIHBvc2l0aW9uYWwgYXJndW1lbnQgKGV4cGxpY2l0KVxuXHRcdFx0XHRcdGFyZyA9IGFyZ3ZbbWF0Y2hbMV1dO1xuXHRcdFx0XHR9XG5cdFx0XHRcdGVsc2UgeyAvLyBwb3NpdGlvbmFsIGFyZ3VtZW50IChpbXBsaWNpdClcblx0XHRcdFx0XHRhcmcgPSBhcmd2W2N1cnNvcisrXTtcblx0XHRcdFx0fVxuXG5cdFx0XHRcdGlmICgvW15zT10vLnRlc3QobWF0Y2hbOF0pICYmIChnZXRfdHlwZShhcmcpICE9ICdudW1iZXInKSkge1xuXHRcdFx0XHRcdHRocm93IG5ldyBFcnJvcihzcHJpbnRmKCdbc3ByaW50Zl0gZXhwZWN0aW5nIG51bWJlciBidXQgZm91bmQgJXMgXCInICsgYXJnICsgJ1wiJywgZ2V0X3R5cGUoYXJnKSkpO1xuXHRcdFx0XHR9XG5cdFx0XHRcdHN3aXRjaCAobWF0Y2hbOF0pIHtcblx0XHRcdFx0XHRjYXNlICdiJzogYXJnID0gYXJnLnRvU3RyaW5nKDIpOyBicmVhaztcblx0XHRcdFx0XHRjYXNlICdjJzogYXJnID0gU3RyaW5nLmZyb21DaGFyQ29kZShhcmcpOyBicmVhaztcblx0XHRcdFx0XHRjYXNlICdkJzogYXJnID0gcGFyc2VJbnQoYXJnLCAxMCk7IGJyZWFrO1xuXHRcdFx0XHRcdGNhc2UgJ2UnOiBhcmcgPSBtYXRjaFs3XSA/IGFyZy50b0V4cG9uZW50aWFsKG1hdGNoWzddKSA6IGFyZy50b0V4cG9uZW50aWFsKCk7IGJyZWFrO1xuXHRcdFx0XHRcdGNhc2UgJ2YnOiBhcmcgPSBtYXRjaFs3XSA/IHBhcnNlRmxvYXQoYXJnKS50b0ZpeGVkKG1hdGNoWzddKSA6IHBhcnNlRmxvYXQoYXJnKTsgYnJlYWs7XG5cdFx0XHRcdCAgICBjYXNlICdPJzogYXJnID0gc3RyX2Zvcm1hdC5vYmplY3Rfc3RyaW5naWZ5KGFyZywgMCwgcGFyc2VJbnQobWF0Y2hbN10pIHx8IDUpOyBicmVhaztcblx0XHRcdFx0XHRjYXNlICdvJzogYXJnID0gYXJnLnRvU3RyaW5nKDgpOyBicmVhaztcblx0XHRcdFx0XHRjYXNlICdzJzogYXJnID0gKChhcmcgPSBTdHJpbmcoYXJnKSkgJiYgbWF0Y2hbN10gPyBhcmcuc3Vic3RyaW5nKDAsIG1hdGNoWzddKSA6IGFyZyk7IGJyZWFrO1xuXHRcdFx0XHRcdGNhc2UgJ3UnOiBhcmcgPSBNYXRoLmFicyhhcmcpOyBicmVhaztcblx0XHRcdFx0XHRjYXNlICd4JzogYXJnID0gYXJnLnRvU3RyaW5nKDE2KTsgYnJlYWs7XG5cdFx0XHRcdFx0Y2FzZSAnWCc6IGFyZyA9IGFyZy50b1N0cmluZygxNikudG9VcHBlckNhc2UoKTsgYnJlYWs7XG5cdFx0XHRcdH1cblx0XHRcdFx0YXJnID0gKC9bZGVmXS8udGVzdChtYXRjaFs4XSkgJiYgbWF0Y2hbM10gJiYgYXJnID49IDAgPyAnKycrIGFyZyA6IGFyZyk7XG5cdFx0XHRcdHBhZF9jaGFyYWN0ZXIgPSBtYXRjaFs0XSA/IG1hdGNoWzRdID09ICcwJyA/ICcwJyA6IG1hdGNoWzRdLmNoYXJBdCgxKSA6ICcgJztcblx0XHRcdFx0cGFkX2xlbmd0aCA9IG1hdGNoWzZdIC0gU3RyaW5nKGFyZykubGVuZ3RoO1xuXHRcdFx0XHRwYWQgPSBtYXRjaFs2XSA/IHN0cl9yZXBlYXQocGFkX2NoYXJhY3RlciwgcGFkX2xlbmd0aCkgOiAnJztcblx0XHRcdFx0b3V0cHV0LnB1c2gobWF0Y2hbNV0gPyBhcmcgKyBwYWQgOiBwYWQgKyBhcmcpO1xuXHRcdFx0fVxuXHRcdH1cblx0XHRyZXR1cm4gb3V0cHV0LmpvaW4oJycpO1xuXHR9O1xuXG5cdHN0cl9mb3JtYXQuY2FjaGUgPSB7fTtcblxuXHRzdHJfZm9ybWF0LnBhcnNlID0gZnVuY3Rpb24oZm10KSB7XG5cdFx0dmFyIF9mbXQgPSBmbXQsIG1hdGNoID0gW10sIHBhcnNlX3RyZWUgPSBbXSwgYXJnX25hbWVzID0gMDtcblx0XHR3aGlsZSAoX2ZtdCkge1xuXHRcdFx0aWYgKChtYXRjaCA9IC9eW15cXHgyNV0rLy5leGVjKF9mbXQpKSAhPT0gbnVsbCkge1xuXHRcdFx0XHRwYXJzZV90cmVlLnB1c2gobWF0Y2hbMF0pO1xuXHRcdFx0fVxuXHRcdFx0ZWxzZSBpZiAoKG1hdGNoID0gL15cXHgyNXsyfS8uZXhlYyhfZm10KSkgIT09IG51bGwpIHtcblx0XHRcdFx0cGFyc2VfdHJlZS5wdXNoKCclJyk7XG5cdFx0XHR9XG5cdFx0XHRlbHNlIGlmICgobWF0Y2ggPSAvXlxceDI1KD86KFsxLTldXFxkKilcXCR8XFwoKFteXFwpXSspXFwpKT8oXFwrKT8oMHwnW14kXSk/KC0pPyhcXGQrKT8oPzpcXC4oXFxkKykpPyhbYi1mb3NPdXhYXSkvLmV4ZWMoX2ZtdCkpICE9PSBudWxsKSB7XG5cdFx0XHRcdGlmIChtYXRjaFsyXSkge1xuXHRcdFx0XHRcdGFyZ19uYW1lcyB8PSAxO1xuXHRcdFx0XHRcdHZhciBmaWVsZF9saXN0ID0gW10sIHJlcGxhY2VtZW50X2ZpZWxkID0gbWF0Y2hbMl0sIGZpZWxkX21hdGNoID0gW107XG5cdFx0XHRcdFx0aWYgKChmaWVsZF9tYXRjaCA9IC9eKFthLXpfXVthLXpfXFxkXSopL2kuZXhlYyhyZXBsYWNlbWVudF9maWVsZCkpICE9PSBudWxsKSB7XG5cdFx0XHRcdFx0XHRmaWVsZF9saXN0LnB1c2goZmllbGRfbWF0Y2hbMV0pO1xuXHRcdFx0XHRcdFx0d2hpbGUgKChyZXBsYWNlbWVudF9maWVsZCA9IHJlcGxhY2VtZW50X2ZpZWxkLnN1YnN0cmluZyhmaWVsZF9tYXRjaFswXS5sZW5ndGgpKSAhPT0gJycpIHtcblx0XHRcdFx0XHRcdFx0aWYgKChmaWVsZF9tYXRjaCA9IC9eXFwuKFthLXpfXVthLXpfXFxkXSopL2kuZXhlYyhyZXBsYWNlbWVudF9maWVsZCkpICE9PSBudWxsKSB7XG5cdFx0XHRcdFx0XHRcdFx0ZmllbGRfbGlzdC5wdXNoKGZpZWxkX21hdGNoWzFdKTtcblx0XHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdFx0XHRlbHNlIGlmICgoZmllbGRfbWF0Y2ggPSAvXlxcWyhcXGQrKVxcXS8uZXhlYyhyZXBsYWNlbWVudF9maWVsZCkpICE9PSBudWxsKSB7XG5cdFx0XHRcdFx0XHRcdFx0ZmllbGRfbGlzdC5wdXNoKGZpZWxkX21hdGNoWzFdKTtcblx0XHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdFx0XHRlbHNlIHtcblx0XHRcdFx0XHRcdFx0XHR0aHJvdyBuZXcgRXJyb3IoJ1tzcHJpbnRmXSAnICsgcmVwbGFjZW1lbnRfZmllbGQpO1xuXHRcdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHRcdGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdbc3ByaW50Zl0gJyArIHJlcGxhY2VtZW50X2ZpZWxkKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0bWF0Y2hbMl0gPSBmaWVsZF9saXN0O1xuXHRcdFx0XHR9XG5cdFx0XHRcdGVsc2Uge1xuXHRcdFx0XHRcdGFyZ19uYW1lcyB8PSAyO1xuXHRcdFx0XHR9XG5cdFx0XHRcdGlmIChhcmdfbmFtZXMgPT09IDMpIHtcblx0XHRcdFx0XHR0aHJvdyBuZXcgRXJyb3IoJ1tzcHJpbnRmXSBtaXhpbmcgcG9zaXRpb25hbCBhbmQgbmFtZWQgcGxhY2Vob2xkZXJzIGlzIG5vdCAoeWV0KSBzdXBwb3J0ZWQnKTtcblx0XHRcdFx0fVxuXHRcdFx0XHRwYXJzZV90cmVlLnB1c2gobWF0Y2gpO1xuXHRcdFx0fVxuXHRcdFx0ZWxzZSB7XG5cdFx0XHRcdHRocm93IG5ldyBFcnJvcignW3NwcmludGZdICcgKyBfZm10KTtcblx0XHRcdH1cblx0XHRcdF9mbXQgPSBfZm10LnN1YnN0cmluZyhtYXRjaFswXS5sZW5ndGgpO1xuXHRcdH1cblx0XHRyZXR1cm4gcGFyc2VfdHJlZTtcblx0fTtcblxuXHRyZXR1cm4gc3RyX2Zvcm1hdDtcbn0pKCk7XG5cbnZhciB2c3ByaW50ZiA9IGZ1bmN0aW9uKGZtdCwgYXJndikge1xuXHR2YXIgYXJndkNsb25lID0gYXJndi5zbGljZSgpO1xuXHRhcmd2Q2xvbmUudW5zaGlmdChmbXQpO1xuXHRyZXR1cm4gc3ByaW50Zi5hcHBseShudWxsLCBhcmd2Q2xvbmUpO1xufTtcblxubW9kdWxlLmV4cG9ydHMgPSBzcHJpbnRmO1xuc3ByaW50Zi5zcHJpbnRmID0gc3ByaW50ZjtcbnNwcmludGYudnNwcmludGYgPSB2c3ByaW50ZjtcbiIsImltcG9ydCB7IHJlYWRGaWxlU3luYyB9IGZyb20gJ2ZzJztcblxuZXhwb3J0IGRlZmF1bHQge1xuXHRoYXJkX3NpZ21vaWQ6IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL2hhcmRfc2lnbW9pZC5nbHNsJywgJ3V0ZjgnKSxcblx0bGluZWFyOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9saW5lYXIuZ2xzbCcsICd1dGY4JyksXG5cdHJlbHU6IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3JlbHUuZ2xzbCcsICd1dGY4JyksXG5cdHJnYjogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvcmdiLmdsc2wnLCAndXRmOCcpLFxuXHRzaWdtb2lkOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9zaWdtb2lkLmdsc2wnLCAndXRmOCcpLFxuXHR0YW5oOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy90YW5oLmdsc2wnLCAndXRmOCcpLFxufSIsImltcG9ydCB7IHJlYWRGaWxlU3luYyB9IGZyb20gJ2ZzJztcblxuZXhwb3J0IGNvbnN0IGVuY29kZVNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL2VuY29kZS5nbHNsJywgJ3V0ZjgnKTtcbmV4cG9ydCBjb25zdCBkZWNvZGVTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9kZWNvZGUuZ2xzbCcsICd1dGY4Jyk7XG5cbmV4cG9ydCBmdW5jdGlvbiBpbml0KHNoYXBlLCBmb3JtYXQpe1xuXHRyZXR1cm4ge1xuXHRcdHJhbmdlOiBmb3JtYXQucmFuZ2UgfHwgNDA5NlxuXHR9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBlbmNvZGUoYnVmLCB2YWx1ZSwgaW5mbyl7XG5cdHZhciB6ID0gTWF0aC5taW4oMSwgTWF0aC5tYXgoMCwgdmFsdWUgLyBpbmZvLnJhbmdlICsgMC41KSk7XG5cdGJ1ZlswXSA9ICh6ICogMjU2ICogMjU2ICogMjU2ICogMjU2KSAlIDI1NlxuXHRidWZbMV0gPSAoeiAqIDI1NiAqIDI1NiAqIDI1NikgJSAyNTZcblx0YnVmWzJdID0gKHogKiAyNTYgKiAyNTYpICUgMjU2XG5cdGJ1ZlszXSA9ICh6ICogMjU2KSAlIDI1NlxufVxuXG5cbmV4cG9ydCBmdW5jdGlvbiBkZWNvZGUoYnVmKXtcblx0cmV0dXJuIGJ1ZlswXSAvIDI1Ni4wIC8gMjU2LjAgLyAyNTYuMCAvIDI1Ni4wICtcblx0XHQgICBidWZbMV0gLyAyNTYuMCAvIDI1Ni4wIC8gMjU2LjAgK1xuXHRcdCAgIGJ1ZlsyXSAvIDI1Ni4wIC8gMjU2LjAgK1xuXHRcdCAgIGJ1ZlszXSAvIDI1Ni4wO1xufVxuIiwiaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xuXG5leHBvcnQgY29uc3QgZW5jb2RlU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvZW5jb2RlLmdsc2wnLCAndXRmOCcpO1xuZXhwb3J0IGNvbnN0IGRlY29kZVNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL2RlY29kZS5nbHNsJywgJ3V0ZjgnKTtcblxuZXhwb3J0IGZ1bmN0aW9uIGluaXQoc2hhcGUsIGZvcm1hdCl7XG5cdHJldHVybiB7IH1cbn1cblxudmFyIHRtcF9mbG9hdCA9IG5ldyBGbG9hdDMyQXJyYXkoMSksXG5cdHRtcF9pbnQgPSBuZXcgVWludDhBcnJheSh0bXBfZmxvYXQuYnVmZmVyKTtcblxuZXhwb3J0IGZ1bmN0aW9uIGVuY29kZShidWYsIHZhbHVlKXtcblx0dG1wX2Zsb2F0WzBdID0gdmFsdWU7XG5cdGJ1Zi5zZXQodG1wX2ludCwgMClcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGRlY29kZShidWYpe1xuXHR0bXBfaW50LnNldChidWYpXG5cdHJldHVybiB0bXBfZmxvYXRbMF1cbn0iLCJpbXBvcnQgKiBhcyBwYWNrX3N0cmlkZSBmcm9tICcuL3BhY2svc3RyaWRlL2luZGV4LmpzJ1xuaW1wb3J0ICogYXMgcGFja190aWxlIGZyb20gJy4vcGFjay90aWxlL2luZGV4LmpzJ1xuXG5pbXBvcnQgKiBhcyBjb2RlY19maXhudW0gZnJvbSAnLi9jb2RlYy9maXhudW0vaW5kZXguanMnXG5pbXBvcnQgKiBhcyBjb2RlY19zb2Z0ZmxvYXQgZnJvbSAnLi9jb2RlYy9zb2Z0ZmxvYXQvaW5kZXguanMnXG5cbmltcG9ydCBhY3RpdmF0aW9ucyBmcm9tICcuL2FjdGl2YXRpb24vaW5kZXguanMnXG5cbmltcG9ydCB7IHJlYWRGaWxlU3luYyB9IGZyb20gJ2ZzJztcblxuZXhwb3J0IGRlZmF1bHQge1xuXHRwYWNrOiB7XG5cdFx0c3RyaWRlOiBwYWNrX3N0cmlkZSxcblx0XHR0aWxlOiBwYWNrX3RpbGVcblx0fSxcblxuXHRyZWFkX3NoaW06IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3BhY2svcmVhZF9zaGltLmdsc2wnLCAndXRmOCcpLFxuXHR3cml0ZV9zaGltOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9wYWNrL3dyaXRlX3NoaW0uZ2xzbCcsICd1dGY4JyksXG5cblx0Y29kZWM6IHtcblx0XHRmaXhudW06IGNvZGVjX2ZpeG51bSxcblx0XHRzb2Z0ZmxvYXQ6IGNvZGVjX3NvZnRmbG9hdCxcblx0fSxcblx0YWN0aXZhdGlvbnM6IGFjdGl2YXRpb25zXG59IiwiaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xuaW1wb3J0IG5kYXJyYXkgZnJvbSAnbmRhcnJheSdcblxuZXhwb3J0IGNvbnN0IHJlYWRTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9yZWFkLmdsc2wnLCAndXRmOCcpO1xuZXhwb3J0IGNvbnN0IHdyaXRlU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvd3JpdGUuZ2xzbCcsICd1dGY4Jyk7XG5cbmV4cG9ydCBmdW5jdGlvbiBpbml0KHNoYXBlKXtcbiAgICAvLyB2YXIgbGVuZ3RoID0gNCAqIE1hdGguY2VpbChzaGFwZVsyXSAvIDQpICogc2hhcGVbM10gKiBzaGFwZVsxXSAqIHNoYXBlWzBdO1xuICAgIC8vIHZhciBjb2xzID0gTWF0aC5jZWlsKE1hdGguc3FydChsZW5ndGgpIC8gNCkgKiA0O1xuXG4gICAgdmFyIGxlbmd0aCA9IHNoYXBlWzJdICogc2hhcGVbM10gKiBzaGFwZVsxXSAqIHNoYXBlWzBdO1xuICAgIHZhciBjb2xzID0gTWF0aC5jZWlsKE1hdGguc3FydChsZW5ndGgpKTtcbiAgICB2YXIgdGV4U2l6ZSA9IFtjb2xzLCBNYXRoLmNlaWwobGVuZ3RoIC8gY29scyldXG4gICAgcmV0dXJuIHtcbiAgICAgICAgdGV4U2l6ZTogdGV4U2l6ZSxcbiAgICAgICAgc2hhcGU6IHNoYXBlLFxuICAgICAgICAvLyB2ZWM0KDEsIEBzaGFwZS54LCBAc2hhcGUueCAqIEBzaGFwZS55LCBAc2hhcGUueCAqIEBzaGFwZS55ICogQHNoYXBlLnopXG4gICAgICAgIHN0cmlkZTogWzEsIHNoYXBlWzBdLCBzaGFwZVswXSAqIHNoYXBlWzFdLCBzaGFwZVswXSAqIHNoYXBlWzFdICogc2hhcGVbMl1dXG4gICAgfVxufVxuXG5cbmV4cG9ydCBmdW5jdGlvbiBwYWNrKGluZm8sIGFycmF5LCBlbmNvZGUxLCBmb3JtYXQpe1xuICAgIC8vIHJldHVybiBVaW50OEFycmF5IG9yIEZsb2F0MzJBcnJheVxuICAgIGFycmF5ID0gbmRhcnJheShhcnJheS5kYXRhLCBcbiAgICAgICAgYXJyYXkuc2hhcGUuY29uY2F0KFsxLCAxLCAxLCAxXSkuc2xpY2UoMCwgNCksXG4gICAgICAgIGFycmF5LnN0cmlkZS5jb25jYXQoWzEsIDEsIDEsIDFdKS5zbGljZSgwLCA0KSxcbiAgICAgICAgYXJyYXkub2Zmc2V0KVxuXG4gICAgdmFyIHNoYXBlID0gaW5mby5zaGFwZTtcbiAgICB2YXIgbGVuZ3RoID0gaW5mby50ZXhTaXplWzBdICogaW5mby50ZXhTaXplWzFdICogNDtcblxuICAgIGlmKGZvcm1hdC50eXBlID09PSAnZmxvYXQzMicpe1xuICAgICAgICB2YXIgZGF0YSA9IG5ldyBGbG9hdDMyQXJyYXkobGVuZ3RoKTsgICAgXG4gICAgfWVsc2UgaWYoZm9ybWF0LnR5cGUgPT09ICd1aW50OCcpe1xuICAgICAgICB2YXIgZGF0YSA9IG5ldyBVaW50OEFycmF5KGxlbmd0aCk7ICAgIFxuICAgIH1cblxuICAgIGZvcih2YXIgeCA9IDA7IHggPCBzaGFwZVswXTsgeCsrKXtcbiAgICAgICAgZm9yKHZhciB5ID0gMDsgeSA8IHNoYXBlWzFdOyB5Kyspe1xuICAgICAgICAgICAgZm9yKHZhciB6ID0gMDsgeiA8IHNoYXBlWzJdOyB6Kyspe1xuICAgICAgICAgICAgICAgIGZvcih2YXIgdyA9IDA7IHcgPCBzaGFwZVszXTsgdysrKXtcbiAgICAgICAgICAgICAgICAgICAgdmFyIHRpbGUgID0geCArIFxuICAgICAgICAgICAgICAgICAgICAgICAgeSAqIHNoYXBlWzBdICsgXG4gICAgICAgICAgICAgICAgICAgICAgICB6ICogc2hhcGVbMF0gKiBzaGFwZVsxXSArXG4gICAgICAgICAgICAgICAgICAgICAgICB3ICogc2hhcGVbMF0gKiBzaGFwZVsxXSAqIHNoYXBlWzJdO1xuXG4gICAgICAgICAgICAgICAgICAgIGVuY29kZTEoZGF0YS5zdWJhcnJheSg0KnRpbGUsIDQqdGlsZSs0KSwgYXJyYXkuZ2V0KHgsIHksIHosIHcpLCBpbmZvKVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiBkYXRhO1xufVxuXG5cbmV4cG9ydCBmdW5jdGlvbiB1bnBhY2soaW5mbywgZGF0YSwgZGVjb2RlMSwgdHlwZSl7XG4gICAgaWYodHlwZSAhPSAnZmxvYXQzMicpIHRocm93IG5ldyBFcnJvcignbm90IGltcGwnKTtcblxuICAgIHZhciBzaGFwZSA9IGluZm8uc2hhcGU7XG4gICAgdmFyIGxlbmd0aCA9IHNoYXBlLnJlZHVjZSgoYSwgYikgPT4gYSAqIGIpO1xuXG4gICAgdmFyIGFycmF5ID0gbmRhcnJheShuZXcgRmxvYXQzMkFycmF5KGxlbmd0aCksIFxuICAgICAgICBzaGFwZS5jb25jYXQoWzEsIDEsIDEsIDFdKS5zbGljZSgwLCA0KSlcblxuXG4gICAgZm9yKHZhciB4ID0gMDsgeCA8IHNoYXBlWzBdOyB4Kyspe1xuICAgICAgICBmb3IodmFyIHkgPSAwOyB5IDwgc2hhcGVbMV07IHkrKyl7XG4gICAgICAgICAgICBmb3IodmFyIHogPSAwOyB6IDwgc2hhcGVbMl07IHorKyl7XG4gICAgICAgICAgICAgICAgZm9yKHZhciB3ID0gMDsgdyA8IHNoYXBlWzNdOyB3Kyspe1xuICAgICAgICAgICAgICAgICAgICB2YXIgdGlsZSAgPSB4ICsgXG4gICAgICAgICAgICAgICAgICAgICAgICB5ICogc2hhcGVbMF0gKyBcbiAgICAgICAgICAgICAgICAgICAgICAgIHogKiBzaGFwZVswXSAqIHNoYXBlWzFdICtcbiAgICAgICAgICAgICAgICAgICAgICAgIHcgKiBzaGFwZVswXSAqIHNoYXBlWzFdICogc2hhcGVbMl07XG5cbiAgICAgICAgICAgICAgICAgICAgYXJyYXkuc2V0KHgsIHksIHosIHcsIGRlY29kZTEoZGF0YS5zdWJhcnJheSg0KnRpbGUsIDQqdGlsZSs0KSwgaW5mbykpXG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgfVxuICAgIHJldHVybiBhcnJheTtcbn0iLCJpbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5cbmV4cG9ydCBjb25zdCByZWFkU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvcmVhZC5nbHNsJywgJ3V0ZjgnKTtcbmV4cG9ydCBjb25zdCB3cml0ZVNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3dyaXRlLmdsc2wnLCAndXRmOCcpO1xuaW1wb3J0IG5kYXJyYXkgZnJvbSAnbmRhcnJheSdcblxuXG5leHBvcnQgZnVuY3Rpb24gaW5pdChzaGFwZSl7XG4gICAgdmFyIHdpZHRoID0gc2hhcGVbMF07XG4gICAgLy8gd2UgcGljayB0aGUgbnVtYmVyIG9mIGNvbHVtbnMgc28gd2UgY2FuIGtlZXBcbiAgICAvLyB0aGUgdGV4dHVyZSBhcyBzcXVhcmUgYXMgcG9zc2libGUsIHdpdGggdGhlXG4gICAgLy8gbWluaW1hbCBhbW91bnQgb2Ygd2FzdGVkIHNwYWNlLlxuXG4gICAgdmFyIHRpbGVzID0gc2hhcGVbMl0gKiBzaGFwZVszXSxcbiAgICAgICAgY29scyA9IE1hdGgubWF4KDEsIE1hdGgubWluKHRpbGVzLCBNYXRoLmNlaWwoXG4gICAgICAgICAgICBNYXRoLnNxcnQoc2hhcGVbMF0gKiBzaGFwZVsxXSAqIHRpbGVzKSAvIHdpZHRoKSkpO1xuXG4gICAgdmFyIHRleFNpemUgPSBbd2lkdGggKiBjb2xzLCBzaGFwZVsxXSAqIE1hdGguY2VpbCh0aWxlcyAvIGNvbHMpXVxuXG4gICAgcmV0dXJuIHtcbiAgICAgICAgdGV4U2l6ZTogdGV4U2l6ZSxcbiAgICAgICAgY29sczogY29scyxcbiAgICAgICAgc2hhcGU6IHNoYXBlLFxuICAgIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHBhY2soaW5mbywgbmRhcnJheSl7XG4gICAgLy8gcmV0dXJuIFVpbnQ4QXJyYXkgb3IgRmxvYXQzMkFycmF5XG5cblxuLy8gdW5pZm9ybSBzYW1wbGVyMkQgQF90ZXg7XG4vLyB1bmlmb3JtIGl2ZWMyIEBfdGV4U2l6ZTtcbi8vIHVuaWZvcm0gaXZlYzQgQF9zaGFwZTtcbi8vIHVuaWZvcm0gaW50IEBfY29scztcblxuICAgIC8vIHJldHVybiB7XG4gICAgLy8gIHRleDpcbiAgICAvLyAgdGV4U2l6ZTpcbiAgICAvLyAgc2hhcGU6XG4gICAgLy8gIGNvbHM6XG4gICAgLy8gfVxuICAgIHRocm93IG5ldyBFcnJvcihcIm5vdCBpbXBsZW1lbnRlZDogZm9ybWF0LzEtNC9wYWNrL3RpbGUvaW5kZXguanM6cGFja1wiKVxufVxuXG5cbmV4cG9ydCBmdW5jdGlvbiB1bnBhY2soaW5mbywgYXJyKXtcbiAgICAvLyByZXR1cm4gbmRhcnJheVxuICAgIHRocm93IG5ldyBFcnJvcihcIm5vdCBpbXBsZW1lbnRlZDogZm9ybWF0LzEtNC9wYWNrL3RpbGUvaW5kZXguanM6dW5wYWNrXCIpXG59IiwiaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xuXG5leHBvcnQgZGVmYXVsdCB7XG5cdGhhcmRfc2lnbW9pZDogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvaGFyZF9zaWdtb2lkLmdsc2wnLCAndXRmOCcpLFxuXHRsaW5lYXI6IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL2xpbmVhci5nbHNsJywgJ3V0ZjgnKSxcblx0cmVsdTogcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvcmVsdS5nbHNsJywgJ3V0ZjgnKSxcblx0cmdiOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9yZ2IuZ2xzbCcsICd1dGY4JyksXG5cdHNpZ21vaWQ6IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3NpZ21vaWQuZ2xzbCcsICd1dGY4JyksXG5cdHRhbmg6IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3RhbmguZ2xzbCcsICd1dGY4JyksXG59IiwiaW1wb3J0IHsgcmVhZEZpbGVTeW5jIH0gZnJvbSAnZnMnO1xuXG5leHBvcnQgY29uc3QgZW5jb2RlU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvZW5jb2RlLmdsc2wnLCAndXRmOCcpO1xuZXhwb3J0IGNvbnN0IGRlY29kZVNoYWRlciA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL2RlY29kZS5nbHNsJywgJ3V0ZjgnKTtcblxuZXhwb3J0IGZ1bmN0aW9uIGluaXQoc2hhcGUsIGZvcm1hdCl7XG5cdHJldHVybiB7XG5cdFx0cmFuZ2U6IFtcblx0XHRcdGlzRmluaXRlKGZvcm1hdC5taW4pID8gZm9ybWF0Lm1pbiA6IDAsXG5cdFx0XHRpc0Zpbml0ZShmb3JtYXQubWF4KSA/IGZvcm1hdC5tYXggOiAxXG5cdFx0XVxuXHRcdC8vIG1heDogLFxuXHRcdC8vIG1pbjogLFxuXHR9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBlbmNvZGUoZGF0YSwgciwgZywgYiwgYSwgaW5mbyl7XG5cblx0ZGF0YVswXSA9IE1hdGgucm91bmQoMjU1ICogTWF0aC5taW4oMSwgTWF0aC5tYXgoMCwgKHIgLSBpbmZvLnJhbmdlWzBdKS8oaW5mby5yYW5nZVsxXSAtIGluZm8ucmFuZ2VbMF0pICkpKVxuXHRkYXRhWzFdID0gTWF0aC5yb3VuZCgyNTUgKiBNYXRoLm1pbigxLCBNYXRoLm1heCgwLCAoZyAtIGluZm8ucmFuZ2VbMF0pLyhpbmZvLnJhbmdlWzFdIC0gaW5mby5yYW5nZVswXSkgKSkpXG5cdGRhdGFbMl0gPSBNYXRoLnJvdW5kKDI1NSAqIE1hdGgubWluKDEsIE1hdGgubWF4KDAsIChiIC0gaW5mby5yYW5nZVswXSkvKGluZm8ucmFuZ2VbMV0gLSBpbmZvLnJhbmdlWzBdKSApKSlcblx0ZGF0YVszXSA9IE1hdGgucm91bmQoMjU1ICogTWF0aC5taW4oMSwgTWF0aC5tYXgoMCwgKGEgLSBpbmZvLnJhbmdlWzBdKS8oaW5mby5yYW5nZVsxXSAtIGluZm8ucmFuZ2VbMF0pICkpKVxuXHQvLyBjb25zb2xlLmxvZyhkYXRhWzBdLCBkYXRhWzFdLCBkYXRhWzJdKVxufVxuXG5cbmV4cG9ydCBmdW5jdGlvbiBkZWNvZGUoZGF0YSwgciwgZywgYiwgYSwgaW5mbyl7XG5cdGRhdGFbMF0gPSAociAvIDI1NSkgKiAoaW5mby5yYW5nZVsxXSAtIGluZm8ucmFuZ2VbMF0pICsgaW5mby5yYW5nZVswXTtcblx0ZGF0YVsxXSA9IChnIC8gMjU1KSAqIChpbmZvLnJhbmdlWzFdIC0gaW5mby5yYW5nZVswXSkgKyBpbmZvLnJhbmdlWzBdO1xuXHRkYXRhWzJdID0gKGIgLyAyNTUpICogKGluZm8ucmFuZ2VbMV0gLSBpbmZvLnJhbmdlWzBdKSArIGluZm8ucmFuZ2VbMF07XG5cdGRhdGFbM10gPSAoYSAvIDI1NSkgKiAoaW5mby5yYW5nZVsxXSAtIGluZm8ucmFuZ2VbMF0pICsgaW5mby5yYW5nZVswXTtcbn0iLCJpbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5cbmV4cG9ydCBjb25zdCBlbmNvZGVTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9lbmNvZGUuZ2xzbCcsICd1dGY4Jyk7XG5leHBvcnQgY29uc3QgZGVjb2RlU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvZGVjb2RlLmdsc2wnLCAndXRmOCcpO1xuXG5leHBvcnQgZnVuY3Rpb24gaW5pdChzaGFwZSwgZm9ybWF0KXtcblx0cmV0dXJuIHsgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZW5jb2RlKGRhdGEsIHIsIGcsIGIsIGEpe1xuXHRkYXRhWzBdID0gcjtcblx0ZGF0YVsxXSA9IGc7XG5cdGRhdGFbMl0gPSBiO1xuXHRkYXRhWzNdID0gYTtcbn1cblxuXG5leHBvcnQgZnVuY3Rpb24gZGVjb2RlKGRhdGEsIHIsIGcsIGIsIGEpe1xuXHRkYXRhWzBdID0gcjtcblx0ZGF0YVsxXSA9IGc7XG5cdGRhdGFbMl0gPSBiO1xuXHRkYXRhWzNdID0gYTtcbn0iLCJpbXBvcnQgKiBhcyBwYWNrX3N0cmlkZSBmcm9tICcuL3BhY2svc3RyaWRlL2luZGV4LmpzJ1xuaW1wb3J0ICogYXMgcGFja190aWxlIGZyb20gJy4vcGFjay90aWxlL2luZGV4LmpzJ1xuXG5pbXBvcnQgKiBhcyBjb2RlY19yYXcgZnJvbSAnLi9jb2RlYy9yYXcvaW5kZXguanMnXG5pbXBvcnQgKiBhcyBjb2RlY19saW5xdWFudCBmcm9tICcuL2NvZGVjL2xpbnF1YW50L2luZGV4LmpzJ1xuXG5pbXBvcnQgYWN0aXZhdGlvbnMgZnJvbSAnLi9hY3RpdmF0aW9uL2luZGV4LmpzJ1xuXG5pbXBvcnQgeyByZWFkRmlsZVN5bmMgfSBmcm9tICdmcyc7XG5cbmV4cG9ydCBkZWZhdWx0IHtcblx0cGFjazoge1xuXHRcdHN0cmlkZTogcGFja19zdHJpZGUsXG5cdFx0dGlsZTogcGFja190aWxlXG5cdH0sXG5cblxuXHRyZWFkX3NoaW06IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnL3BhY2svcmVhZF9zaGltLmdsc2wnLCAndXRmOCcpLFxuXHR3cml0ZV9zaGltOiByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9wYWNrL3dyaXRlX3NoaW0uZ2xzbCcsICd1dGY4JyksXG5cblx0Y29kZWM6IHtcblx0XHRyYXc6IGNvZGVjX3Jhdyxcblx0XHRsaW5xdWFudDogY29kZWNfbGlucXVhbnQsXG5cdH0sXG5cdGFjdGl2YXRpb25zOiBhY3RpdmF0aW9uc1xufSIsImltcG9ydCB7IHJlYWRGaWxlU3luYyB9IGZyb20gJ2ZzJztcblxuZXhwb3J0IGNvbnN0IHJlYWRTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9yZWFkLmdsc2wnLCAndXRmOCcpO1xuZXhwb3J0IGNvbnN0IHdyaXRlU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvd3JpdGUuZ2xzbCcsICd1dGY4Jyk7XG5pbXBvcnQgbmRhcnJheSBmcm9tICduZGFycmF5J1xuXG5leHBvcnQgZnVuY3Rpb24gaW5pdChzaGFwZSl7XG4gICAgdmFyIGxlbmd0aCA9IE1hdGguY2VpbChzaGFwZVsyXSAvIDQpICogc2hhcGVbM10gKiBzaGFwZVsxXSAqIHNoYXBlWzBdO1xuICAgIHZhciBjb2xzID0gTWF0aC5jZWlsKE1hdGguc3FydChsZW5ndGgpKTtcbiAgICB2YXIgdGV4U2l6ZSA9IFtjb2xzLCBNYXRoLmNlaWwobGVuZ3RoIC8gY29scyldXG5cbiAgICBjb25zb2xlLmFzc2VydCh0ZXhTaXplWzBdICogdGV4U2l6ZVsxXSA+PSBsZW5ndGgpXG4gICAgcmV0dXJuIHtcbiAgICAgICAgdGV4U2l6ZTogdGV4U2l6ZSxcbiAgICAgICAgc2hhcGU6IHNoYXBlLFxuXG4gICAgICAgIHN0cmlkZTogW1xuICAgICAgICAgICAgMSwgXG4gICAgICAgICAgICBzaGFwZVswXSwgXG4gICAgICAgICAgICBzaGFwZVswXSAqIHNoYXBlWzFdIC8gNCwgIC8vIHRoZSAvNCBpcyBiZWNhdXNlIG9mIHRoZSBjb2xvciBjaGFubmVsXG4gICAgICAgICAgICBzaGFwZVswXSAqIHNoYXBlWzFdICogTWF0aC5jZWlsKHNoYXBlWzJdIC8gNClcbiAgICAgICAgXSxcbiAgICAgICAgLy8gZGVjdmVjOiBbMSwgc2hhcGVbMF0sIHNoYXBlWzBdICogc2hhcGVbMV0sIHNoYXBlWzBdICogc2hhcGVbMV0gKiBNYXRoLmNlaWwoc2hhcGVbMl0gLyA0KV1cbiAgICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBwYWNrKGluZm8sIGFycmF5LCBlbmNvZGU0LCBmb3JtYXQpe1xuICAgIC8vIHJldHVybiBVaW50OEFycmF5IG9yIEZsb2F0MzJBcnJheVxuXG4gICAgYXJyYXkgPSBuZGFycmF5KGFycmF5LmRhdGEsIFxuICAgICAgICBhcnJheS5zaGFwZS5jb25jYXQoWzEsIDEsIDEsIDFdKS5zbGljZSgwLCA0KSxcbiAgICAgICAgYXJyYXkuc3RyaWRlLmNvbmNhdChbMSwgMSwgMSwgMV0pLnNsaWNlKDAsIDQpLFxuICAgICAgICBhcnJheS5vZmZzZXQpXG4gICAgXG4gICAgdmFyIFt3aWR0aCwgaGVpZ2h0XSA9IGluZm8udGV4U2l6ZSxcbiAgICAgICAgbGVuZ3RoID0gd2lkdGggKiBoZWlnaHQgKiA0O1xuICAgIHZhciBzaGFwZSA9IGluZm8uc2hhcGU7XG5cbiAgICBpZihmb3JtYXQudHlwZSA9PT0gJ2Zsb2F0MzInKXtcbiAgICAgICAgdmFyIGRhdGEgPSBuZXcgRmxvYXQzMkFycmF5KGxlbmd0aCk7ICAgIFxuICAgIH1lbHNlIGlmKGZvcm1hdC50eXBlID09PSAndWludDgnKXtcbiAgICAgICAgdmFyIGRhdGEgPSBuZXcgVWludDhBcnJheShsZW5ndGgpOyAgICBcbiAgICB9XG5cbiAgICB2YXIgY2hhbnMgPSBNYXRoLmNlaWwoaW5mby5zaGFwZVsyXSAvIDQpO1xuXG4gICAgZm9yKHZhciBpID0gMDsgaSA8IGluZm8uc2hhcGVbMF07IGkrKyl7XG4gICAgICAgIGZvcih2YXIgaiA9IDA7IGogPCBpbmZvLnNoYXBlWzFdOyBqKyspe1xuICAgICAgICAgICAgZm9yKHZhciBrID0gMDsgayA8IGNoYW5zOyBrKyspe1xuICAgICAgICAgICAgICAgIHZhciBiID0gTWF0aC5taW4oayo0KzQsIHNoYXBlWzJdKS1rKjQ7XG4gICAgICAgICAgICAgICAgZm9yKHZhciB3ID0gMDsgdyA8IGluZm8uc2hhcGVbM107IHcrKyl7XG5cbiAgICAgICAgICAgICAgICAgICAgdmFyIHRpbGUgID0gaSArIFxuICAgICAgICAgICAgICAgICAgICAgICAgaiAqIHNoYXBlWzBdICsgXG4gICAgICAgICAgICAgICAgICAgICAgICBrICogc2hhcGVbMF0gKiBzaGFwZVsxXSArXG4gICAgICAgICAgICAgICAgICAgICAgICB3ICogc2hhcGVbMF0gKiBzaGFwZVsxXSAqIGNoYW5zO1xuXG5cbiAgICAgICAgICAgICAgICAgICAgdmFyIHBvcyA9IDQgKiB0aWxlO1xuICAgICAgICAgICAgICAgICAgICBlbmNvZGU0KFxuICAgICAgICAgICAgICAgICAgICAgICAgZGF0YS5zdWJhcnJheShwb3MsIHBvcyArIDQpLFxuICAgICAgICAgICAgICAgICAgICAgICAgYiA8IDEgPyAwIDogYXJyYXkuZ2V0KGksIGosIDQqayswLCB3KSwgXG4gICAgICAgICAgICAgICAgICAgICAgICBiIDwgMiA/IDAgOiBhcnJheS5nZXQoaSwgaiwgNCprKzEsIHcpLCBcbiAgICAgICAgICAgICAgICAgICAgICAgIGIgPCAzID8gMCA6IGFycmF5LmdldChpLCBqLCA0KmsrMiwgdyksIFxuICAgICAgICAgICAgICAgICAgICAgICAgYiA8IDQgPyAwIDogYXJyYXkuZ2V0KGksIGosIDQqayszLCB3KSwgaW5mbylcbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gZGF0YVxufVxuXG5cbmV4cG9ydCBmdW5jdGlvbiB1bnBhY2soaW5mbywgZGF0YSwgZGVjb2RlNCwgdHlwZSl7XG5cblxuXG4gICAgdmFyIHNoYXBlID0gaW5mby5zaGFwZTtcbiAgICB2YXIgc2hhcGVsZW5ndGggPSBzaGFwZS5yZWR1Y2UoKGEsIGIpID0+IGEgKiBiKTtcbiAgICBcbiAgICB2YXIgW3dpZHRoLCBoZWlnaHRdID0gaW5mby50ZXhTaXplLFxuICAgICAgICBsZW5ndGggPSB3aWR0aCAqIGhlaWdodCAqIDQ7XG4gICAgdmFyIGNoYW5zID0gTWF0aC5jZWlsKGluZm8uc2hhcGVbMl0gLyA0KTtcblxuICAgIC8vIGlmKHR5cGUgPT09ICdmbG9hdDMyJyl7XG4gICAgdmFyIGFycmF5ID0gbmRhcnJheShuZXcgRmxvYXQzMkFycmF5KHNoYXBlbGVuZ3RoKSwgc2hhcGUpXG4gICAgdmFyIGJ1ZiA9IG5ldyBGbG9hdDMyQXJyYXkoNCk7XG4gICAgLy8gfWVsc2UgaWYodHlwZSA9PSAndWludDgnKXtcbiAgICAvLyAgICAgdmFyIGFycmF5ID0gbmRhcnJheShuZXcgVWludDhBcnJheShzaGFwZWxlbmd0aCksIHNoYXBlKVxuICAgIC8vICAgICB2YXIgYnVmID0gbmV3IFVpbnQ4QXJyYXkoNCk7XG4gICAgLy8gfWVsc2UgdGhyb3cgbmV3IEVycm9yKCd1bmltcGxlbWVudGVkIHR5cGUnKTtcbiAgICBcblxuICAgIGZvcih2YXIgaSA9IDA7IGkgPCBpbmZvLnNoYXBlWzBdOyBpKyspe1xuICAgICAgICBmb3IodmFyIGogPSAwOyBqIDwgaW5mby5zaGFwZVsxXTsgaisrKXtcbiAgICAgICAgICAgIGZvcih2YXIgayA9IDA7IGsgPCBjaGFuczsgaysrKXtcbiAgICAgICAgICAgICAgICB2YXIgYiA9IE1hdGgubWluKGsqNCs0LCBzaGFwZVsyXSktayo0O1xuICAgICAgICAgICAgICAgIGZvcih2YXIgdyA9IDA7IHcgPCBpbmZvLnNoYXBlWzNdOyB3Kyspe1xuXG4gICAgICAgICAgICAgICAgICAgIHZhciB0aWxlICA9IFxuICAgICAgICAgICAgICAgICAgICAgICAgaSArIFxuICAgICAgICAgICAgICAgICAgICAgICAgaiAqIHNoYXBlWzBdICsgXG4gICAgICAgICAgICAgICAgICAgICAgICBrICogc2hhcGVbMF0gKiBzaGFwZVsxXSArXG4gICAgICAgICAgICAgICAgICAgICAgICB3ICogc2hhcGVbMF0gKiBzaGFwZVsxXSAqIGNoYW5zO1xuXG4gICAgICAgICAgICAgICAgICAgIGRlY29kZTQoYnVmLCBcbiAgICAgICAgICAgICAgICAgICAgICAgIGRhdGFbNCAqIHRpbGUgKyAwXSxcbiAgICAgICAgICAgICAgICAgICAgICAgIGRhdGFbNCAqIHRpbGUgKyAxXSxcbiAgICAgICAgICAgICAgICAgICAgICAgIGRhdGFbNCAqIHRpbGUgKyAyXSxcbiAgICAgICAgICAgICAgICAgICAgICAgIGRhdGFbNCAqIHRpbGUgKyAzXSwgaW5mbylcblxuXG4gICAgICAgICAgICAgICAgICAgIGZvcih2YXIgeCA9IDA7IHggPCBiOyB4Kyspe1xuICAgICAgICAgICAgICAgICAgICAgICAgYXJyYXkuc2V0KGksIGosIDQqayt4LCB3LCBidWZbeF0pXG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gYXJyYXk7XG5cbn1cbiIsImltcG9ydCB7IHJlYWRGaWxlU3luYyB9IGZyb20gJ2ZzJztcblxuZXhwb3J0IGNvbnN0IHJlYWRTaGFkZXIgPSByZWFkRmlsZVN5bmMoX19kaXJuYW1lICsgJy9yZWFkLmdsc2wnLCAndXRmOCcpO1xuZXhwb3J0IGNvbnN0IHdyaXRlU2hhZGVyID0gcmVhZEZpbGVTeW5jKF9fZGlybmFtZSArICcvd3JpdGUuZ2xzbCcsICd1dGY4Jyk7XG5cbmV4cG9ydCBmdW5jdGlvbiBpbml0KHNoYXBlKXtcbiAgICB2YXIgd2lkdGggPSBzaGFwZVswXTsgLy8gdmFyIHdpZHRoID0gc2hhcGVbMF0gKiA0OyAgICBcbiAgICAvLyB3ZSBwaWNrIHRoZSBudW1iZXIgb2YgY29sdW1ucyBzbyB3ZSBjYW4ga2VlcFxuICAgIC8vIHRoZSB0ZXh0dXJlIGFzIHNxdWFyZSBhcyBwb3NzaWJsZSwgd2l0aCB0aGVcbiAgICAvLyBtaW5pbWFsIGFtb3VudCBvZiB3YXN0ZWQgc3BhY2UuXG5cbiAgICB2YXIgdGlsZXMgPSBNYXRoLmNlaWwoc2hhcGVbMl0gLyA0KSAqIHNoYXBlWzNdLFxuICAgICAgICBjb2xzID0gTWF0aC5tYXgoMSwgTWF0aC5taW4odGlsZXMsIE1hdGgucm91bmQoXG4gICAgICAgICAgICBNYXRoLnNxcnQoc2hhcGVbMF0gKiBzaGFwZVsxXSAqIHRpbGVzKSAvIHdpZHRoKSkpO1xuXG4gICAgdmFyIHRleFNpemUgPSBbd2lkdGggKiBjb2xzLCBzaGFwZVsxXSAqIE1hdGguY2VpbCh0aWxlcyAvIGNvbHMpXVxuXG4gICAgcmV0dXJuIHtcbiAgICBcdHRleFNpemU6IHRleFNpemUsXG4gICAgXHRjb2xzOiBjb2xzLFxuICAgIFx0c2hhcGU6IHNoYXBlLFxuICAgIH1cbn1cblxuaW1wb3J0IG5kYXJyYXkgZnJvbSBcIm5kYXJyYXlcIlxuXG5leHBvcnQgZnVuY3Rpb24gcGFjayhpbmZvLCBhcnJheSwgZW5jb2RlNCwgZm9ybWF0KXtcbiAgICBhcnJheSA9IG5kYXJyYXkoYXJyYXkuZGF0YSwgXG4gICAgICAgIGFycmF5LnNoYXBlLmNvbmNhdChbMSwgMSwgMSwgMV0pLnNsaWNlKDAsIDQpLFxuICAgICAgICBhcnJheS5zdHJpZGUuY29uY2F0KFsxLCAxLCAxLCAxXSkuc2xpY2UoMCwgNCksXG4gICAgICAgIGFycmF5Lm9mZnNldClcblxuICAgIHZhciBzaGFwZSA9IGFycmF5LnNoYXBlLFxuICAgICAgICB0aWxlcyA9IE1hdGguY2VpbChzaGFwZVsyXSAvIDQpICogc2hhcGVbM10sXG4gICAgICAgIHR3ID0gc2hhcGVbMF0sXG4gICAgICAgIHRoID0gc2hhcGVbMV0sXG4gICAgICAgIGNvbHMgPSBpbmZvLmNvbHMsXG4gICAgICAgIFt3aWR0aCwgaGVpZ2h0XSA9IGluZm8udGV4U2l6ZSxcbiAgICAgICAgY2h1bmtzID0gTWF0aC5jZWlsKHNoYXBlWzJdIC8gNCksXG4gICAgICAgIGxlbmd0aCA9IHdpZHRoICogaGVpZ2h0ICogNDtcblxuICAgIGlmKGZvcm1hdC50eXBlID09PSAnZmxvYXQzMicpe1xuICAgICAgICB2YXIgZGF0YSA9IG5ldyBGbG9hdDMyQXJyYXkobGVuZ3RoKTsgICAgXG4gICAgfWVsc2UgaWYoZm9ybWF0LnR5cGUgPT09ICd1aW50OCcpe1xuICAgICAgICB2YXIgZGF0YSA9IG5ldyBVaW50OEFycmF5KGxlbmd0aCk7ICAgIFxuICAgIH1cbiAgICBcblxuICAgIGZvcih2YXIgeiA9IDA7IHogPCBjaHVua3M7IHorKyl7XG4gICAgICAgIGZvcih2YXIgdyA9IDA7IHcgPCBzaGFwZVszXTsgdysrKXtcbiAgICAgICAgICAgIHZhciB0aWxlID0gdyAqIGNodW5rcyArIHo7XG4gICAgICAgICAgICB2YXIgYiA9IE1hdGgubWluKHoqNCs0LCBzaGFwZVsyXSkteio0O1xuICAgICAgICAgICAgXG4gICAgICAgICAgICB2YXIgaWggPSB0aCAqIE1hdGguZmxvb3IodGlsZSAvIGNvbHMpO1xuICAgICAgICAgICAgdmFyIGp3ID0gdHcgKiAodGlsZSAlIGNvbHMpO1xuXG4gICAgICAgICAgICBmb3IodmFyIGkgPSAwOyBpIDwgdHc7IGkrKyl7XG4gICAgICAgICAgICAgICAgZm9yKHZhciBqID0gMDsgaiA8IHRoOyBqKyspe1xuXG4gICAgICAgICAgICAgICAgICAgIHZhciBwb3MgPSA0ICogKChpaCtqKSAqIHdpZHRoICsgancgKyBpKTtcbiAgICAgICAgICAgICAgICAgICAgZW5jb2RlNChcbiAgICAgICAgICAgICAgICAgICAgICAgIGRhdGEuc3ViYXJyYXkocG9zLCBwb3MgKyA0KSxcbiAgICAgICAgICAgICAgICAgICAgICAgIGIgPCAxID8gMCA6IGFycmF5LmdldChpLCBqLCA0KnorMCwgdyksIFxuICAgICAgICAgICAgICAgICAgICAgICAgYiA8IDIgPyAwIDogYXJyYXkuZ2V0KGksIGosIDQqeisxLCB3KSwgXG4gICAgICAgICAgICAgICAgICAgICAgICBiIDwgMyA/IDAgOiBhcnJheS5nZXQoaSwgaiwgNCp6KzIsIHcpLCBcbiAgICAgICAgICAgICAgICAgICAgICAgIGIgPCA0ID8gMCA6IGFycmF5LmdldChpLCBqLCA0KnorMywgdyksIGluZm8pXG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICB9XG4gICAgfVxuICAgIHJldHVybiBkYXRhO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdW5wYWNrKGluZm8sIGRhdGEsIGRlY29kZTQsIHR5cGUpe1xuICAgIHRocm93IG5ldyBFcnJvcihcIm5vdCBpbXBsZW1lbnRlZDogZm9ybWF0LzQtNC9wYWNrL3RpbGUvaW5kZXguanM6dW5wYWNrXCIpXG59IiwiaW1wb3J0IEZvcm1hdDQ0IGZyb20gJy4vNC00L2luZGV4LmpzJ1xuaW1wb3J0IEZvcm1hdDE0IGZyb20gJy4vMS00L2luZGV4LmpzJ1xuXG5leHBvcnQgZGVmYXVsdCB7XG5cdCc0OjQnOiBGb3JtYXQ0NCxcblx0JzE6NCc6IEZvcm1hdDE0LFxufSIsIi8vIGRvIHlvdSBldmVyIGhvcGUgdGhhdCBwZXJoYXBzIGluZGV4IGZpbGVzIHNob3VsZCBcbi8vIGFjdHVhbGx5IGJlIGluZGV4IGZpbGVzIGxhY2tpbmcgYW55IGltcGxlbWVudGF0aW9uIFxuLy8gY29kZT8gd2VsbCwgdG9kYXkgeW91J3JlIGluIGx1Y2shXG5cbmV4cG9ydCB7IFRlbnNvciwgT3V0cHV0VGVuc29yLCBJblBsYWNlVGVuc29yIH0gZnJvbSAnLi90ZW5zb3IvaW5kZXguanMnXG5leHBvcnQgeyBSdW4sIENvbXBpbGUgfSBmcm9tICcuL3J1bnRpbWUvaW5kZXguanMnXG5leHBvcnQgeyBjcmVhdGVHTCB9IGZyb20gJy4vdXRpbC5qcyciLCIvLyBjb2RlIGZvciBwcmV0dHkgcHJpbnRpbmcgc2hhZGVyIGVycm9ycyBmcm9tIHJlZ2xcblxuZXhwb3J0IGZ1bmN0aW9uIGNoZWNrTGlua0Vycm9yIChnbCwgcHJvZ3JhbSwgZnJhZ1NoYWRlciwgdmVydFNoYWRlciwgY29tbWFuZCkge1xuICAgIGlmICghZ2wuZ2V0UHJvZ3JhbVBhcmFtZXRlcihwcm9ncmFtLCBnbC5MSU5LX1NUQVRVUykpIHtcbiAgICAgICAgdmFyIGVyckxvZyA9IGdsLmdldFByb2dyYW1JbmZvTG9nKHByb2dyYW0pXG4gICAgICAgIHZhciBmcmFnUGFyc2UgPSBwYXJzZVNvdXJjZShmcmFnU2hhZGVyLCBjb21tYW5kKVxuICAgICAgICB2YXIgdmVydFBhcnNlID0gcGFyc2VTb3VyY2UodmVydFNoYWRlciwgY29tbWFuZClcblxuICAgICAgICB2YXIgaGVhZGVyID0gJ0Vycm9yIGxpbmtpbmcgcHJvZ3JhbSB3aXRoIHZlcnRleCBzaGFkZXIsIFwiJyArXG4gICAgICAgICAgICB2ZXJ0UGFyc2VbMF0ubmFtZSArICdcIiwgYW5kIGZyYWdtZW50IHNoYWRlciBcIicgKyBmcmFnUGFyc2VbMF0ubmFtZSArICdcIidcblxuICAgICAgICBpZiAodHlwZW9mIGRvY3VtZW50ICE9PSAndW5kZWZpbmVkJykge1xuICAgICAgICAgICAgY29uc29sZS5sb2coJyVjJyArIGhlYWRlciArICdcXG4lYycgKyBlcnJMb2csXG4gICAgICAgICAgICAgICAgJ2NvbG9yOnJlZDt0ZXh0LWRlY29yYXRpb246dW5kZXJsaW5lO2ZvbnQtd2VpZ2h0OmJvbGQnLFxuICAgICAgICAgICAgICAgICdjb2xvcjpyZWQnKVxuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgY29uc29sZS5sb2coaGVhZGVyICsgJ1xcbicgKyBlcnJMb2cpXG4gICAgICAgIH1cblxuICAgICAgICBjb25zb2xlLmxvZyhmcmFnU2hhZGVyKTtcbiAgICAgICAgXG4gICAgICAgIHRocm93IG5ldyBFcnJvcihoZWFkZXIpXG4gICAgfVxufVxuXG5cbmV4cG9ydCBmdW5jdGlvbiBjaGVja1NoYWRlckVycm9yIChnbCwgc2hhZGVyLCBzb3VyY2UsIHR5cGUsIGNvbW1hbmQpIHtcbiAgICBpZiAoIWdsLmdldFNoYWRlclBhcmFtZXRlcihzaGFkZXIsIGdsLkNPTVBJTEVfU1RBVFVTKSkge1xuICAgICAgICB2YXIgZXJyTG9nID0gZ2wuZ2V0U2hhZGVySW5mb0xvZyhzaGFkZXIpXG4gICAgICAgIHZhciB0eXBlTmFtZSA9IHR5cGUgPT09IGdsLkZSQUdNRU5UX1NIQURFUiA/ICdmcmFnbWVudCcgOiAndmVydGV4J1xuICAgICAgICAvLyBjaGVja0NvbW1hbmRUeXBlKHNvdXJjZSwgJ3N0cmluZycsIHR5cGVOYW1lICsgJyBzaGFkZXIgc291cmNlIG11c3QgYmUgYSBzdHJpbmcnLCBjb21tYW5kKVxuXG4gICAgICAgIHZhciBmaWxlcyA9IHBhcnNlU291cmNlKHNvdXJjZSwgY29tbWFuZClcbiAgICAgICAgdmFyIGVycm9ycyA9IHBhcnNlRXJyb3JMb2coZXJyTG9nKVxuICAgICAgICBhbm5vdGF0ZUZpbGVzKGZpbGVzLCBlcnJvcnMpXG5cbiAgICAgICAgT2JqZWN0LmtleXMoZmlsZXMpLmZvckVhY2goZnVuY3Rpb24gKGZpbGVOdW1iZXIpIHtcbiAgICAgICAgICAgIHZhciBmaWxlID0gZmlsZXNbZmlsZU51bWJlcl1cbiAgICAgICAgICAgIGlmICghZmlsZS5oYXNFcnJvcnMpIHtcbiAgICAgICAgICAgICAgICByZXR1cm5cbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgdmFyIHN0cmluZ3MgPSBbJyddXG4gICAgICAgICAgICB2YXIgc3R5bGVzID0gWycnXVxuXG4gICAgICAgICAgICBmdW5jdGlvbiBwdXNoIChzdHIsIHN0eWxlKSB7XG4gICAgICAgICAgICAgICAgc3RyaW5ncy5wdXNoKHN0cilcbiAgICAgICAgICAgICAgICBzdHlsZXMucHVzaChzdHlsZSB8fCAnJylcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgcHVzaCgnZmlsZSBudW1iZXIgJyArIGZpbGVOdW1iZXIgKyAnOiAnICsgZmlsZS5uYW1lICsgJ1xcbicsICdjb2xvcjpyZWQ7dGV4dC1kZWNvcmF0aW9uOnVuZGVybGluZTtmb250LXdlaWdodDpib2xkJylcblxuICAgICAgICAgICAgZmlsZS5saW5lcy5mb3JFYWNoKGZ1bmN0aW9uIChsaW5lKSB7XG4gICAgICAgICAgICAgICAgaWYgKGxpbmUuZXJyb3JzLmxlbmd0aCA+IDApIHtcbiAgICAgICAgICAgICAgICAgICAgcHVzaChsZWZ0UGFkKGxpbmUubnVtYmVyLCA0KSArICd8ICAnLCAnYmFja2dyb3VuZC1jb2xvcjp5ZWxsb3c7IGZvbnQtd2VpZ2h0OmJvbGQnKVxuICAgICAgICAgICAgICAgICAgICBwdXNoKGxpbmUubGluZSArICdcXG4nLCAnY29sb3I6cmVkOyBiYWNrZ3JvdW5kLWNvbG9yOnllbGxvdzsgZm9udC13ZWlnaHQ6Ym9sZCcpXG5cbiAgICAgICAgICAgICAgICAgICAgLy8gdHJ5IHRvIGd1ZXNzIHRva2VuXG4gICAgICAgICAgICAgICAgICAgIHZhciBvZmZzZXQgPSAwXG4gICAgICAgICAgICAgICAgICAgIGxpbmUuZXJyb3JzLmZvckVhY2goZnVuY3Rpb24gKGVycm9yKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICB2YXIgbWVzc2FnZSA9IGVycm9yLm1lc3NhZ2VcbiAgICAgICAgICAgICAgICAgICAgICAgIHZhciB0b2tlbiA9IC9eXFxzKlxcJyguKilcXCdcXHMqXFw6XFxzKiguKikkLy5leGVjKG1lc3NhZ2UpXG4gICAgICAgICAgICAgICAgICAgICAgICBpZiAodG9rZW4pIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB2YXIgdG9rZW5QYXQgPSB0b2tlblsxXVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1lc3NhZ2UgPSB0b2tlblsyXVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgIHN3aXRjaCAodG9rZW5QYXQpIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY2FzZSAnYXNzaWduJzpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRva2VuUGF0ID0gJz0nXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBicmVha1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBvZmZzZXQgPSBNYXRoLm1heChsaW5lLmxpbmUuaW5kZXhPZih0b2tlblBhdCwgb2Zmc2V0KSwgMClcbiAgICAgICAgICAgICAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgb2Zmc2V0ID0gMFxuICAgICAgICAgICAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgICAgICAgICAgICBwdXNoKGxlZnRQYWQoJ3wgJywgNikpXG4gICAgICAgICAgICAgICAgICAgICAgICBwdXNoKGxlZnRQYWQoJ15eXicsIG9mZnNldCArIDMpICsgJ1xcbicsICdmb250LXdlaWdodDpib2xkJylcbiAgICAgICAgICAgICAgICAgICAgICAgIHB1c2gobGVmdFBhZCgnfCAnLCA2KSlcbiAgICAgICAgICAgICAgICAgICAgICAgIHB1c2gobWVzc2FnZSArICdcXG4nLCAnZm9udC13ZWlnaHQ6Ym9sZCcpXG4gICAgICAgICAgICAgICAgICAgIH0pXG4gICAgICAgICAgICAgICAgICAgIHB1c2gobGVmdFBhZCgnfCAnLCA2KSArICdcXG4nKVxuICAgICAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgICAgICAgIHB1c2gobGVmdFBhZChsaW5lLm51bWJlciwgNCkgKyAnfCAgJylcbiAgICAgICAgICAgICAgICAgICAgcHVzaChsaW5lLmxpbmUgKyAnXFxuJywgJ2NvbG9yOnJlZCcpXG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfSlcbiAgICAgICAgICAgIGlmICh0eXBlb2YgZG9jdW1lbnQgIT09ICd1bmRlZmluZWQnKSB7XG4gICAgICAgICAgICAgICAgc3R5bGVzWzBdID0gc3RyaW5ncy5qb2luKCclYycpXG4gICAgICAgICAgICAgICAgY29uc29sZS5sb2cuYXBwbHkoY29uc29sZSwgc3R5bGVzKVxuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBjb25zb2xlLmxvZyhzdHJpbmdzLmpvaW4oJycpKVxuICAgICAgICAgICAgfVxuICAgICAgICB9KVxuXG4gICAgICAgIHRocm93IG5ldyBFcnJvcignRXJyb3IgY29tcGlsaW5nICcgKyB0eXBlTmFtZSArICcgc2hhZGVyLCAnICsgZmlsZXNbMF0ubmFtZSlcbiAgICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjaGVja0ZyYW1lYnVmZmVyRXJyb3IoZ2wpe1xuICAgIFxuICAgIHZhciBzdGF0dXMgPSBnbC5jaGVja0ZyYW1lYnVmZmVyU3RhdHVzKGdsLkZSQU1FQlVGRkVSKTtcbiAgICBpZihzdGF0dXMgIT0gZ2wuRlJBTUVCVUZGRVJfQ09NUExFVEUpe1xuICAgICAgICB2YXIgc3RhdHVzQ29kZSA9IHt9XG4gICAgICAgIHN0YXR1c0NvZGVbZ2wuRlJBTUVCVUZGRVJfQ09NUExFVEVdID0gJ2NvbXBsZXRlJ1xuICAgICAgICBzdGF0dXNDb2RlW2dsLkZSQU1FQlVGRkVSX0lOQ09NUExFVEVfQVRUQUNITUVOVF0gPSAnaW5jb21wbGV0ZSBhdHRhY2htZW50J1xuICAgICAgICBzdGF0dXNDb2RlW2dsLkZSQU1FQlVGRkVSX0lOQ09NUExFVEVfRElNRU5TSU9OU10gPSAnaW5jb21wbGV0ZSBkaW1lbnNpb25zJ1xuICAgICAgICBzdGF0dXNDb2RlW2dsLkZSQU1FQlVGRkVSX0lOQ09NUExFVEVfTUlTU0lOR19BVFRBQ0hNRU5UXSA9ICdpbmNvbXBsZXRlLCBtaXNzaW5nIGF0dGFjaG1lbnQnXG4gICAgICAgIHN0YXR1c0NvZGVbZ2wuRlJBTUVCVUZGRVJfVU5TVVBQT1JURURdID0gJ3Vuc3VwcG9ydGVkJ1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ2ZyYW1lYnVmZmVyIGNvbmZpZ3VyYXRpb24gbm90IHN1cHBvcnRlZCwgc3RhdHVzID0gJyArIHN0YXR1c0NvZGVbc3RhdHVzXSlcbiAgICB9XG59XG5cblxuZnVuY3Rpb24gbGVmdFBhZCAoc3RyLCBuKSB7XG4gICAgc3RyID0gc3RyICsgJydcbiAgICB3aGlsZSAoc3RyLmxlbmd0aCA8IG4pIHtcbiAgICAgICAgc3RyID0gJyAnICsgc3RyXG4gICAgfVxuICAgIHJldHVybiBzdHJcbn1cblxuZnVuY3Rpb24gU2hhZGVyRmlsZSAoKSB7XG4gICAgdGhpcy5uYW1lID0gJ3Vua25vd24nXG4gICAgdGhpcy5saW5lcyA9IFtdXG4gICAgdGhpcy5pbmRleCA9IHt9XG4gICAgdGhpcy5oYXNFcnJvcnMgPSBmYWxzZVxufVxuXG5mdW5jdGlvbiBTaGFkZXJMaW5lIChudW1iZXIsIGxpbmUpIHtcbiAgICB0aGlzLm51bWJlciA9IG51bWJlclxuICAgIHRoaXMubGluZSA9IGxpbmVcbiAgICB0aGlzLmVycm9ycyA9IFtdXG59XG5cbmZ1bmN0aW9uIFNoYWRlckVycm9yIChmaWxlTnVtYmVyLCBsaW5lTnVtYmVyLCBtZXNzYWdlKSB7XG4gICAgdGhpcy5maWxlID0gZmlsZU51bWJlclxuICAgIHRoaXMubGluZSA9IGxpbmVOdW1iZXJcbiAgICB0aGlzLm1lc3NhZ2UgPSBtZXNzYWdlXG59XG5cbmZ1bmN0aW9uIHBhcnNlU291cmNlIChzb3VyY2UsIGNvbW1hbmQpIHtcbiAgICB2YXIgbGluZXMgPSBzb3VyY2Uuc3BsaXQoJ1xcbicpXG4gICAgdmFyIGxpbmVOdW1iZXIgPSAxXG4gICAgdmFyIGZpbGVOdW1iZXIgPSAwXG4gICAgdmFyIGZpbGVzID0ge1xuICAgICAgICB1bmtub3duOiBuZXcgU2hhZGVyRmlsZSgpLFxuICAgICAgICAwOiBuZXcgU2hhZGVyRmlsZSgpXG4gICAgfVxuICAgIGZpbGVzLnVua25vd24ubmFtZSA9IGZpbGVzWzBdLm5hbWUgPSAndW5rbm93bidcbiAgICBmaWxlcy51bmtub3duLmxpbmVzLnB1c2gobmV3IFNoYWRlckxpbmUoMCwgJycpKVxuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgbGluZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgICAgdmFyIGxpbmUgPSBsaW5lc1tpXVxuICAgICAgICB2YXIgcGFydHMgPSAvXlxccypcXCNcXHMqKFxcdyspXFxzKyguKylcXHMqJC8uZXhlYyhsaW5lKVxuICAgICAgICBpZiAocGFydHMpIHtcbiAgICAgICAgICAgIHN3aXRjaCAocGFydHNbMV0pIHtcbiAgICAgICAgICAgICAgICBjYXNlICdsaW5lJzpcbiAgICAgICAgICAgICAgICAgICAgdmFyIGxpbmVOdW1iZXJJbmZvID0gLyhcXGQrKShcXHMrXFxkKyk/Ly5leGVjKHBhcnRzWzJdKVxuICAgICAgICAgICAgICAgICAgICBpZiAobGluZU51bWJlckluZm8pIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGxpbmVOdW1iZXIgPSBsaW5lTnVtYmVySW5mb1sxXSB8IDBcbiAgICAgICAgICAgICAgICAgICAgICAgIGlmIChsaW5lTnVtYmVySW5mb1syXSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZpbGVOdW1iZXIgPSBsaW5lTnVtYmVySW5mb1syXSB8IDBcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBpZiAoIShmaWxlTnVtYmVyIGluIGZpbGVzKSkge1xuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmaWxlc1tmaWxlTnVtYmVyXSA9IG5ldyBTaGFkZXJGaWxlKClcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgYnJlYWtcbiAgICAgICAgICAgICAgICBjYXNlICdkZWZpbmUnOlxuICAgICAgICAgICAgICAgICAgICB2YXIgbmFtZUluZm8gPSAvU0hBREVSX05BTUUoX0I2NCk/XFxzKyguKikkLy5leGVjKHBhcnRzWzJdKVxuICAgICAgICAgICAgICAgICAgICBpZiAobmFtZUluZm8pIHtcbiAgICAgICAgICAgICAgICAgICAgICAgIGZpbGVzW2ZpbGVOdW1iZXJdLm5hbWUgPSAobmFtZUluZm9bMV1cbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgPyBkZWNvZGVCNjQobmFtZUluZm9bMl0pXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIDogbmFtZUluZm9bMl0pXG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgYnJlYWtcbiAgICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBmaWxlc1tmaWxlTnVtYmVyXS5saW5lcy5wdXNoKG5ldyBTaGFkZXJMaW5lKGxpbmVOdW1iZXIrKywgbGluZSkpXG4gICAgfVxuICAgIE9iamVjdC5rZXlzKGZpbGVzKS5mb3JFYWNoKGZ1bmN0aW9uIChmaWxlTnVtYmVyKSB7XG4gICAgICAgIHZhciBmaWxlID0gZmlsZXNbZmlsZU51bWJlcl1cbiAgICAgICAgZmlsZS5saW5lcy5mb3JFYWNoKGZ1bmN0aW9uIChsaW5lKSB7XG4gICAgICAgICAgICBmaWxlLmluZGV4W2xpbmUubnVtYmVyXSA9IGxpbmVcbiAgICAgICAgfSlcbiAgICB9KVxuICAgIHJldHVybiBmaWxlc1xufVxuXG5mdW5jdGlvbiBwYXJzZUVycm9yTG9nIChlcnJMb2cpIHtcbiAgICB2YXIgcmVzdWx0ID0gW11cbiAgICBlcnJMb2cuc3BsaXQoJ1xcbicpLmZvckVhY2goZnVuY3Rpb24gKGVyck1zZykge1xuICAgICAgICBpZiAoZXJyTXNnLmxlbmd0aCA8IDUpIHtcbiAgICAgICAgICAgIHJldHVyblxuICAgICAgICB9XG4gICAgICAgIHZhciBwYXJ0cyA9IC9eRVJST1JcXDpcXHMrKFxcZCspXFw6KFxcZCspXFw6XFxzKiguKikkLy5leGVjKGVyck1zZylcbiAgICAgICAgaWYgKHBhcnRzKSB7XG4gICAgICAgICAgICByZXN1bHQucHVzaChuZXcgU2hhZGVyRXJyb3IoXG4gICAgICAgICAgICAgICAgcGFydHNbMV0gfCAwLFxuICAgICAgICAgICAgICAgIHBhcnRzWzJdIHwgMCxcbiAgICAgICAgICAgICAgICBwYXJ0c1szXS50cmltKCkpKVxuICAgICAgICB9IGVsc2UgaWYgKGVyck1zZy5sZW5ndGggPiAwKSB7XG4gICAgICAgICAgICByZXN1bHQucHVzaChuZXcgU2hhZGVyRXJyb3IoJ3Vua25vd24nLCAwLCBlcnJNc2cpKVxuICAgICAgICB9XG4gICAgfSlcbiAgICByZXR1cm4gcmVzdWx0XG59XG5cbmZ1bmN0aW9uIGFubm90YXRlRmlsZXMgKGZpbGVzLCBlcnJvcnMpIHtcbiAgICBlcnJvcnMuZm9yRWFjaChmdW5jdGlvbiAoZXJyb3IpIHtcbiAgICAgICAgdmFyIGZpbGUgPSBmaWxlc1tlcnJvci5maWxlXVxuICAgICAgICBpZiAoZmlsZSkge1xuICAgICAgICAgICAgdmFyIGxpbmUgPSBmaWxlLmluZGV4W2Vycm9yLmxpbmVdXG4gICAgICAgICAgICBpZiAobGluZSkge1xuICAgICAgICAgICAgICAgIGxpbmUuZXJyb3JzLnB1c2goZXJyb3IpXG4gICAgICAgICAgICAgICAgZmlsZS5oYXNFcnJvcnMgPSB0cnVlXG4gICAgICAgICAgICAgICAgcmV0dXJuXG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgZmlsZXMudW5rbm93bi5oYXNFcnJvcnMgPSB0cnVlXG4gICAgICAgIGZpbGVzLnVua25vd24ubGluZXNbMF0uZXJyb3JzLnB1c2goZXJyb3IpXG4gICAgfSlcbn1cbiIsIi8vIGltcG9ydCB7IFRlbnNvciwgT3V0cHV0VGVuc29yLCBJblBsYWNlVGVuc29yIH0gZnJvbSAnLi4vdGVuc29yL2luZGV4LmpzJ1xuaW1wb3J0IEJhc2VUZW5zb3IgZnJvbSAnLi4vdGVuc29yL2Jhc2UuanMnXG5cbmltcG9ydCB7IHJlYWRGaWxlU3luYyB9IGZyb20gJ2ZzJztcblxuY29uc3QgVEVOU09SX0ZSQUdNRU5UX0hFQURFUiA9IHJlYWRGaWxlU3luYyhfX2Rpcm5hbWUgKyAnLy4uL2Zvcm1hdC91dGlsLmdsc2wnLCAndXRmOCcpXG5cblxuZXhwb3J0IGRlZmF1bHQgZnVuY3Rpb24gYXNzZW1ibGVGcmFnbWVudFNoYWRlcihzaGFkZXJHZW4sIG91dHB1dCwgdW5pZm9ybXMpe1xuICAgIHZhciB0ZW5zb3JTaGFkZXIgPSBzaGFkZXJHZW4odW5pZm9ybXMsIG91dHB1dCk7XG4gICAgXG4gICAgdmFyIGZyYWdtZW50U2hhZGVyID0gVEVOU09SX0ZSQUdNRU5UX0hFQURFUjtcbiAgICBmb3IobGV0IHVuaWZvcm0gaW4gdW5pZm9ybXMpe1xuICAgICAgICBpZih1bmlmb3Jtc1t1bmlmb3JtXSBpbnN0YW5jZW9mIEJhc2VUZW5zb3Ipe1xuICAgICAgICAgICAgbGV0IHRlbnNvciA9IHVuaWZvcm1zW3VuaWZvcm1dO1xuXG4gICAgICAgICAgICBmcmFnbWVudFNoYWRlciArPSB0ZW5zb3IuX2Zvcm1hdC5jb2RlYy5kZWNvZGVTaGFkZXIucmVwbGFjZSgvQC9nLCB1bmlmb3JtICsgJ18nKSArICdcXG4nXG4gICAgICAgICAgICBmcmFnbWVudFNoYWRlciArPSB0ZW5zb3IuX2Zvcm1hdC5wYWNrLnJlYWRTaGFkZXIucmVwbGFjZSgvQC9nLCB1bmlmb3JtICsgJ18nKSArICdcXG4nXG5cbiAgICAgICAgICAgIGlmKCh0ZW5zb3IuZm9ybWF0LmRlbnNpdHkgPT0gJzE6NCcgJiYgKG5ldyBSZWdFeHAodW5pZm9ybSArICdfcmVhZDRcXFxcYicpKS50ZXN0KHRlbnNvclNoYWRlcikpIHx8IFxuICAgICAgICAgICAgICAgICh0ZW5zb3IuZm9ybWF0LmRlbnNpdHkgPT0gJzQ6NCcgJiYgKG5ldyBSZWdFeHAodW5pZm9ybSArICdfcmVhZFxcXFxiJykpLnRlc3QodGVuc29yU2hhZGVyKSkpe1xuICAgICAgICAgICAgICAgIGZyYWdtZW50U2hhZGVyICs9IHRlbnNvci5fZm9ybWF0LnJlYWRfc2hpbS5yZXBsYWNlKC9AL2csIHVuaWZvcm0gKyAnXycpICsgJ1xcbic7XG4gICAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICB2YXIgYWN0aXZhdGlvbiA9ICh0eXBlb2YgdW5pZm9ybXMuX2FjdGl2YXRpb24gPT0gJ3N0cmluZycgJiYgdW5pZm9ybXMuX2FjdGl2YXRpb24gIT0gJ2xpbmVhcicpID9cbiAgICAgICAgdW5pZm9ybXMuX2FjdGl2YXRpb24udG9Mb3dlckNhc2UoKSA6ICdsaW5lYXInO1xuXG4gICAgaWYoIShhY3RpdmF0aW9uIGluIG91dHB1dC5fZm9ybWF0LmFjdGl2YXRpb25zKSlcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdVbmtub3duIGFjdGl2YXRpb24gdHlwZSAnICsgYWN0aXZhdGlvbik7XG5cbiAgICBmcmFnbWVudFNoYWRlciArPSBvdXRwdXQuX2Zvcm1hdC5hY3RpdmF0aW9uc1thY3RpdmF0aW9uXS5yZXBsYWNlKC9AL2csICdvdXRfJykgKyAnXFxuJztcbiAgICBmcmFnbWVudFNoYWRlciArPSBvdXRwdXQuX2Zvcm1hdC5jb2RlYy5lbmNvZGVTaGFkZXIucmVwbGFjZSgvQC9nLCAnb3V0XycpICsgJ1xcbic7XG4gICAgZnJhZ21lbnRTaGFkZXIgKz0gb3V0cHV0Ll9mb3JtYXQucGFjay53cml0ZVNoYWRlci5yZXBsYWNlKC9AL2csICdvdXRfJykgKyAnXFxuJztcblxuXG4gICAgaWYoKG91dHB1dC5mb3JtYXQuZGVuc2l0eSA9PSAnMTo0JyAmJiAvcHJvY2VzczRcXGIvLnRlc3QodGVuc29yU2hhZGVyKSkgfHwgXG4gICAgICAgIChvdXRwdXQuZm9ybWF0LmRlbnNpdHkgPT0gJzQ6NCcgJiYgL3Byb2Nlc3NcXGIvLnRlc3QodGVuc29yU2hhZGVyKSkpe1xuICAgICAgICBmcmFnbWVudFNoYWRlciArPSBvdXRwdXQuX2Zvcm1hdC53cml0ZV9zaGltLnJlcGxhY2UoL0AvZywgJ291dF8nKSArICdcXG4nO1xuICAgIH1cblxuICAgIGZyYWdtZW50U2hhZGVyICs9IHRlbnNvclNoYWRlci5yZXBsYWNlKC9AL2csICdvdXRfJylcblxuICAgIC8vIGNvbnNvbGUubG9nKGZyYWdtZW50U2hhZGVyKVxuXG4gICAgcmV0dXJuIGZyYWdtZW50U2hhZGVyO1xufSIsImltcG9ydCBnZXRUZW5zb3JQcm9ncmFtIGZyb20gJy4vcHJvZ3JhbS5qcydcbmltcG9ydCBhc3NlbWJsZUZyYWdtZW50U2hhZGVyIGZyb20gJy4vZnJhZy5qcydcbmltcG9ydCB7IFRlbnNvciwgT3V0cHV0VGVuc29yLCBJblBsYWNlVGVuc29yIH0gZnJvbSAnLi4vdGVuc29yL2luZGV4LmpzJ1xuaW1wb3J0IHsgY2hlY2tGcmFtZWJ1ZmZlckVycm9yIH0gZnJvbSAnLi9jaGVjay5qcydcbmltcG9ydCBUTlNMIGZyb20gJy4vdG5zbC5qcydcbmltcG9ydCB7IGJlZ2luVGltZXIsIGVuZFRpbWVyLCBub3cgfSBmcm9tICcuL3RpbWVyLmpzJ1xuXG5cbmV4cG9ydCBmdW5jdGlvbiBDb21waWxlKHNoYWRlckdlbiwgb3V0cHV0LCB1bmlmb3JtcyA9IHt9KXtcbiAgICB2YXIgc3RhcnRUaW1lID0gbm93KCk7XG4gICAgaWYoIShvdXRwdXQgaW5zdGFuY2VvZiBPdXRwdXRUZW5zb3IpKSBcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFwiRmlyc3QgYXJndW1lbnQgbXVzdCBiZSBhbiBpbnN0YW5jZSBvZiBPdXRwdXRUZW5zb3JcIik7XG4gICAgXG4gICAgaWYodHlwZW9mIHNoYWRlckdlbiA9PT0gJ3N0cmluZycpIHNoYWRlckdlbiA9IFROU0woc2hhZGVyR2VuKTtcbiAgICBcbiAgICB2YXIgZ2wgPSBvdXRwdXQuZ2w7XG4gICAgdmFyIHByb2dyYW0gPSBnZXRUZW5zb3JQcm9ncmFtKGdsLCBhc3NlbWJsZUZyYWdtZW50U2hhZGVyKHNoYWRlckdlbiwgb3V0cHV0LCB1bmlmb3JtcykpO1xuICAgIHZhciBjb21waWxlVGltZSA9IG5vdygpIC0gc3RhcnRUaW1lO1xuICAgIC8vIGNvbnNvbGUubG9nKCdDb21waWxlIFRpbWUnLCBjb21waWxlVGltZSlcbiAgICByZXR1cm4gcHJvZ3JhbTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIFJ1bihzaGFkZXJHZW4sIG91dHB1dCwgdW5pZm9ybXMgPSB7fSwgY2FsbGJhY2sgPSBudWxsKXtcbiAgICB2YXIgdHAgPSBDb21waWxlKHNoYWRlckdlbiwgb3V0cHV0LCB1bmlmb3Jtcyk7XG5cbiAgICB2YXIgZ2wgPSBvdXRwdXQuZ2w7XG4gICAgXG4gICAgaWYoY2FsbGJhY2sgJiYgdHlwZW9mIGNhbGxiYWNrICE9ICdmdW5jdGlvbicpIHRocm93IG5ldyBFcnJvcignQ2FsbGJhY2sgbXVzdCBiZSBhIGZ1bmN0aW9uJyk7XG4gICAgaWYoY2FsbGJhY2spe1xuICAgICAgICBiZWdpblRpbWVyKGdsLCB7XG4gICAgICAgICAgICBzaGFkZXI6IHNoYWRlckdlbixcbiAgICAgICAgICAgIG91dHB1dDogb3V0cHV0XG4gICAgICAgIH0pXG4gICAgfVxuXG4gICAgZ2wudXNlUHJvZ3JhbSh0cC5wcm9ncmFtKTtcbiAgICBnbC5kaXNhYmxlKGdsLkRFUFRIX1RFU1QpO1xuICAgIGdsLmRpc2FibGUoZ2wuQkxFTkQpO1xuXG4gICAgdmFyIHNldFVuaWZvcm0gPSB0cC5zZXRVbmlmb3JtLFxuICAgICAgICB0ZXhJbmRleCA9IDAsXG4gICAgICAgIG11c3RTd2FwID0gZmFsc2U7XG4gICAgICAgIFxuICAgIGZvcihsZXQgbmFtZSBpbiB1bmlmb3Jtcyl7XG4gICAgICAgIGlmKG5hbWUuc3RhcnRzV2l0aCgnXycpKSBjb250aW51ZTtcbiAgICAgICAgXG4gICAgICAgIGlmKChuYW1lICsgJ190ZXgnKSBpbiB0cC51bmlmb3JtVHlwZXMpe1xuICAgICAgICAgICAgbGV0IHRlbnNvciA9IHVuaWZvcm1zW25hbWVdO1xuICAgICAgICAgICAgaWYodGVuc29yLmdsICE9PSBvdXRwdXQuZ2wpIHRocm93IG5ldyBFcnJvcignVW5pZm9ybXMgbXVzdCBiZWxvbmcgdG8gc2FtZSBHTCBjb250ZXh0IGFzIG91dHB1dCcpO1xuICAgICAgICAgICAgaWYodGVuc29yID09PSBvdXRwdXQpIG11c3RTd2FwID0gdHJ1ZTtcblxuICAgICAgICAgICAgZm9yKGxldCB1bmlmb3JtIGluIHRlbnNvci5pbmZvKXtcbiAgICAgICAgICAgICAgICBzZXRVbmlmb3JtKG5hbWUgKyAnXycgKyB1bmlmb3JtLCB0ZW5zb3IuaW5mb1t1bmlmb3JtXSlcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgZ2wuYWN0aXZlVGV4dHVyZShnbFsnVEVYVFVSRScgKyB0ZXhJbmRleF0pO1xuICAgICAgICAgICAgZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGVuc29yLnRleCk7XG4gICAgICAgICAgICBzZXRVbmlmb3JtKG5hbWUgKyAnX3RleCcsIHRleEluZGV4KTtcblxuICAgICAgICAgICAgdGV4SW5kZXgrK1xuICAgICAgICB9ZWxzZSBpZihuYW1lIGluIHRwLnVuaWZvcm1UeXBlcyl7XG4gICAgICAgICAgICBzZXRVbmlmb3JtKG5hbWUsIHVuaWZvcm1zW25hbWVdKVxuICAgICAgICB9ZWxzZXtcbiAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcihcIlVua25vd24gdW5pZm9ybSBcIiArIG5hbWUpO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgLy8gT3JkaW5hcmlseSB3ZSBjYW4ndCB3cml0ZSB0byB0aGUgc2FtZSB0ZXh0dXJlIHRoYXQgd2UncmUgdXNpbmcgYXNcbiAgICAvLyBhbiBpbnB1dCwgYXMgdGhpcyBjb3VsZCBsZWFkIHRvIGFsbCBzb3J0cyBvZiB0ZXJyaWJsZSByYWNlIGNvbmRpdGlvbnMsXG4gICAgLy8gdW5kZWZpbmVkIGJlaGF2aW9yLCBhbmQgaW52YWxpZCBzdGF0ZS4gSW5QbGFjZVRlbnNvcnMgYWN0dWFsbHkgY29uc2lzdFxuICAgIC8vIG9mIGEgcGFpciBvZiB0ZXh0dXJlcyB3aGljaCBhcmUgc3dhcHBlZCBmb3IgdGhlc2UgaW4tcGxhY2Ugb3BlcmF0aW9ucy4gXG4gICAgaWYobXVzdFN3YXApIG91dHB1dC5zd2FwKCk7XG5cbiAgICBmb3IobGV0IHVuaWZvcm0gaW4gb3V0cHV0LmluZm8pe1xuICAgICAgICBzZXRVbmlmb3JtKCdvdXRfJyArIHVuaWZvcm0sIG91dHB1dC5pbmZvW3VuaWZvcm1dKVxuICAgIH1cblxuICAgIGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgb3V0cHV0LmZibyk7XG4gICAgZ2wudmlld3BvcnQoMCwgMCwgb3V0cHV0LmluZm8udGV4U2l6ZVswXSwgb3V0cHV0LmluZm8udGV4U2l6ZVsxXSk7XG4gICAgZ2wuZHJhd0FycmF5cyhnbC5UUklBTkdMRV9TVFJJUCwgMCwgNCk7IC8vIGRyYXcgdG8gZnJhbWVidWZmZXJcblxuICAgIGNoZWNrRnJhbWVidWZmZXJFcnJvcihnbCk7XG4gICAgXG4gICAgLy8gdmFyIHJ1blRpbWUgPSBub3coKSAtIHN0YXJ0VGltZTtcbiAgICAvLyB0aW1lci5lbmQoKVxuICAgIGlmKGNhbGxiYWNrKXtcbiAgICAgICAgZW5kVGltZXIoZ2wsIGZ1bmN0aW9uKGluZm8pe1xuICAgICAgICAgICAgLy8gY29uc29sZS5sb2coJ0dQVSB0aW1lOiAnLCBpbmZvKVxuICAgICAgICAgICAgY2FsbGJhY2soaW5mbyk7XG4gICAgICAgIH0pICAgIFxuICAgIH1cbiAgICAvLyBjb25zb2xlLmxvZygnQ1BVIFJ1biBUaW1lJywgcnVuVGltZSlcblxuICAgIHJldHVybiBvdXRwdXQ7XG59IiwiaW1wb3J0IHsgY2hlY2tMaW5rRXJyb3IsIGNoZWNrU2hhZGVyRXJyb3IgfSBmcm9tICcuL2NoZWNrLmpzJ1xuXG5jb25zdCBURU5TT1JfVkVSVEVYX1NIQURFUiA9IGBcbiAgICBwcmVjaXNpb24gaGlnaHAgZmxvYXQ7XG4gICAgYXR0cmlidXRlIHZlYzIgYV9wb3NpdGlvbjtcbiAgICB2b2lkIG1haW4oKSB7XG4gICAgICAgIGdsX1Bvc2l0aW9uID0gdmVjNChhX3Bvc2l0aW9uLCAwLCAxKTtcbiAgICB9XG5gXG5cblxuY29uc3QgVU5JRk9STV9TRVRURVJTID0geyB2ZWM0OiAnNGZ2JywgdmVjMzogJzNmdicsIHZlYzI6ICcyZnYnLCBmbG9hdDogJzFmJyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgaXZlYzQ6ICc0aXYnLCBpdmVjMzogJzNpdicsIGl2ZWMyOiAnMml2JywgaW50OiAnMWknLFxuICAgICAgICAgICAgICAgICAgICAgICAgICBzYW1wbGVyMkQ6ICcxaScgfTtcblxuZXhwb3J0IGRlZmF1bHQgZnVuY3Rpb24gZ2V0VGVuc29yUHJvZ3JhbShnbCwgZnJhZ21lbnRTaGFkZXIpe1xuICAgIGlmKCFnbC5fdGVuc29yUHJvZ3JhbXMpIGdsLl90ZW5zb3JQcm9ncmFtcyA9IHt9O1xuICAgIGlmKGZyYWdtZW50U2hhZGVyIGluIGdsLl90ZW5zb3JQcm9ncmFtcyl7XG4gICAgICAgIHJldHVybiBnbC5fdGVuc29yUHJvZ3JhbXNbZnJhZ21lbnRTaGFkZXJdXG4gICAgfVxuICAgIHZhciBwcm9ncmFtID0gY3JlYXRlVGVuc29yUHJvZ3JhbShnbCwgZnJhZ21lbnRTaGFkZXIpO1xuICAgIGdsLl90ZW5zb3JQcm9ncmFtc1tmcmFnbWVudFNoYWRlcl0gPSBwcm9ncmFtO1xuICAgIHJldHVybiBwcm9ncmFtO1xufVxuXG5mdW5jdGlvbiBjcmVhdGVUZW5zb3JQcm9ncmFtKGdsLCBmcmFnbWVudFNoYWRlcil7XG4gICAgdmFyIHByb2dyYW0gPSBjcmVhdGVTaGFkZXJQcm9ncmFtKGdsLCBURU5TT1JfVkVSVEVYX1NIQURFUiwgZnJhZ21lbnRTaGFkZXIpO1xuICAgIFxuICAgIGdsLnVzZVByb2dyYW0ocHJvZ3JhbSk7XG4gICAgYmluZEF0dHJpYnV0ZUJ1ZmZlcihnbCwgcHJvZ3JhbSk7XG5cbiAgICB2YXIgdW5pZm9ybVR5cGVzID0gZXh0cmFjdFVuaWZvcm1EZWNsYXJhdGlvbnMoZnJhZ21lbnRTaGFkZXIpLFxuICAgICAgICB1bmlmb3JtTG9jcyA9IHt9O1xuXG4gICAgZnVuY3Rpb24gYWRkVW5pZm9ybShuYW1lLCB0eXBlKXtcbiAgICAgICAgdW5pZm9ybUxvY3NbbmFtZV0gPSB7IGxvYzogZ2wuZ2V0VW5pZm9ybUxvY2F0aW9uKHByb2dyYW0sIG5hbWUpLCB0eXBlOiB0eXBlIH1cbiAgICB9XG5cbiAgICBmb3IobGV0IG5hbWUgaW4gdW5pZm9ybVR5cGVzKXtcbiAgICAgICAgbGV0IHR5cGUgPSB1bmlmb3JtVHlwZXNbbmFtZV07XG4gICAgICAgIGlmKCh0eXBlKSBpbiBVTklGT1JNX1NFVFRFUlMpe1xuICAgICAgICAgICAgYWRkVW5pZm9ybShuYW1lLCB0eXBlKTtcbiAgICAgICAgfWVsc2UgdGhyb3cgbmV3IEVycm9yKFwiVW5rbm93biB1bmlmb3JtIHR5cGUgXCIgKyB0eXBlKTtcbiAgICB9XG5cbiAgICBmdW5jdGlvbiBzZXRVbmlmb3JtKG5hbWUsIHZhbHVlKXtcbiAgICAgICAgaWYoIShuYW1lIGluIHVuaWZvcm1Mb2NzKSl7XG4gICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXCJDb3VsZCBub3QgZmluZCB1bmlmb3JtIFwiICsgbmFtZSk7XG4gICAgICAgIH1cbiAgICAgICAgZ2xbJ3VuaWZvcm0nICsgVU5JRk9STV9TRVRURVJTW3VuaWZvcm1Mb2NzW25hbWVdLnR5cGVdXSh1bmlmb3JtTG9jc1tuYW1lXS5sb2MsIHZhbHVlKVxuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICAgIHByb2dyYW06IHByb2dyYW0sXG4gICAgICAgIHVuaWZvcm1Mb2NzOiB1bmlmb3JtTG9jcyxcbiAgICAgICAgdW5pZm9ybVR5cGVzOiB1bmlmb3JtVHlwZXMsXG4gICAgICAgIHNldFVuaWZvcm06IHNldFVuaWZvcm0sXG4gICAgfVxufVxuXG5cbmV4cG9ydCBmdW5jdGlvbiBiaW5kQXR0cmlidXRlQnVmZmVyKGdsLCBwcm9ncmFtKSB7XG4gICAgZ2wuYmluZEJ1ZmZlcihnbC5BUlJBWV9CVUZGRVIsIGdsLmNyZWF0ZUJ1ZmZlcigpKTtcbiAgICBnbC5idWZmZXJEYXRhKGdsLkFSUkFZX0JVRkZFUiwgbmV3IEZsb2F0MzJBcnJheShbIC0xLC0xLCAxLC0xLCAtMSwgMSwgMSwgMV0pLCBnbC5TVEFUSUNfRFJBVyk7XG5cbiAgICB2YXIgcG9zaXRpb25Mb2NhdGlvbiA9IGdsLmdldEF0dHJpYkxvY2F0aW9uKHByb2dyYW0sIFwiYV9wb3NpdGlvblwiKTtcbiAgICBnbC5lbmFibGVWZXJ0ZXhBdHRyaWJBcnJheShwb3NpdGlvbkxvY2F0aW9uKTtcbiAgICBnbC52ZXJ0ZXhBdHRyaWJQb2ludGVyKHBvc2l0aW9uTG9jYXRpb24sIDIsIGdsLkZMT0FULCBmYWxzZSwgMCwgMCk7XG59XG5cblxuZnVuY3Rpb24gZXh0cmFjdFVuaWZvcm1EZWNsYXJhdGlvbnMoc3RyKXtcbiAgICB2YXIgdW5pZm9ybXMgPSB7fTtcbiAgICBzdHIgPSBzdHIucmVwbGFjZSgvKCg/OlxcL1xcKig/OlteKl18KD86XFwqK1teKlxcL10pKSpcXCorXFwvKXwoPzpcXC9cXC8uKikpL2csICcnKVxuICAgIHN0ciA9IHN0ci5yZXBsYWNlKC9cXC9cXC8uKlxcbi9nLCAnJylcbiAgICB2YXIgbSwgcmUgPSAvdW5pZm9ybVxccyooW1xcd19dKylcXHMqKFtcXHdfXSspL2c7XG4gICAgd2hpbGUgKG0gPSByZS5leGVjKHN0cikpIHVuaWZvcm1zW21bMl1dID0gbVsxXTtcbiAgICByZXR1cm4gdW5pZm9ybXM7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVTaGFkZXJQcm9ncmFtKGdsLCB2ZXJ0ZXhTb3VyY2UsIGZyYWdtZW50U291cmNlKSB7XG4gICAgdmFyIHZlcnRleFNoYWRlciA9IGNvbXBpbGVTaGFkZXIoZ2wsIHZlcnRleFNvdXJjZSwgZ2wuVkVSVEVYX1NIQURFUik7XG4gICAgdmFyIGZyYWdtZW50U2hhZGVyID0gY29tcGlsZVNoYWRlcihnbCwgZnJhZ21lbnRTb3VyY2UsIGdsLkZSQUdNRU5UX1NIQURFUik7XG5cbiAgICAvLyB2YXIgZGVidWcgPSBnbC5nZXRFeHRlbnNpb24oJ1dFQkdMX2RlYnVnX3NoYWRlcnMnKVxuICAgIC8vIGNvbnNvbGUubG9nKGRlYnVnLmdldFRyYW5zbGF0ZWRTaGFkZXJTb3VyY2UodmVydGV4U2hhZGVyKSk7XG4gICAgLy8gY29uc29sZS5sb2coZGVidWcuZ2V0VHJhbnNsYXRlZFNoYWRlclNvdXJjZShmcmFnbWVudFNoYWRlcikpO1xuXG4gICAgdmFyIHByb2dyYW0gPSBnbC5jcmVhdGVQcm9ncmFtKCk7XG4gICAgZ2wuYXR0YWNoU2hhZGVyKHByb2dyYW0sIHZlcnRleFNoYWRlcik7XG4gICAgZ2wuYXR0YWNoU2hhZGVyKHByb2dyYW0sIGZyYWdtZW50U2hhZGVyKTtcbiAgICBnbC5saW5rUHJvZ3JhbShwcm9ncmFtKTtcblxuICAgIC8vIGludGVyZXN0aW5nbHkgZW5vdWdoIGl0IHNlZW1zIGxpa2UgU2FmYXJpIG5ldmVyIGVtaXRzXG4gICAgLy8gYSBzaGFkZXIgcHJvZ3JhbSBsaW5rIGVycm9yLiBcbiAgICBjaGVja0xpbmtFcnJvcihnbCwgcHJvZ3JhbSwgZnJhZ21lbnRTb3VyY2UsIHZlcnRleFNvdXJjZSk7XG5cbiAgICByZXR1cm4gcHJvZ3JhbTtcbn1cblxuXG5mdW5jdGlvbiBjb21waWxlU2hhZGVyKGdsLCBzaGFkZXJTb3VyY2UsIHNoYWRlclR5cGUpIHtcbiAgICB2YXIgc2hhZGVyID0gZ2wuY3JlYXRlU2hhZGVyKHNoYWRlclR5cGUpO1xuICAgIGdsLnNoYWRlclNvdXJjZShzaGFkZXIsIHNoYWRlclNvdXJjZSk7XG4gICAgZ2wuY29tcGlsZVNoYWRlcihzaGFkZXIpO1xuICAgIHZhciBzdWNjZXNzID0gZ2wuZ2V0U2hhZGVyUGFyYW1ldGVyKHNoYWRlciwgZ2wuQ09NUElMRV9TVEFUVVMpO1xuICAgIGNoZWNrU2hhZGVyRXJyb3IoZ2wsIHNoYWRlciwgc2hhZGVyU291cmNlLCBzaGFkZXJUeXBlKVxuICAgIHJldHVybiBzaGFkZXI7XG59XG5cblxuIiwiZXhwb3J0IGZ1bmN0aW9uIG5vdygpIHtcbiAgICBpZiAodHlwZW9mIHBlcmZvcm1hbmNlID09PSAndW5kZWZpbmVkJykge1xuICAgICAgICByZXR1cm4gRGF0ZS5ub3coKVxuICAgIH0gZWxzZSB7XG4gICAgICAgIHJldHVybiBwZXJmb3JtYW5jZS5ub3coKTtcbiAgICB9XG59XG5cbmZ1bmN0aW9uIGdldFRpbWVyKGdsKXtcblx0aWYoZ2wuTk9fUFJPRklMRSkgcmV0dXJuO1xuXHRpZih0eXBlb2YgZ2wuVElNRVJfUE9PTCA9PT0gJ3VuZGVmaW5lZCcpe1xuXHRcdHZhciBleHRUaW1lciA9IGdsLmdldEV4dGVuc2lvbignZXh0X2Rpc2pvaW50X3RpbWVyX3F1ZXJ5Jyk7XG5cdFx0aWYoIWV4dFRpbWVyIHx8ICFleHRUaW1lci5jcmVhdGVRdWVyeUVYVCl7XG5cdFx0XHRnbC5OT19QUk9GSUxFID0gdHJ1ZTtcblx0XHRcdHJldHVybjtcblx0XHR9XG5cdFx0Z2wuVElNRVJfUE9PTCA9IGNyZWF0ZVRpbWVyKGdsKVxuXHR9XG5cdHJldHVybiBnbC5USU1FUl9QT09MO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmVnaW5UaW1lcihnbCwgaW5mbz17fSl7XG5cdHZhciB0aW1lciA9IGdldFRpbWVyKGdsKTtcblx0aWYodGltZXIpe1xuXHRcdHRpbWVyLmJlZ2luKGluZm8pXG5cdH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGVuZFRpbWVyKGdsLCBjYWxsYmFjayl7XG5cdHZhciB0aW1lciA9IGdldFRpbWVyKGdsKTtcblx0aWYodGltZXIpe1xuXHRcdHRpbWVyLmVuZChjYWxsYmFjaylcblx0fWVsc2UgaWYoY2FsbGJhY2spe1xuXHRcdGNvbnNvbGUud2FybihcIkNhbiBub3QgdHJpZ2dlciBjYWxsYmFjazogaW1wbGVtZW50YXRpb24gZG9lcyBub3Qgc3VwcG9ydCBleHRfZGlzam9pbnRfdGltZXJfcXVlcnlcIilcblx0fVxufVxuXG5mdW5jdGlvbiBjcmVhdGVUaW1lcihnbCl7XHRcblx0dmFyIGV4dFRpbWVyID0gZ2wuZ2V0RXh0ZW5zaW9uKCdleHRfZGlzam9pbnRfdGltZXJfcXVlcnknKTtcblxuXHR2YXIgcXVlcnlQb29sID0gW11cbiAgICBmdW5jdGlvbiBhbGxvY1F1ZXJ5ICgpIHtcbiAgICAgICAgcmV0dXJuIHF1ZXJ5UG9vbC5wb3AoKSB8fCBleHRUaW1lci5jcmVhdGVRdWVyeUVYVCgpXG4gICAgfVxuICAgIGZ1bmN0aW9uIGZyZWVRdWVyeSAocXVlcnkpIHtcbiAgICAgICAgcXVlcnlQb29sLnB1c2gocXVlcnkpXG4gICAgfVxuXG5cdHZhciBwZW5kaW5nUXVlcmllcyA9IFtdXG5cdGZ1bmN0aW9uIGJlZ2luUXVlcnkgKGluZm8pIHtcblx0XHR2YXIgcXVlcnkgPSBhbGxvY1F1ZXJ5KClcblx0XHRleHRUaW1lci5iZWdpblF1ZXJ5RVhUKGV4dFRpbWVyLlRJTUVfRUxBUFNFRF9FWFQsIHF1ZXJ5KVxuXHRcdHBlbmRpbmdRdWVyaWVzLnB1c2goW3F1ZXJ5LCBpbmZvXSlcblx0fVxuXG5cdGZ1bmN0aW9uIGVuZFF1ZXJ5ICgpIHtcblx0XHRleHRUaW1lci5lbmRRdWVyeUVYVChleHRUaW1lci5USU1FX0VMQVBTRURfRVhUKVxuXHR9XG5cblx0ZnVuY3Rpb24gY2FsbGJhY2soaW5mbywgdGltZSl7XG5cdFx0dmFyIGZuID0gaW5mby5jYWxsYmFjaztcblx0XHRpbmZvLmdwdVRpbWUgPSB0aW1lO1xuXHRcdGRlbGV0ZSBpbmZvLmNhbGxiYWNrO1xuXHRcdGlmKGZuKSBmbihpbmZvKTtcblx0fVxuXG5cdGZ1bmN0aW9uIG1vbml0b3JQZW5kaW5nKCl7XG5cdFx0Zm9yICh2YXIgaSA9IDA7IGkgPCBwZW5kaW5nUXVlcmllcy5sZW5ndGg7ICsraSkge1xuICAgICAgXHRcdHZhciBxdWVyeSA9IHBlbmRpbmdRdWVyaWVzW2ldWzBdXG4gICAgICBcdFx0aWYgKGV4dFRpbWVyLmdldFF1ZXJ5T2JqZWN0RVhUKHF1ZXJ5LCBleHRUaW1lci5RVUVSWV9SRVNVTFRfQVZBSUxBQkxFX0VYVCkpIHtcbiAgICAgICAgXHRcdHZhciBxdWVyeVRpbWUgPSBleHRUaW1lci5nZXRRdWVyeU9iamVjdEVYVChxdWVyeSwgZXh0VGltZXIuUVVFUllfUkVTVUxUX0VYVClcbiAgICAgICAgXHRcdGNhbGxiYWNrKHBlbmRpbmdRdWVyaWVzW2ldWzFdLCBxdWVyeVRpbWUgLyAxZTYpXG4gICAgICAgIFx0XHRmcmVlUXVlcnkocXVlcnkpXG4gICAgICAgIFx0XHRwZW5kaW5nUXVlcmllcy5zcGxpY2UoaSwgMSlcbiAgICAgICAgXHRcdGktLVxuICAgICAgXHRcdH1cblx0ICAgIH1cblx0fVxuXG5cblx0dmFyIGlzUG9sbGluZyA9IGZhbHNlO1xuXHRmdW5jdGlvbiBsb29wKCl7XG5cdFx0aWYocGVuZGluZ1F1ZXJpZXMubGVuZ3RoID4gMCl7XG5cdFx0XHRtb25pdG9yUGVuZGluZygpXG5cdFx0XHRyZXF1ZXN0QW5pbWF0aW9uRnJhbWUobG9vcClcblx0XHR9ZWxzZXtcblx0XHRcdGlzUG9sbGluZyA9IGZhbHNlO1xuXHRcdH1cblx0fVxuXG5cdHZhciBjdXJyZW50SW5mbyA9IG51bGw7XG4gICAgcmV0dXJuIHtcbiAgICBcdGJlZ2luKGluZm8gPSB7fSl7XG4gICAgXHRcdGlmKGN1cnJlbnRJbmZvKSB0aHJvdyBuZXcgRXJyb3IoJ2JlZ2luVGltZXIgd2FzIGNhbGxlZCBiZWZvcmUgcHJldmlvdXMgZW5kVGltZXInKTtcbiAgICBcdFx0Y3VycmVudEluZm8gPSBpbmZvXG4gICAgXHRcdGluZm8uY3B1U3RhcnRUaW1lID0gbm93KCk7XG4gICAgXHRcdGJlZ2luUXVlcnkoY3VycmVudEluZm8pXG4gICAgXHR9LFxuXG4gICAgXHRlbmQoZm4pe1xuICAgIFx0XHRjdXJyZW50SW5mby5jcHVUaW1lID0gbm93KCkgLSBjdXJyZW50SW5mby5jcHVTdGFydFRpbWVcbiAgICBcdFx0ZGVsZXRlIGN1cnJlbnRJbmZvLmNwdVN0YXJ0VGltZTtcbiAgICBcdFx0Y3VycmVudEluZm8uY2FsbGJhY2sgPSBmbjtcbiAgICBcdFx0Y3VycmVudEluZm8gPSBudWxsO1xuICAgIFx0XHRlbmRRdWVyeSgpXG5cbiAgICBcdFx0aWYoaXNQb2xsaW5nID09PSBmYWxzZSl7XG4gICAgXHRcdFx0aXNQb2xsaW5nID0gdHJ1ZTtcbiAgICBcdFx0XHRyZXF1ZXN0QW5pbWF0aW9uRnJhbWUobG9vcClcbiAgICBcdFx0fVxuICAgIFx0fVxuICAgIH1cbn0iLCIvLyBUTlNMIChwcm9ub3VuY2VkIHRpbnNlbClcbi8vIGlzIGEgZG9tYWluIHNwZWNpZmljIGxhbmd1YWdlIGJhc2VkIG9uIEdMU0xcbi8vIGZvciBoZWxwaW5nIHdpdGggdGhlIHdyaXRpbmcgY29kZSB0aGF0XG4vLyBjb21wdXRlcyB3aXRoIHRlbnNvcnMuIFxuXG4vLyBBIGxpbWl0YXRpb24gb2YgR0xTTCBpcyB0aGF0IHRoZSBjb25kaXRpb25cbi8vIG9mIGFueSBsb29wIGhhcyB0byBiZSBzdGF0aWNhbGx5IGtub3duIFxuLy8gKGUuZy4gY291bnRlcnMgdXAgdG8gYSBmaXhlZCBjb25zdGFudFxuLy8gdmFsdWUpIHdoaWNoIGlzIHByb2JsZW1hdGljIGlmIHdlIHdhbnRcbi8vIHRvIHdyaXRlIGdlbmVyYWwgY29kZSB0aGF0IGRlcGVuZHMgb25cbi8vIHRoZSBzaXplIG9mIHRoZSBpbnB1dCB0ZW5zb3JzXG5cbi8vIFROU0wgYWRkcyB0aGUgZm9sbG93aW5nIHN5bnRheDpcbi8vICAgICAgIyhpbWFnZS5zaGFwZSlcbi8vIHdoaWNoIHdpbGwgYmUgcmVwbGFjZWQgd2l0aCBhbiBpdmVjNFxuLy8gY29udGFpbmluZyB0aGUgc2hhcGUgb2YgdGhlIGlucHV0IHRlbnNvciBcImltYWdlXCJcbi8vIGF1dG9tYXRpY2FsbHlcblxuZXhwb3J0IGRlZmF1bHQgZnVuY3Rpb24gVE5TTChzdHIpe1xuICAgIGlmKHR5cGVvZiBzdHIgIT0gJ3N0cmluZycpIFxuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ1ROU0wgc2hhZGVyIHByZXByb2Nlc3NvciBvbmx5IGFjY2VwdHMgc3RyaW5ncycpO1xuICAgIFxuICAgIHJldHVybiBmdW5jdGlvbih1bmlmb3Jtcywgb3V0cHV0KXtcbiAgICAgICAgcmV0dXJuIHN0clxuICAgICAgICAvLyBjb21tZW50IG91dCB0aGUgdGVuc29yIHN0cnVjdCBkZWZpbml0aW9uc1xuICAgICAgICAucmVwbGFjZSgvdW5pZm9ybVxccypUZW5zb3JcXHMqKFtcXHdfXSspXFxzKjsvZywgJy8qIChUZW5zb3IgJDEpICovJylcblxuICAgICAgICAvLyB0aGlzIGlzIHRoZSBtYWNybyBzeW50YXhcbiAgICAgICAgLnJlcGxhY2UoL1xcI1xcKChbXFx3XFwuXFxzXSspXFwpL2csIGZ1bmN0aW9uKGFsbCwgYm9keSl7XG4gICAgICAgICAgICB2YXIgb2JqID0gdW5pZm9ybXM7XG4gICAgICAgICAgICBmb3IobGV0IHBhcnQgb2YgYm9keS5zcGxpdCgnLicpKVxuICAgICAgICAgICAgICAgIG9iaiA9IG9ialtwYXJ0LnRyaW0oKV07XG4gICAgICAgICAgICBpZih0eXBlb2Ygb2JqID09ICdudW1iZXInKXtcbiAgICAgICAgICAgICAgICByZXR1cm4gb2JqLnRvU3RyaW5nKClcbiAgICAgICAgICAgIH1lbHNlIGlmKEFycmF5LmlzQXJyYXkob2JqKSAmJiBvYmoubGVuZ3RoIDw9IDQgJiYgb2JqLmxlbmd0aCA+IDEpe1xuICAgICAgICAgICAgICAgIHJldHVybiAob2JqLmV2ZXJ5KE51bWJlci5pc0ludGVnZXIpID8gJ2knIDogJycpICsgXG4gICAgICAgICAgICAgICAgICAgICd2ZWMnICsgb2JqLmxlbmd0aCArICcoJyArIG9iai5qb2luKCcsJykgKyAnKSdcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcignQ2FuIG5vdCBpbmxpbmUgZXhwcmVzc2lvbiAnICsgYm9keSk7XG4gICAgICAgIH0pXG4gICAgICAgIC8vIHRlbnNvci5yZWFkNCh4LCAwKSA9PiB0ZW5zb3IucmVhZDQoaXZlYzQoeCwgMCwgMCwgMCkpXG4gICAgICAgIC8vIHRoaXMgdHJhbnNmb3JtYXRpb24gdGFrZXMgcGxhY2Ugd2hlbiB0aGVyZSBhcmUgMiBvciBtb3JlIGFyZ3VtZW50c1xuICAgICAgICAvLyBhcyBvdGhlcndpc2UgaXQncyBub3QgcG9zc2libGUgdG8gc3RhdGljYWxseSBkZXRlcm1pbmUgd2hldGhlciB4IGlzXG4gICAgICAgIC8vIG9mIHR5cGUgaXZlYzQgb3IgYSBudW1iZXJcbiAgICAgICAgLnJlcGxhY2UoL1xcYihcXHcrKVxccypcXC5cXHMqKHJlYWQ0PylcXGJcXHMqXFwoKFteXFwoXFwpXSspXFwpL2csIGZ1bmN0aW9uKGFsbCwgbmFtZSwgcHJvcCwgYXJnKXtcbiAgICAgICAgICAgIGlmKG5hbWUgaW4gdW5pZm9ybXMgJiYgdW5pZm9ybXNbbmFtZV0uc2hhcGUpe1xuICAgICAgICAgICAgICAgIHZhciBwYXJ0cyA9IGFyZy5zcGxpdCgnLCcpLFxuICAgICAgICAgICAgICAgICAgICBwYWRkZWQgPSBwYXJ0cy5jb25jYXQoWycwJywgJzAnLCAnMCcsICcwJ10uc2xpY2UoMCwgNCAtIHBhcnRzLmxlbmd0aCkpO1xuICAgICAgICAgICAgICAgIGlmKHBhcnRzLmxlbmd0aCA8IDIgfHwgcGFydHMubGVuZ3RoID4gNCkgcmV0dXJuIGFsbDtcbiAgICAgICAgICAgICAgICB2YXIgdmVjID0gJ2l2ZWM0KCcgKyBwYWRkZWQuam9pbignLCcpICsgJyknO1xuICAgICAgICAgICAgICAgIHJldHVybiBuYW1lICsgJ18nICsgcHJvcCArICcoJyArIHZlYyArICcpJztcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHJldHVybiBhbGw7XG4gICAgICAgIH0pXG5cbiAgICAgICAgLy8gdGVuc29yLnNoYXBlID0+IHRlbnNvcl9zaGFwZVxuICAgICAgICAucmVwbGFjZSgvXFxiKFxcdyspXFxzKlxcLlxccyooXFx3KylcXGIvZywgZnVuY3Rpb24oYWxsLCBuYW1lLCBwcm9wKXtcbiAgICAgICAgICAgIGlmKG5hbWUgaW4gdW5pZm9ybXMgJiYgdW5pZm9ybXNbbmFtZV0uc2hhcGUpe1xuICAgICAgICAgICAgICAgIHJldHVybiBuYW1lICsgJ18nICsgcHJvcDtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHJldHVybiBhbGw7XG4gICAgICAgIH0pXG4gICAgICAgIC8vIC5yZXBsYWNlKC9cXCNcXHMqKFxcdyspXFxzKlxcWyguKj8pXFxdL2csIGZ1bmN0aW9uKGFsbCwgdGVuc29yLCBib2R5KXtcbiAgICAgICAgLy8gICAgIHJldHVybiB0ZW5zb3IgKyAnX3JlYWQoaXZlYzQoJyArIGJvZHkgKyAnKSknXG4gICAgICAgIC8vIH0pXG4gICAgfVxufVxuIiwiaW1wb3J0IHsgbWFrZVRleHR1cmUsIG1ha2VGcmFtZUJ1ZmZlciwgY2hlY2tSZW5kZXJGbG9hdCB9IGZyb20gJy4vaGVscGVycy5qcydcbmltcG9ydCBGb3JtYXRzIGZyb20gJy4uL2Zvcm1hdC9pbmRleC5qcydcblxuLy8gVGhlIHRlbnNvciBmb3JtYXQgaXMgYSBKU09OIG9iamVjdCB0aGF0IHNwZWNpZmllcyBob3cgXG4vLyB0aGUgdGVuc29yIGlzIHJlcHJlc2VudGVkIGFzIGEgdGV4dHVyZVxuLy8gaXQgY29uc2lzdHMgb2Ygc2V2ZXJhbCBrZXlzOlxuXG4vLyAgICAgdHlwZTogdWludDggfCBmbG9hdDMyXG4vLyAgICAgZGVuc2l0eTogNDo0IHwgMTo0XG4vLyAgICAgcGFjazogc3RyaWRlIHwgdGlsZVxuLy8gICAgIGNvZGVjOiBcbi8vXHRcdFx0c29mdGZsb2F0IHwgZml4bnVtICgxOjQpXG4vLyAgICAgICAgICByYXcgfCBsaW5xdWFudCAoNDo0KVxuXG5leHBvcnQgZGVmYXVsdCBjbGFzcyBCYXNlVGVuc29yIHtcblx0Ly8gd2UgYXJlbnQgdXNpbmcgYSBjb25zdHJ1Y3RvciBiZWNhdXNlIHdlIHdhbnQgdG8gYmUgYWJsZSB0byBydW5cblx0Ly8gdGhpcyBpbnN0YW5jZW9mIE91dHB1dFRlbnNvciBmcm9tIHdpdGhpbiB0aGUgVGVuc29yIGNvbnN0cnVjdG9yXG5cdFxuXHRfaW5pdChnbCwgZm9ybWF0LCBzaGFwZSwgZGF0YSl7XG5cdFx0Ly8gdmFsaWRhdGUgZ2xjb250ZXh0XG5cdFx0aWYoIWdsLmNyZWF0ZVRleHR1cmUpIHRocm93IG5ldyBFcnJvcignSW52YWxpZCBXZWJHTFJlbmRlcmluZ0NvbnRleHQnKTtcblx0XHR0aGlzLmdsID0gZ2w7XG5cblx0XHQvLyB2YWxpZGF0ZSBzaGFwZVxuXHRcdGlmKCFBcnJheS5pc0FycmF5KHNoYXBlKSkgdGhyb3cgbmV3IEVycm9yKFwic2hhcGUgbXVzdCBiZSBBcnJheVwiKTtcblx0XHRpZihzaGFwZS5sZW5ndGggPiA0KSB0aHJvdyBuZXcgRXJyb3IoXCJUZW5zb3IgbXVzdCBoYXZlIGRpbWVuc2lvbiA8PSA0XCIpO1xuICAgICAgICBpZihzaGFwZS5zb21lKGsgPT4gIWlzRmluaXRlKGspIHx8IGsgPCAxIHx8ICFOdW1iZXIuaXNJbnRlZ2VyKGspKSkgXG4gICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ0ludmFsaWQgc2hhcGU6ICcgKyBzaGFwZSk7XG4gICAgICAgIHNoYXBlID0gc2hhcGUuY29uY2F0KFsxLCAxLCAxLCAxXSkuc2xpY2UoMCwgNClcblx0XHR0aGlzLnNoYXBlID0gc2hhcGU7XG5cdFx0XG5cdFx0Ly8gdmFsaWRhdGUgZm9ybWF0XG5cdFx0aWYoIVsnZmxvYXQzMicsICd1aW50OCddLmluY2x1ZGVzKGZvcm1hdC50eXBlKSlcblx0XHRcdHRocm93IG5ldyBFcnJvcignZm9ybWF0LnR5cGUgbXVzdCBiZSB1aW50OCBvciBmbG9hdDMyJyk7XG5cdFx0aWYoZm9ybWF0LmRlbnNpdHkgaW4gRm9ybWF0cyl7XG5cdFx0XHRsZXQgZmQgPSBGb3JtYXRzW2Zvcm1hdC5kZW5zaXR5XTtcblx0XHRcdGlmKCEoZm9ybWF0LnBhY2sgaW4gZmQucGFjaykpIFxuXHRcdFx0XHR0aHJvdyBuZXcgRXJyb3IoJ2Zvcm1hdC5wYWNrIG11c3QgYmUgJyArIE9iamVjdC5rZXlzKGZkLnBhY2spLmpvaW4oJyBvciAnKSk7XG5cdFx0XHRpZighKGZvcm1hdC5jb2RlYyBpbiBmZC5jb2RlYykpIFxuXHRcdFx0XHR0aHJvdyBuZXcgRXJyb3IoJ2Zvcm1hdC5jb2RlYyBtdXN0IGJlICcgKyBPYmplY3Qua2V5cyhmZC5jb2RlYykuam9pbignIG9yICcpKTtcblx0XHR9ZWxzZSB0aHJvdyBuZXcgRXJyb3IoJ2Zvcm1hdC5kZW5zaXR5IG11c3QgYmUgJyArIE9iamVjdC5rZXlzKEZvcm1hdHMpLmpvaW4oJyBvciAnKSk7XG5cblx0XHR0aGlzLmZvcm1hdCA9IGZvcm1hdDtcblxuXHRcdC8vIGNhbGN1bGF0ZSB0ZXh0dXJlIHNpemVcblx0XHR0aGlzLmluZm8gPSBPYmplY3QuYXNzaWduKHt9LFxuXHRcdFx0dGhpcy5fZm9ybWF0LnBhY2suaW5pdChzaGFwZSwgZm9ybWF0KSxcblx0XHRcdHRoaXMuX2Zvcm1hdC5jb2RlYy5pbml0KHNoYXBlLCBmb3JtYXQpXG5cdFx0KTtcblx0XHRpZighdGhpcy5pbmZvLnRleFNpemUpIHRocm93IG5ldyBFcnJvcignRm9ybWF0IGRpZCBub3QgeWllbGQgdGV4U2l6ZScpO1xuXG5cdFx0Ly8gaW5pdGlhbGl6ZSB0ZXh0dXJlXG5cdFx0dGhpcy50ZXggPSBtYWtlVGV4dHVyZShnbCk7XG5cdFx0dGhpcy51cGRhdGUoZGF0YSlcblx0fVxuXHRfdXBkYXRlKGRhdGEpe1xuXHRcdGlmKGRhdGEgIT09IG51bGwpe1xuXHRcdFx0aWYodGhpcy5mb3JtYXQudHlwZSA9PT0gJ3VpbnQ4Jyl7XG5cdFx0XHRcdGlmKEFycmF5LmlzQXJyYXkoZGF0YSkgfHwgZGF0YSBpbnN0YW5jZW9mIFVpbnQ4Q2xhbXBlZEFycmF5KVxuXHRcdFx0XHRcdGRhdGEgPSBuZXcgVWludDhBcnJheShkYXRhKTtcblx0XHRcdFx0aWYoIShkYXRhIGluc3RhbmNlb2YgVWludDhBcnJheSkpXG5cdFx0XHRcdFx0dGhyb3cgbmV3IEVycm9yKCdkYXRhIG11c3QgYmUgVWludDhBcnJheScpO1xuXHRcdFx0fWVsc2UgaWYodGhpcy5mb3JtYXQudHlwZSA9PT0gJ2Zsb2F0MzInKXtcblx0XHRcdFx0aWYoQXJyYXkuaXNBcnJheShkYXRhKSB8fCBkYXRhIGluc3RhbmNlb2YgRmxvYXQ2NEFycmF5KVxuXHRcdFx0XHRcdGRhdGEgPSBuZXcgRmxvYXQzMkFycmF5KGRhdGEpO1xuXHRcdFx0XHRpZighKGRhdGEgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkpKVxuXHRcdFx0XHRcdHRocm93IG5ldyBFcnJvcignZGF0YSBtdXN0IGJlIEZsb2F0MzJBcnJheScpO1xuXHRcdFx0fWVsc2UgdGhyb3cgbmV3IEVycm9yKCdUeXBlIG11c3QgYmUgdWludDggb3IgZmxvYXQzMicpO1xuXHRcdFx0aWYoZGF0YS5sZW5ndGggIT09IHRoaXMuaW5mby50ZXhTaXplWzBdICogdGhpcy5pbmZvLnRleFNpemVbMV0gKiA0KVxuXHRcdFx0XHR0aHJvdyBuZXcgRXJyb3IoJ2RhdGEgaXMgdGhlIHdyb25nIGxlbmd0aCcpO1xuXHRcdH1cblx0XHQvLyBpZihkYXRhKSBjb25zb2xlLmxvZygnX3VwZGF0ZScsIGRhdGEpO1xuXHRcdHZhciBnbCA9IHRoaXMuZ2w7XG4gICAgICAgIGdsLmJpbmRUZXh0dXJlKGdsLlRFWFRVUkVfMkQsIHRoaXMudGV4KTtcbiAgICAgICAgZ2wudGV4SW1hZ2UyRChnbC5URVhUVVJFXzJELCAwLCBnbC5SR0JBLCBcbiAgICAgICAgXHR0aGlzLmluZm8udGV4U2l6ZVswXSwgdGhpcy5pbmZvLnRleFNpemVbMV0sIDAsIGdsLlJHQkEsIFxuICAgICAgICBcdHRoaXMuZm9ybWF0LnR5cGUgPT0gJ3VpbnQ4JyA/IGdsLlVOU0lHTkVEX0JZVEUgOiBnbC5GTE9BVCwgZGF0YSk7XG5cdH1cblxuXHR1cGRhdGUoZGF0YSl7XG5cdFx0aWYoIWRhdGEpIHJldHVybiB0aGlzLl91cGRhdGUobnVsbCk7XG5cdFx0aWYoZGF0YS5zaGFwZSkgcmV0dXJuIHRoaXMuX3VwZGF0ZShcblx0XHRcdHRoaXMuX2Zvcm1hdC5wYWNrLnBhY2sodGhpcy5pbmZvLCBkYXRhLCB0aGlzLl9mb3JtYXQuY29kZWMuZW5jb2RlLCB0aGlzLmZvcm1hdCkpO1xuXHRcdGlmKHRoaXMudHlwZSAhPSAndWludDgnKSBjb25zb2xlLndhcm4oJ0NhbGxpbmcgdXBkYXRlIHdpdGggcmF3IFR5cGVkQXJyYXkgbWF5IG5vdCB3b3JrIGFjcm9zcyBhbGwgYnJvd3NlcnMuJyk7XG5cdFx0cmV0dXJuIHRoaXMuX3VwZGF0ZShkYXRhKTtcblx0fVxuXG5cdGdldCBfZm9ybWF0KCl7XG5cdFx0cmV0dXJuIHtcblx0XHRcdHBhY2s6IEZvcm1hdHNbdGhpcy5mb3JtYXQuZGVuc2l0eV0ucGFja1t0aGlzLmZvcm1hdC5wYWNrXSxcblx0XHRcdGNvZGVjOiBGb3JtYXRzW3RoaXMuZm9ybWF0LmRlbnNpdHldLmNvZGVjW3RoaXMuZm9ybWF0LmNvZGVjXSxcblx0XHRcdGFjdGl2YXRpb25zOiBGb3JtYXRzW3RoaXMuZm9ybWF0LmRlbnNpdHldLmFjdGl2YXRpb25zLFxuXHRcdFx0cmVhZF9zaGltOiBGb3JtYXRzW3RoaXMuZm9ybWF0LmRlbnNpdHldLnJlYWRfc2hpbSxcblx0XHRcdHdyaXRlX3NoaW06IEZvcm1hdHNbdGhpcy5mb3JtYXQuZGVuc2l0eV0ud3JpdGVfc2hpbVxuXHRcdH1cblx0fVxuXG4gICAgZGVzdHJveSgpeyB0aGlzLmdsLmRlbGV0ZVRleHR1cmUodGhpcy50ZXgpIH1cbn0iLCJpbXBvcnQgeyBiaW5kQXR0cmlidXRlQnVmZmVyLCBjcmVhdGVTaGFkZXJQcm9ncmFtIH0gZnJvbSAnLi4vcnVudGltZS9wcm9ncmFtLmpzJ1xuaW1wb3J0IHsgbWFrZUZyYW1lQnVmZmVyLCBtYWtlVGV4dHVyZSB9IGZyb20gJy4vaGVscGVycy5qcydcblxuZXhwb3J0IGRlZmF1bHQgZnVuY3Rpb24gcnVuRmVhdHVyZVRlc3RzKGdsKXtcbiAgICBcbiAgICBpZighZ2wuRkxPQVRfVEVYVFVSRVNfVEVTVEVEICYmICFnbC5OT19GTE9BVF9URVhUVVJFUyl7XG4gICAgICAgIGlmKCFnbC5nZXRFeHRlbnNpb24oJ09FU190ZXh0dXJlX2Zsb2F0Jykpe1xuICAgICAgICAgICAgY29uc29sZS5pbmZvKFwiVGhpcyBicm93c2VyIGRvZXMgbm90IHNlZW0gdG8gc3VwcG9ydCBPRVNfdGV4dHVyZV9mbG9hdC4gXCJcbiAgICAgICAgICAgICAgICArIFwiVXNpbmcgZmxvYXQgY29kZWMgd29ya2Fyb3VuZCBmcm9tIG5vdyBvbi5cIilcbiAgICAgICAgICAgIGdsLk5PX0ZMT0FUX1RFWFRVUkVTID0gdHJ1ZTtcbiAgICAgICAgfVxuICAgICAgICBnbC5GTE9BVF9URVhUVVJFU19URVNURUQgPSB0cnVlO1xuICAgIH1cblxuICAgIGlmKCFnbC5OT19GTE9BVF9URVhUVVJFUyl7XG4gICAgICAgIGlmKCFnbC5SRU5ERVJfRkxPQVRfVEVTVEVEICYmICFnbC5OT19SRU5ERVJfRkxPQVQpe1xuICAgICAgICAgICAgaWYoIXRlc3RSZW5kZXJGbG9hdChnbCkpe1xuICAgICAgICAgICAgICAgIGNvbnNvbGUuaW5mbyhcIlRoaXMgYnJvd3NlciBzdXBwb3J0cyBPRVNfdGV4dHVyZV9mbG9hdCwgXCIgKyBcbiAgICAgICAgICAgICAgICAgICAgXCJidXQgY2FuIG5vdCByZW5kZXIgdG8gZmxvYXRpbmcgdGV4dHVyZXMuIFwiICsgXG4gICAgICAgICAgICAgICAgICAgIFwiVXNpbmcgZmxvYXQgY29kZWMgd29ya2Fyb3VuZCBmb3Igb3V0cHV0IHRlbnNvcnMgZnJvbSBub3cgb24uXCIpXG4gICAgICAgICAgICAgICAgZ2wuTk9fUkVOREVSX0ZMT0FUID0gdHJ1ZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGdsLlJFTkRFUl9GTE9BVF9URVNURUQgPSB0cnVlO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYoIWdsLlJFQURfRkxPQVRfVEVTVEVEICYmICFnbC5OT19SRUFEX0ZMT0FUICYmICFnbC5OT19SRUFEX0ZMT0FUKXtcbiAgICAgICAgICAgIGlmKCF0ZXN0UmVhZEZsb2F0KGdsKSl7XG4gICAgICAgICAgICAgICAgY29uc29sZS5pbmZvKFwiVGhpcyBicm93c2VyIHN1cHBvcnRzIE9FU190ZXh0dXJlX2Zsb2F0LCBcIiArIFxuICAgICAgICAgICAgICAgICAgICBcImNhbiByZW5kZXIgdG8gZmxvYXRpbmcgcG9pbnQgdGV4dHVyZXMsIGJ1dCBjYW4gbm90IFwiICtcbiAgICAgICAgICAgICAgICAgICAgXCJyZWFkIGludG8gYSBGbG9hdDMyQXJyYXkgYnVmZmVyLiBVc2luZyBmbG9hdCBjb2RlYyBcIiArXG4gICAgICAgICAgICAgICAgICAgIFwid29ya2Fyb3VuZCBmb3IgcmVhZGluZyBmcm9tIG91dHB1dCB0ZW5zb3JzIGZyb20gbm93IG9uLlwiKVxuICAgICAgICAgICAgICAgIGdsLk5PX1JFQURfRkxPQVQgPSB0cnVlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgZ2wuUkVBRF9GTE9BVF9URVNURUQgPSB0cnVlO1xuICAgICAgICB9XG4gICAgfVxuXG5cbn1cblxuXG5jb25zdCBDSEVDS19GTE9BVF9WRVJURVggPSBgXG4gICAgYXR0cmlidXRlIHZlYzIgYV9wb3NpdGlvbjtcbiAgICB2b2lkIG1haW4oKSB7XG4gICAgICAgIGdsX1Bvc2l0aW9uID0gdmVjNChhX3Bvc2l0aW9uLCAwLCAxKTtcbiAgICB9XG5gXG5jb25zdCBDSEVDS19GTE9BVF9GUkFHTUVOVCA9IGBcbiAgICB2b2lkIG1haW4oKSB7XG4gICAgICAgIGdsX0ZyYWdDb2xvciA9IHZlYzQoMy4xNDE1OSwgLTIuNzE4MjgsIDEuNjE4MjgsIDQyKTtcbiAgICB9XG5gO1xuXG4vLyBzb21lIGJyb3dzZXJzIChlLmcuIG1vYmlsZSBzYWZhcmkpIGFyZSBjYXBhYmxlIG9mIGluaXRpYWxpemluZyBmbG9hdGluZyBcbi8vIHBvaW50IHRleHR1cmVzIGJ1dCB1bmFibGUgdG8gd3JpdGUgdG8gdGhlbS4gVGhlIG9ubHkgd2F5IG9mIGZpbmRpbmcgdGhpc1xuLy8gb3V0IGlzIGJ5IHRyeWluZyB0byByZW5kZXIgdG8gYSBmbG9hdGluZyBwb2ludCB0ZXh0dXJlIGFuZCBub3RpY2luZ1xuLy8gdGhlIGludmFsaWQgZnJhbWVidWZmZXIgc3RhdHVzLlxuXG5leHBvcnQgZnVuY3Rpb24gdGVzdFJlbmRlckZsb2F0KGdsKXtcbiAgICB2YXIgdGV4ID0gbWFrZVRleHR1cmUoZ2wpXG4gICAgZ2wudGV4SW1hZ2UyRChnbC5URVhUVVJFXzJELCAwLCBnbC5SR0JBLCAxMCwgMTAsIDAsIGdsLlJHQkEsIGdsLkZMT0FULCBudWxsKTtcbiAgICB2YXIgZmJvID0gbWFrZUZyYW1lQnVmZmVyKGdsLCB0ZXgpO1xuXG4gICAgdmFyIHByb2dyYW0gPSBjcmVhdGVTaGFkZXJQcm9ncmFtKGdsLCBDSEVDS19GTE9BVF9WRVJURVgsIENIRUNLX0ZMT0FUX0ZSQUdNRU5UKTtcbiAgICBnbC51c2VQcm9ncmFtKHByb2dyYW0pO1xuICAgIGJpbmRBdHRyaWJ1dGVCdWZmZXIoZ2wsIHByb2dyYW0pO1xuXG4gICAgZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBmYm8pO1xuICAgIGdsLnZpZXdwb3J0KDAsIDAsIDEwLCAxMCk7XG4gICAgZ2wuZHJhd0FycmF5cyhnbC5UUklBTkdMRV9TVFJJUCwgMCwgNCk7XG5cbiAgICB2YXIgc3RhdHVzID0gZ2wuY2hlY2tGcmFtZWJ1ZmZlclN0YXR1cyhnbC5GUkFNRUJVRkZFUik7XG4gICAgZ2wuZGVsZXRlVGV4dHVyZSh0ZXgpXG4gICAgZ2wuZGVsZXRlRnJhbWVidWZmZXIoZmJvKVxuICAgIGdsLmRlbGV0ZVByb2dyYW0ocHJvZ3JhbSlcblxuICAgIHJldHVybiBzdGF0dXMgPT0gZ2wuRlJBTUVCVUZGRVJfQ09NUExFVEU7XG59XG5cblxuZnVuY3Rpb24gdGVzdFJlYWRGbG9hdChnbCl7XG4gICAgdmFyIHRleCA9IG1ha2VUZXh0dXJlKGdsKVxuICAgIGdsLnRleEltYWdlMkQoZ2wuVEVYVFVSRV8yRCwgMCwgZ2wuUkdCQSwgMTAsIDEwLCAwLCBnbC5SR0JBLCBnbC5GTE9BVCwgbnVsbCk7XG4gICAgdmFyIGZibyA9IG1ha2VGcmFtZUJ1ZmZlcihnbCwgdGV4KTtcblxuICAgIHZhciBwcm9ncmFtID0gY3JlYXRlU2hhZGVyUHJvZ3JhbShnbCwgQ0hFQ0tfRkxPQVRfVkVSVEVYLCBDSEVDS19GTE9BVF9GUkFHTUVOVCk7XG4gICAgZ2wudXNlUHJvZ3JhbShwcm9ncmFtKTtcbiAgICBiaW5kQXR0cmlidXRlQnVmZmVyKGdsLCBwcm9ncmFtKTtcblxuICAgIGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgZmJvKTtcbiAgICBnbC52aWV3cG9ydCgwLCAwLCAxMCwgMTApO1xuICAgIGdsLmRyYXdBcnJheXMoZ2wuVFJJQU5HTEVfU1RSSVAsIDAsIDQpO1xuXG4gICAgdmFyIHNpemUgPSBbMywgM107XG4gICAgdmFyIHBpeGVscyA9IHBpeGVscyA9IG5ldyBGbG9hdDMyQXJyYXkoc2l6ZVswXSAqIHNpemVbMV0gKiA0KVxuICAgIGdsLnJlYWRQaXhlbHMoMCwgMCwgc2l6ZVswXSwgc2l6ZVsxXSwgZ2wuUkdCQSwgZ2wuRkxPQVQsIHBpeGVscyk7XG5cbiAgICBnbC5kZWxldGVUZXh0dXJlKHRleClcbiAgICBnbC5kZWxldGVGcmFtZWJ1ZmZlcihmYm8pXG4gICAgZ2wuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKVxuXG5cbiAgICB2YXIgdG90YWxfZXJyb3IgPSBNYXRoLmFicyhwaXhlbHNbMF0gLSAzLjE0MTU5KSArXG4gICAgICAgICAgICBNYXRoLmFicyhwaXhlbHNbMV0gKyAyLjcxODI4KSArXG4gICAgICAgICAgICBNYXRoLmFicyhwaXhlbHNbMl0gLSAxLjYxODI4KSArXG4gICAgICAgICAgICBNYXRoLmFicyhwaXhlbHNbM10gLSA0Mik7XG5cbiAgICByZXR1cm4gdG90YWxfZXJyb3IgPCAwLjAxO1xufVxuIiwiZXhwb3J0IGZ1bmN0aW9uIG1ha2VGcmFtZUJ1ZmZlcihnbCwgdGV4dHVyZSl7XG4gICAgdmFyIGZyYW1lYnVmZmVyID0gZ2wuY3JlYXRlRnJhbWVidWZmZXIoKTtcbiAgICBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIGZyYW1lYnVmZmVyKTtcbiAgICBnbC5mcmFtZWJ1ZmZlclRleHR1cmUyRChnbC5GUkFNRUJVRkZFUiwgZ2wuQ09MT1JfQVRUQUNITUVOVDAsIGdsLlRFWFRVUkVfMkQsIHRleHR1cmUsIDApO1xuICAgIHJldHVybiBmcmFtZWJ1ZmZlcjtcbn1cblxuXG5leHBvcnQgZnVuY3Rpb24gbWFrZVRleHR1cmUoZ2wpe1xuICAgIHZhciB0ZXh0dXJlID0gZ2wuY3JlYXRlVGV4dHVyZSgpO1xuICAgIGdsLmJpbmRUZXh0dXJlKGdsLlRFWFRVUkVfMkQsIHRleHR1cmUpO1xuICAgIGdsLnRleFBhcmFtZXRlcmkoZ2wuVEVYVFVSRV8yRCwgZ2wuVEVYVFVSRV9XUkFQX1MsIGdsLkNMQU1QX1RPX0VER0UpO1xuICAgIGdsLnRleFBhcmFtZXRlcmkoZ2wuVEVYVFVSRV8yRCwgZ2wuVEVYVFVSRV9XUkFQX1QsIGdsLkNMQU1QX1RPX0VER0UpO1xuICAgIGdsLnRleFBhcmFtZXRlcmkoZ2wuVEVYVFVSRV8yRCwgZ2wuVEVYVFVSRV9NSU5fRklMVEVSLCBnbC5ORUFSRVNUKTtcbiAgICBnbC50ZXhQYXJhbWV0ZXJpKGdsLlRFWFRVUkVfMkQsIGdsLlRFWFRVUkVfTUFHX0ZJTFRFUiwgZ2wuTkVBUkVTVCk7XG5cbiAgICByZXR1cm4gdGV4dHVyZTtcbn1cblxuIiwiaW1wb3J0IEJhc2VUZW5zb3IgZnJvbSAnLi9iYXNlLmpzJztcbmltcG9ydCBzaG93VGV4dHVyZSBmcm9tICcuL3Nob3cuanMnXG5pbXBvcnQgcnVuRmVhdHVyZVRlc3RzIGZyb20gJy4vZmVhdHVyZS5qcydcbmltcG9ydCB7IG1ha2VUZXh0dXJlLCBtYWtlRnJhbWVCdWZmZXIgfSBmcm9tICcuL2hlbHBlcnMuanMnXG5pbXBvcnQgeyBSdW4sIENvbXBpbGUgfSBmcm9tICcuLi9ydW50aW1lL2luZGV4LmpzJ1xuaW1wb3J0IG5kc2hvdyBmcm9tICduZGFycmF5LXNob3cnXG5pbXBvcnQgbmRhcnJheSBmcm9tICduZGFycmF5J1xuXG5leHBvcnQgY2xhc3MgVGVuc29yIGV4dGVuZHMgQmFzZVRlbnNvciB7XG4gICAgLy8gbmV3IFRlbnNvcihnbClcbiAgICAvLyBuZXcgVGVuc29yKGdsLCBbMSwgMV0pXG4gICAgLy8gbmV3IFRlbnNvcihnbCwgWzEsIDFdLCBudWxsKVxuICAgIC8vIG5ldyBUZW5zb3IoZ2wsIFsxLCAxXSwgZGF0YSlcbiAgICAvLyBuZXcgVGVuc29yKGdsLCBbMSwgMV0sIGRhdGEsIHsgdHlwZSwgcGFjaywgY29kZWMsIGRlbnNpdHkgfSlcbiAgICAvLyBuZXcgVGVuc29yKGdsLCBbMSwgMV0sIHsgdHlwZSwgcGFjaywgY29kZWMsIGRlbnNpdHkgfSlcbiAgICAvLyBuZXcgVGVuc29yKGdsLCBbMSwgMV0sICdzb2Z0ZmxvYXQnKVxuICAgIC8vIG5ldyBUZW5zb3IoZ2wsIFsxLCAxXSwgJ2Zsb2F0MzInKVxuICAgIC8vIG5ldyBUZW5zb3IoZ2wsIFsxLCAxXSwgJ3VpbnQ4JylcbiAgICAvLyBuZXcgVGVuc29yKGdsLCB7IHNoYXBlLCBkYXRhIH0pXG4gICAgLy8gbmV3IFRlbnNvcihnbCwgeyB3aWR0aCwgaGVpZ2h0LCBkYXRhIH0pXG4gICAgLy8gcGl4ID0gbmV3IFRlbnNvcihnbCwgWzEsIDEsIDRdLCBbMSwgMC40LCAzLCA0XSwgJ3VpbnQ4JylcblxuXHRjb25zdHJ1Y3RvcihnbCwgc2hhcGUgPSBbXSwgZGF0YSA9IG51bGwsIGZvcm1hdCA9IG51bGwpe1xuICAgICAgICBzdXBlcigpXG4gICAgICAgIHJ1bkZlYXR1cmVUZXN0cyhnbCk7XG5cbiAgICAgICAgdmFyIHhkYXRhID0gZGF0YTtcbiAgICAgICAgaWYoc2hhcGUuc2hhcGUpeyAvLyBuZGFycmF5c1xuICAgICAgICAgICAgZm9ybWF0ID0gZGF0YTtcbiAgICAgICAgICAgIHhkYXRhID0gc2hhcGUuZGF0YTtcbiAgICAgICAgICAgIGRhdGEgPSBzaGFwZTtcbiAgICAgICAgICAgIHNoYXBlID0gc2hhcGUuc2hhcGU7XG4gICAgICAgIH1cblxuICAgICAgICBpZihzaGFwZS53aWR0aCAmJiBzaGFwZS5oZWlnaHQgJiYgc2hhcGUuZGF0YSl7IC8vIGltYWdlZGF0YVxuICAgICAgICAgICAgZGF0YSA9IHNoYXBlLmRhdGE7XG4gICAgICAgICAgICBzaGFwZSA9IFtzaGFwZS53aWR0aCwgc2hhcGUuaGVpZ2h0XVxuICAgICAgICB9XG5cbiAgICAgICAgaWYodHlwZW9mIGRhdGEgPT09ICdzdHJpbmcnKXsgLy8gZGF0YSA9IHVpbnQ4IHwgZmxvYXQzMlxuICAgICAgICAgICAgaWYoZm9ybWF0ICE9PSBudWxsKVxuICAgICAgICAgICAgICAgIHRocm93IG5ldyBFcnJvcignRm9ybWF0IG11c3Qgbm90IGJlIHNwZWNpZmllZCBpZiBkYXRhIGlzIGEgc3RyaW5nLicpO1xuICAgICAgICAgICAgZm9ybWF0ID0gZGF0YTtcbiAgICAgICAgICAgIGRhdGEgPSBudWxsO1xuICAgICAgICB9ZWxzZSBpZihkYXRhICYmIHR5cGVvZiBkYXRhID09PSAnb2JqZWN0JyAmJiBkYXRhLnR5cGUgJiYgZGF0YS5jb2RlYyAmJiBkYXRhLnBhY2sgJiYgZGF0YS5kZW5zaXR5KXtcbiAgICAgICAgICAgIGlmKGZvcm1hdCAhPT0gbnVsbClcbiAgICAgICAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ0Zvcm1hdCBtdXN0IG5vdCBiZSBzcGVjaWZpZWQgaWYgZGF0YSBpcyBhbiBvYmplY3QuJyk7XG4gICAgICAgICAgICBmb3JtYXQgPSBkYXRhO1xuICAgICAgICAgICAgZGF0YSA9IG51bGw7XG4gICAgICAgIH1cblxuICAgICAgICBpZihmb3JtYXQgPT09IG51bGwpeyAvLyBhdXRvLWluZmVyIGZvcm1hdCBiYXNlZCBvbiBkYXRhXG4gICAgICAgICAgICBpZihkYXRhID09PSBudWxsKXtcbiAgICAgICAgICAgICAgICBmb3JtYXQgPSAnZmxvYXQzMidcbiAgICAgICAgICAgIH1lbHNlIGlmKHhkYXRhIGluc3RhbmNlb2YgVWludDhBcnJheSB8fCB4ZGF0YSBpbnN0YW5jZW9mIFVpbnQ4Q2xhbXBlZEFycmF5KXtcbiAgICAgICAgICAgICAgICBmb3JtYXQgPSAndWludDgnXG4gICAgICAgICAgICB9ZWxzZSBpZih4ZGF0YSBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSB8fCB4ZGF0YSBpbnN0YW5jZW9mIEZsb2F0NjRBcnJheSB8fCBBcnJheS5pc0FycmF5KHhkYXRhKSl7XG4gICAgICAgICAgICAgICAgZm9ybWF0ID0gJ2Zsb2F0MzInXG4gICAgICAgICAgICB9ZWxzZSB0aHJvdyBuZXcgRXJyb3IoXCJJbnZhbGlkIGZvcm1hdCBmb3IgZGF0YTogbXVzdCBiZSBVaW50OEFycmF5IG9yIEZsb2F0MzJBcnJheSBvciBuZGFycmF5XCIpO1xuICAgICAgICB9XG5cbiAgICAgICAgdmFyIHR5cGUgPSBudWxsO1xuICAgICAgICBpZigoZm9ybWF0ID09PSAnZmxvYXQzMicgJiYgXG4gICAgICAgICAgICAoZ2wuTk9fRkxPQVRfVEVYVFVSRVMgfHwgXG4gICAgICAgICAgICAoZ2wuTk9fUkVOREVSX0ZMT0FUICYmIHRoaXMgaW5zdGFuY2VvZiBPdXRwdXRUZW5zb3IpKSlcbiAgICAgICAgICAgIHx8IGZvcm1hdCA9PT0gJ3NvZnRmbG9hdCcpe1xuICAgICAgICAgICAgZm9ybWF0ID0geyB0eXBlOiAndWludDgnLCBwYWNrOiAnc3RyaWRlJywgZGVuc2l0eTogJzE6NCcsIGNvZGVjOiAnc29mdGZsb2F0JyB9XG4gICAgICAgICAgICB0eXBlID0gJ2Zsb2F0MzInXG4gICAgICAgIH1lbHNlIGlmKGZvcm1hdCA9PT0gJ3VpbnQ4JyB8fCBmb3JtYXQgPT09ICdmbG9hdDMyJyl7XG4gICAgICAgICAgICBmb3JtYXQgPSB7IHR5cGU6IGZvcm1hdCwgcGFjazogJ3N0cmlkZScsIGRlbnNpdHk6ICc0OjQnLCBjb2RlYzogJ3JhdycgfVxuICAgICAgICB9XG5cbiAgICAgICAgdGhpcy50eXBlID0gdHlwZSB8fCBmb3JtYXQudHlwZTtcbiAgICAgICAgdGhpcy5faW5pdChnbCwgZm9ybWF0LCBzaGFwZSwgZGF0YSk7XG5cdH1cblxuXG5cdGNvcHkoZm9ybWF0ID0gdGhpcy50eXBlLCBUID0gT3V0cHV0VGVuc29yKXtcbiAgICAgICAgY29uc3QgVEVOU09SX0lERU5USVRZID0gYFxuICAgICAgICAgICAgdW5pZm9ybSBUZW5zb3IgaW1hZ2U7XG4gICAgICAgICAgICB2ZWM0IHByb2Nlc3M0KGl2ZWM0IHBvcykgeyByZXR1cm4gaW1hZ2UucmVhZDQocG9zKTsgfVxuICAgICAgICBgO1xuICAgICAgICB2YXIgb3V0ID0gbmV3IFQodGhpcy5nbCwgdGhpcy5zaGFwZSwgZm9ybWF0KTtcbiAgICAgICAgb3V0LnJ1bihURU5TT1JfSURFTlRJVFksIHsgaW1hZ2U6IHRoaXMgfSlcbiAgICAgICAgcmV0dXJuIG91dFxuICAgIH1cblxuICAgIHdpdGhDb3B5KGZuLCAuLi5hcmdzKXtcbiAgICAgICAgdmFyIGNvcHkgPSB0aGlzLmNvcHkoLi4uYXJncyk7XG4gICAgICAgIHZhciByZXN1bHQgPSBmbihjb3B5KVxuICAgICAgICBjb3B5LmRlc3Ryb3koKVxuICAgICAgICByZXR1cm4gcmVzdWx0O1xuICAgIH1cblxuXHRfc2hvdyhvcHQgPSB7fSl7IHNob3dUZXh0dXJlKHRoaXMuZ2wsIHRoaXMudGV4LCBvcHQpIH1cbiAgICBzaG93KG9wdCA9IHt9KXtcbiAgICAgICAgdmFyIGdsID0gdGhpcy5nbDtcbiAgICAgICAgaWYodGhpcy5mb3JtYXQucGFjayA9PSAndGlsZScgXG4gICAgICAgICAgICAmJiB0aGlzLmZvcm1hdC5kZW5zaXR5ID09ICc0OjQnIFxuICAgICAgICAgICAgJiYgdGhpcy5mb3JtYXQuY29kZWMgPT0gJ3Jhdycpe1xuICAgICAgICAgICAgdGhpcy5fc2hvdyhvcHQpXG4gICAgICAgIH1lbHNle1xuICAgICAgICAgICAgLy8gQy5pbmZvLm1haW5faW5wdXQub3V0cHV0LmNvcHkoeyB0eXBlOiAndWludDgnLCBwYWNrOiAndGlsZScsIGRlbnNpdHk6ICc0OjQnLCBjb2RlYzogJ2xpbnF1YW50JywgbWluOiAwLCBtYXg6IDI1NSB9KS5fc2hvdyh7IH0pXG4gICAgICAgICAgICB0aGlzLndpdGhDb3B5KHggPT4geC5zaG93KG9wdCksIFxuICAgICAgICAgICAgICAgIHsgdHlwZTogXG4gICAgICAgICAgICAgICAgICAgIChnbC5OT19GTE9BVF9URVhUVVJFUyB8fCBnbC5OT19SRU5ERVJfRkxPQVQpID8gJ3VpbnQ4JyA6ICdmbG9hdDMyJywgXG4gICAgICAgICAgICAgICAgICAgIHBhY2s6ICd0aWxlJywgZGVuc2l0eTogJzQ6NCcsIGNvZGVjOiAncmF3JyB9KVxuICAgICAgICB9O1xuICAgIH1cblxuICAgIHJ1bihzaGFkZXIsIHBhcmFtcyl7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcignT25seSBPdXRwdXRUZW5zb3IgY2FuIHJ1biBzaGFkZXJzLicpXG4gICAgfVxuICAgIGNvbXBpbGUoc2hhZGVyLCBwYXJhbXMpe1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ09ubHkgT3V0cHV0VGVuc29yIGNhbiBjb21waWxlIHNoYWRlcnMuJylcbiAgICB9XG4gICAgcmVhZCgpe1xuICAgICAgICBjb25zb2xlLndhcm4oXCJDb3B5aW5nIGJlZm9yZSByZWFkLi4uXCIpXG4gICAgICAgIHJldHVybiB0aGlzLndpdGhDb3B5KHggPT4geC5yZWFkKCkpXG4gICAgfVxuICAgIHByaW50KCl7XG4gICAgICAgIHJldHVybiBuZHNob3codGhpcy5yZWFkKCkpXG4gICAgfVxuICAgIHN3YXAoKXtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFwiT25seSBJblBsYWNlVGVuc29yIGNhbiBiZSBib3RoIGEgcGFyYW1ldGVyIGFuZCBkZXN0aW5hdGlvbi5cIik7XG4gICAgfVxufVxuXG5leHBvcnQgY2xhc3MgT3V0cHV0VGVuc29yIGV4dGVuZHMgVGVuc29yIHtcblx0Y29uc3RydWN0b3IoLi4uYXJncyl7XG4gICAgICAgIHN1cGVyKC4uLmFyZ3MpO1xuXHRcdHRoaXMuZmJvID0gbWFrZUZyYW1lQnVmZmVyKHRoaXMuZ2wsIHRoaXMudGV4KTtcblx0fVxuXG4gICAgZGVzdHJveSgpe1xuICAgICAgICBzdXBlci5kZXN0cm95KClcbiAgICAgICAgdGhpcy5nbC5kZWxldGVGcmFtZWJ1ZmZlcih0aGlzLmZibylcbiAgICB9XG5cbiAgICBfcmVhZCgpe1xuICAgICAgICB2YXIgZ2wgPSB0aGlzLmdsLFxuICAgICAgICAgICAgc2l6ZSA9IHRoaXMuaW5mby50ZXhTaXplO1xuXG4gICAgICAgIGlmKHRoaXMuZm9ybWF0LnR5cGUgPT0gJ3VpbnQ4Jyl7XG4gICAgICAgICAgICB2YXIgZ2xUeXBlID0gZ2wuVU5TSUdORURfQllURSxcbiAgICAgICAgICAgICAgICBwaXhlbHMgPSBuZXcgVWludDhBcnJheShzaXplWzBdICogc2l6ZVsxXSAqIDQpXG4gICAgICAgIH1lbHNlIGlmKHRoaXMuZm9ybWF0LnR5cGUgPT09ICdmbG9hdDMyJyl7XG4gICAgICAgICAgICB2YXIgZ2xUeXBlID0gZ2wuRkxPQVQsXG4gICAgICAgICAgICAgICAgcGl4ZWxzID0gbmV3IEZsb2F0MzJBcnJheShzaXplWzBdICogc2l6ZVsxXSAqIDQpXG4gICAgICAgIH1cblxuICAgICAgICBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIHRoaXMuZmJvKTtcbiAgICAgICAgZ2wucmVhZFBpeGVscygwLCAwLCBzaXplWzBdLCBzaXplWzFdLCBnbC5SR0JBLCBnbFR5cGUsIHBpeGVscyk7XG5cbiAgICAgICAgLy8gY29uc29sZS5sb2coJ19fX3JlYWQnLCBwaXhlbHMpXG4gICAgICAgIHJldHVybiBwaXhlbHM7XG4gICAgfVxuXG4gICAgcnVuKHNoYWRlciwgcGFyYW1zLCBjYWxsYmFjayl7XG4gICAgICAgIHJldHVybiBSdW4oc2hhZGVyLCB0aGlzLCBwYXJhbXMsIGNhbGxiYWNrKTtcbiAgICB9XG4gICAgY29tcGlsZShzaGFkZXIsIHBhcmFtcyl7XG4gICAgICAgIHJldHVybiBDb21waWxlKHNoYWRlciwgdGhpcywgcGFyYW1zKTtcbiAgICB9XG5cblx0cmVhZCgpe1xuICAgICAgICBpZih0aGlzLmZvcm1hdC50eXBlID09PSAnZmxvYXQzMicgJiYgdGhpcy5nbC5OT19SRUFEX0ZMT0FUKXtcbiAgICAgICAgICAgIHJldHVybiB0aGlzLndpdGhDb3B5KHggPT4geC5yZWFkKCksICdzb2Z0ZmxvYXQnKVxuICAgICAgICB9XG5cblx0XHR2YXIgYXJyYXkgPSB0aGlzLl9mb3JtYXQucGFjay51bnBhY2sodGhpcy5pbmZvLCB0aGlzLl9yZWFkKCksIHRoaXMuX2Zvcm1hdC5jb2RlYy5kZWNvZGUsIHRoaXMudHlwZSk7XG4gICAgICAgIFxuICAgICAgICAvLyBzdHJpcCB0cmFpbGluZyBzaW5nbGV0b24gZGltZW5zaW9uc1xuICAgICAgICB2YXIgc2hhcGUgPSBhcnJheS5zaGFwZS5zbGljZSgwKSxcbiAgICAgICAgICAgIHN0cmlkZSA9IGFycmF5LnN0cmlkZS5zbGljZSgwKTtcbiAgICAgICAgd2hpbGUoc2hhcGVbc2hhcGUubGVuZ3RoIC0gMV0gPT0gMSAmJiBzaGFwZS5sZW5ndGggPiAxKXtcbiAgICAgICAgICAgIHNoYXBlLnBvcCgpXG4gICAgICAgICAgICBzdHJpZGUucG9wKClcbiAgICAgICAgfVxuICAgICAgICByZXR1cm4gbmRhcnJheShhcnJheS5kYXRhLCBzaGFwZSwgc3RyaWRlLCBhcnJheS5vZmZzZXQpO1xuXHR9XG59XG5cbmV4cG9ydCBjbGFzcyBJblBsYWNlVGVuc29yIGV4dGVuZHMgT3V0cHV0VGVuc29yIHtcblx0Y29uc3RydWN0b3IoLi4uYXJncyl7XG5cdFx0c3VwZXIoLi4uYXJncylcblxuICAgICAgICB0aGlzLnRleDIgPSB0aGlzLnRleDtcbiAgICAgICAgdGhpcy50ZXggPSBtYWtlVGV4dHVyZSh0aGlzLmdsKTtcblx0XHR0aGlzLnVwZGF0ZShudWxsKTtcbiAgICAgICAgdGhpcy5zd2FwKClcblx0fVxuICAgIGRlc3Ryb3koKXtcbiAgICAgICAgc3VwZXIuZGVzdHJveSgpXG4gICAgICAgIHRoaXMuZ2wuZGVsZXRlVGV4dHVyZSh0aGlzLnRleDIpXG4gICAgfVxuICAgIHN3YXAoKXtcbiAgICAgICAgdmFyIHRtcCA9IHRoaXMudGV4O1xuICAgICAgICB0aGlzLnRleCA9IHRoaXMudGV4MjtcbiAgICAgICAgdGhpcy50ZXgyID0gdG1wO1xuXG4gICAgICAgIC8vIFRPRE86IGludmVzdGlnYXRlIHBlcmZvcm1hbmNlIG9mIHVzaW5nIG11bHRpcGxlIEZCT3MgaW5zdGVhZFxuICAgICAgICAvLyBvZiByZWJpbmRpbmcgdGhlIGZyYW1lYnVmZmVyXG4gICAgICAgIHZhciBnbCA9IHRoaXMuZ2w7XG4gICAgICAgIGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgdGhpcy5mYm8pO1xuICAgICAgICBnbC5mcmFtZWJ1ZmZlclRleHR1cmUyRChnbC5GUkFNRUJVRkZFUiwgZ2wuQ09MT1JfQVRUQUNITUVOVDAsIGdsLlRFWFRVUkVfMkQsIHRoaXMudGV4LCAwKTtcbiAgICB9XG59IiwiaW1wb3J0IHsgYmluZEF0dHJpYnV0ZUJ1ZmZlciwgY3JlYXRlU2hhZGVyUHJvZ3JhbSB9IGZyb20gJy4uL3J1bnRpbWUvcHJvZ3JhbS5qcydcblxuY29uc3QgU0hPV19URVhUVVJFX1ZFUlRFWCA9IGBcbiAgICBhdHRyaWJ1dGUgdmVjMiBhX3Bvc2l0aW9uO1xuICAgIHZhcnlpbmcgbWVkaXVtcCB2ZWMyIHBvcztcbiAgICB2b2lkIG1haW4oKSB7XG4gICAgICAgIHBvcyA9IChhX3Bvc2l0aW9uICsgdmVjMigxLCAxKSkgLyAyLjA7XG4gICAgICAgIGdsX1Bvc2l0aW9uID0gdmVjNChhX3Bvc2l0aW9uLCAwLCAxKTtcbiAgICB9XG5gXG5cbmNvbnN0IFNIT1dfVEVYVFVSRV9GUkFHTUVOVCA9IGBcbiAgICBwcmVjaXNpb24gbWVkaXVtcCBmbG9hdDtcblxuICAgIHVuaWZvcm0gc2FtcGxlcjJEIHRleDtcbiAgICB1bmlmb3JtIGZsb2F0IHNjYWxlO1xuICAgIHVuaWZvcm0gZmxvYXQgb2Zmc2V0O1xuICAgIHVuaWZvcm0gYm9vbCB0cmFuc3Bvc2U7XG4gICAgdW5pZm9ybSBib29sIGZsaXBYO1xuICAgIHVuaWZvcm0gYm9vbCBmbGlwWTtcbiAgICB1bmlmb3JtIGludCBjaGFubmVscztcblxuICAgIHZhcnlpbmcgdmVjMiBwb3M7XG5cbiAgICB2ZWM0IGNvbG9ybWFwKGZsb2F0IHgpIHtcbiAgICAgICAgZmxvYXQgciA9IGNsYW1wKDguMCAvIDMuMCAqIHgsIDAuMCwgMS4wKTtcbiAgICAgICAgZmxvYXQgZyA9IGNsYW1wKDguMCAvIDMuMCAqIHggLSAxLjAsIDAuMCwgMS4wKTtcbiAgICAgICAgZmxvYXQgYiA9IGNsYW1wKDQuMCAqIHggLSAzLjAsIDAuMCwgMS4wKTtcbiAgICAgICAgcmV0dXJuIHZlYzQociwgZywgYiwgMS4wKTtcbiAgICB9XG5cbiAgICB2b2lkIG1haW4oKSB7XG4gICAgICAgIHZlYzIgcCA9IHBvcztcbiAgICAgICAgaWYoZmxpcFgpIHAueCA9IDEuMCAtIHAueDtcbiAgICAgICAgaWYoZmxpcFkpIHAueSA9IDEuMCAtIHAueTtcbiAgICAgICAgaWYodHJhbnNwb3NlKSBwID0gcC55eDtcbiAgICAgICAgaWYoY2hhbm5lbHMgPT0gNCl7XG4gICAgICAgICAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KHZlYzQob2Zmc2V0LCBvZmZzZXQsIG9mZnNldCwgb2Zmc2V0KSBcbiAgICAgICAgICAgICAgICArIHNjYWxlICogdGV4dHVyZTJEKHRleCwgcCkpO1xuICAgICAgICB9ZWxzZSBpZihjaGFubmVscyA9PSAzKXtcbiAgICAgICAgICAgIGdsX0ZyYWdDb2xvciA9IHZlYzQodmVjMyhvZmZzZXQsIG9mZnNldCwgb2Zmc2V0KSBcbiAgICAgICAgICAgICAgICArIHNjYWxlICogdGV4dHVyZTJEKHRleCwgcCkucmdiLCAxKTtcbiAgICAgICAgfWVsc2UgaWYoY2hhbm5lbHMgPT0gMil7XG4gICAgICAgICAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KHZlYzIob2Zmc2V0LCBvZmZzZXQpIFxuICAgICAgICAgICAgICAgICsgc2NhbGUgKiB0ZXh0dXJlMkQodGV4LCBwKS5yZywgMCwgMSk7XG4gICAgICAgIH1lbHNlIGlmKGNoYW5uZWxzID09IDEpe1xuICAgICAgICAgICAgZ2xfRnJhZ0NvbG9yID0gY29sb3JtYXAob2Zmc2V0ICsgc2NhbGUgKiB0ZXh0dXJlMkQodGV4LCBwKS5yKTtcbiAgICAgICAgfVxuICAgIH1cbmBcblxuZXhwb3J0IGRlZmF1bHQgZnVuY3Rpb24gc2hvd1RleHR1cmUoZ2wsIHRleCwgb3B0ID0ge30pe1xuICAgIGlmKCFnbC5fc2hvd1Byb2dyYW0pe1xuICAgICAgICBnbC5fc2hvd1Byb2dyYW0gPSBjcmVhdGVTaGFkZXJQcm9ncmFtKGdsLCBTSE9XX1RFWFRVUkVfVkVSVEVYLCBTSE9XX1RFWFRVUkVfRlJBR01FTlQpO1xuICAgICAgICBnbC51c2VQcm9ncmFtKGdsLl9zaG93UHJvZ3JhbSk7XG4gICAgICAgIGJpbmRBdHRyaWJ1dGVCdWZmZXIoZ2wsIGdsLl9zaG93UHJvZ3JhbSk7XG4gICAgICAgIGdsLnVuaWZvcm0xaShnbC5nZXRVbmlmb3JtTG9jYXRpb24oZ2wuX3Nob3dQcm9ncmFtLCAndGV4JyksIDApO1xuICAgIH1cbiAgICBcblxuICAgIGlmKGdsLmNhbnZhcyAmJiBnbC5jYW52YXMuX3RmQXV0byl7XG4gICAgICAgIGdsLmNhbnZhcy5zdHlsZS5kaXNwbGF5ID0gJ2Jsb2NrJ1xuICAgICAgICBnbC5jYW52YXMuc3R5bGUucG9zaXRpb24gPSAnYWJzb2x1dGUnXG4gICAgICAgIGdsLmNhbnZhcy5zdHlsZS50b3AgPSAwO1xuICAgICAgICBnbC5jYW52YXMuc3R5bGUubGVmdCA9IDA7XG4gICAgICAgIGdsLmNhbnZhcy5zdHlsZS53aWR0aCA9IE1hdGgubWluKGlubmVySGVpZ2h0LCBpbm5lcldpZHRoKSArICdweCdcbiAgICAgICAgZ2wuY2FudmFzLnN0eWxlLmhlaWdodCA9IE1hdGgubWluKGlubmVySGVpZ2h0LCBpbm5lcldpZHRoKSArICdweCdcbiAgICB9XG5cbiAgICBnbC51c2VQcm9ncmFtKGdsLl9zaG93UHJvZ3JhbSk7XG4gICAgZ2wuYWN0aXZlVGV4dHVyZShnbC5URVhUVVJFMCk7XG4gICAgZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGV4KTtcbiAgICBnbC51bmlmb3JtMWYoZ2wuZ2V0VW5pZm9ybUxvY2F0aW9uKGdsLl9zaG93UHJvZ3JhbSwgJ3NjYWxlJyksIG9wdC5zY2FsZSB8fCAxKVxuICAgIGdsLnVuaWZvcm0xZihnbC5nZXRVbmlmb3JtTG9jYXRpb24oZ2wuX3Nob3dQcm9ncmFtLCAnb2Zmc2V0JyksIG9wdC5vZmZzZXQgfHwgMClcbiAgICBnbC51bmlmb3JtMWkoZ2wuZ2V0VW5pZm9ybUxvY2F0aW9uKGdsLl9zaG93UHJvZ3JhbSwgJ3RyYW5zcG9zZScpLCBvcHQudHJhbnNwb3NlID8gMSA6IDApXG4gICAgZ2wudW5pZm9ybTFpKGdsLmdldFVuaWZvcm1Mb2NhdGlvbihnbC5fc2hvd1Byb2dyYW0sICdmbGlwWCcpLCBvcHQuZmxpcFggPyAxIDogMClcbiAgICBnbC51bmlmb3JtMWkoZ2wuZ2V0VW5pZm9ybUxvY2F0aW9uKGdsLl9zaG93UHJvZ3JhbSwgJ2ZsaXBZJyksIG9wdC5mbGlwWSA/IDEgOiAwKVxuICAgIGdsLnVuaWZvcm0xaShnbC5nZXRVbmlmb3JtTG9jYXRpb24oZ2wuX3Nob3dQcm9ncmFtLCAnY2hhbm5lbHMnKSwgb3B0LmNoYW5uZWxzIHx8IDMpO1xuXG4gICAgZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBudWxsKTtcbiAgICBnbC52aWV3cG9ydCgwLCAwLCBnbC5kcmF3aW5nQnVmZmVyV2lkdGgsIGdsLmRyYXdpbmdCdWZmZXJIZWlnaHQpO1xuICAgIGdsLmRyYXdBcnJheXMoZ2wuVFJJQU5HTEVfU1RSSVAsIDAsIDQpO1xuXG59XG4iLCJleHBvcnQgZnVuY3Rpb24gY3JlYXRlR0woY2FudmFzKXtcbiAgICBpZighY2FudmFzKXtcbiAgICAgICAgY2FudmFzID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnY2FudmFzJyk7XG4gICAgICAgIGNhbnZhcy53aWR0aCA9IDUxMlxuICAgICAgICBjYW52YXMuaGVpZ2h0ID0gNTEyXG4gICAgICAgIGNhbnZhcy5zdHlsZS5kaXNwbGF5ID0gJ25vbmUnO1xuICAgICAgICBjYW52YXMuX3RmQXV0byA9IHRydWU7XG4gICAgICAgIGRvY3VtZW50LmJvZHkuYXBwZW5kQ2hpbGQoY2FudmFzKVxuICAgIH1cbiAgICB2YXIgZ2wgPSBjYW52YXMuZ2V0Q29udGV4dChcIndlYmdsXCIsIHsgYW50aWFsaWFzOiBmYWxzZSB9KSBcbiAgICAgICAgICB8fCBjYW52YXMuZ2V0Q29udGV4dChcImV4cGVyaW1lbnRhbC13ZWJnbFwiLCB7IGFudGlhbGlhczogZmFsc2UgfSk7XG4gICAgaWYgKCFnbCkgYWxlcnQoJ0NvdWxkIG5vdCBpbml0aWFsaXplIFdlYkdMLCB0cnkgYW5vdGhlciBicm93c2VyJyk7XG4gICAgcmV0dXJuIGdsO1xufVxuIl19
