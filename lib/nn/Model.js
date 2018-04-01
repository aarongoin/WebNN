const
Layers = require('./Layers'),
Input = require('./layers/Input'),
Output = require('./layers/Output');

/**
 * This is the Model class
 * @example
 * let myModel = new Model({
 * 	layers: [
 * 		{ input: 30 },
 * 		{ dense: 'tanh', bias: true, out: 30 }
 * 		{ output: 3, loss: 'mean_squared_error' }
 * 	]
 * });
 */
class Model {
	/**
	 * Creates a new model ready to execute and train on input.
	 * @param {Object[]} model - Array of objects describing model layers
	 * @param {?(Float32Array|ArrayBuffer)} weights - Weights for all layers
	 */
	constructor(model, weights, learning_rate) {
		// /* this.input = new Input(model.shift());
		// this.inputs = this.input.layer.input;
		// this.output = new Output(model.pop());
		// this.outputs = this.output.layer.output;
		// /**
		//  * Hidden layers of the model. The input layer is at index: 0, and the output layer is at index: layers.length - 1
		//  * @type {Object[]}
		//  * */
		// this.hidden = new Array(model.length);
		// /**
		//  * Number of weights and biases in entire model.
		//  * @type {number}
		//  * */
		// this.size = 0;
		// /**
		//  * The model description
		//  * @type {Object}
		//  */
		// this.model = model;
		//
		// this.load(weights);

		this.model = tf.sequential();

		for (let layer of model) {
			let key = Object.keys(layer)[0];
			if (key !== 'output') {
				this.model.add(tf.layers[key](layer[key]));
			}
			else {
				layer.output.optimizer = tf.train[layer.output.optimizer](learning_rate);
				this.model.compile(layer.output);
			}
		}

	}
	/**
	 * Execute forward pass of input through model.
	 * @param {Float32Array|Tensor} input - Features input to model
	 * @returns {Tensor} Tensor containing model output
	 */
	forward(input) {
		// forward propogation
		var output = input;
		// var output = this.input.forward(input);
		var l = -1;
		for (var layer of this.hidden)
			output = layer.forward(output);

		return this.output.forward(output);
	}
	/**
	 * Execute backward pass of error through model.
	 * @param {Tensor} output - Error from output
	 */
	backward(output, learn) {
		// backward propogation
		var l = this.hidden.length;
		while (l--)
			output = this.hidden[l].backward(output, learn);
	}

	/**
	 * Validate model and pass output accuracy as argument to callback.
	 * @param {Float32Array|Tensor} input - Features input to model
	 * @param {Float32Array|Tensor} expect - Expected model output
	 * @param {?validtionCallback} callback
	 */
	validate(input, expect, callback) {
		this.forward(input);
		this.output.backward(expect);

		if (typeof callback === "function") callback(this.output.accuracy)
	}
	/**
	 * Callback for validation
	 * @callback validtionCallback
	 * @param {number} accuracy
	 */

	/**
	 * Train model and pass new weights and model accuracy as arguments to callback.
	 * @param {Float32Array|Tensor} input - Features input to model
	 * @param {Float32Array|Tensor} expect - Expected model output
	 * @param {?trainingCallback} callback
	 */
	train(input, expect, learn, callback) {
		this.forward(input);
		// console.log('expected:');
		// console.log(expect);
		// console.log('output:');
		// console.log(this.output.outputTensor.read().data);
		this.backward(this.output.backward(expect), learn);
		// console.log('gradient:');
		// console.log(this.output.outputTensor.read().data);


		if (typeof callback === "function") callback(this.save(false), this.output.accuracy);
	}
	/**
	 * Callback for training
	 * @callback trainingCallback
	 * @param {Float32Array} weights - Updated weights from model
	 * @param {number} accuracy - Accuracy of model's output
	 */

	/**
	 * Saves model weights.
	 * @returns {ArrayBuffer} ArrayBuffer containing weights and biases from all layers in model
	 */
	save(asBuffer) {
		// TypedArray to hold weights, bias, etc. from every layer of model
		var weights = new Float32Array(this.size);
		var o = 0;
		// pull out trained weights for each layer
		for (var layer of this.hidden) {
			weights.set(layer.save(), o);
			o += layer.size;
		}
		return asBuffer ? weights.buffer : weights;
	}

	/**
	 * Loads weights into model, or generates new ones if needed.
	 * @param {?(Float32Array|ArrayBuffer)} weights - weights to load into model, if null then weights will be generated
	 */
	load(weights) {
		// construct layers
		let offset = 0;
		let prev = this.inputs; // number of input nodes

		this.size = 0;
		if (weights != null && weights.buffer === undefined) {
			weights = new Float32Array(weights);
		}
		for (let i in this.model) {
			let layer = null;
			if (this.model[i].dense) {
				layer = new Layers['dense'](this.model[i], prev);
			}

			prev = layer.outputs; // hold number of output nodes from this layer as inputs for next layer
			this.size += layer.size;

			if (weights != null)
				offset = layer.load(weights, offset);
			else layer.randomWeights();
			this.hidden[i] = layer;
		}
	}

	/**
	 * Merge weights into current set
	 * @param {*} weights - Weights to merge into current model weights
	 * @param {number} scale - Determines how weights are merged
	 */
	merge(weights) {
		var offset = 0;
		for (var layer of this.hidden) {
			layer.merge(weights, offset);
		}
	}
}
module.exports = Model;
