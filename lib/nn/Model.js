// const
// Layers = require('./Layers'),
// Input = require('./layers/Input'),
// Output = require('./layers/Output');
const tf = require('@tensorflow/tfjs');

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

		if (weights) this.load(weights);

	}


	/**
	 * Validate model and pass output accuracy as argument to callback.
	 * @param {Float32Array|Tensor} input - Features input to model
	 * @param {Float32Array|Tensor} expect - Expected model output
	 * @param {?validtionCallback} callback
	 */
	validate(input, expect, callback) {

		let result = this.model.evaluate(input, expect); // scalar or list of scalars
		// loss + optional metrics (like accuracy)
		if (typeof callback === "function") callback(result);

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
	train(input, expect, callback) {
		let result = this.model.fit(input, expect, {batchsize: this.batch_size});
		if (typeof callback === "function") callback(this.save(), result);
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
	save() {
		// TypedArray to hold weights, bias, etc. from every layer of model
		var weights = [];
		for (let layer of this.model.layers) { // LayerVarable
			weights.push(layer.weights.read());
		}
		return weights;
	}

	/**
	 * Loads weights into model, or generates new ones if needed.
	 * @param {Tensor} weights - weights to load into model, if null then weights will be generated
	 */
	load(weights) {
		for (let layer of this.model.layers) { // LayerVarable
			layer.weights.write(weights);
		}
		//.read()
		//.write(LayerVariable)
	}

	/**
	 * Merge weights into current set
	 * @param {*} weights - Weights to merge into current model weights
	 * @param {number} scale - Determines how weights are merged
	 */
	merge(weights) {
		// var offset = 0;
		// for (var layer of this.hidden) {
		// 	layer.merge(weights, offset);
		// }
	}
}
module.exports = Model;
