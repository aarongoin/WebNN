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
		this.description = model;

		for (let layer of model) {
			let key = Object.keys(layer)[0];
			if (key !== 'output') {
				this.model.add(tf.layers[key](layer[key]));
			}
			else {
				let optimizer = Object.assign({},
					layer.output,
					{ optimizer: tf.train[layer.output.optimizer](learning_rate) }
				);
				this.model.compile(optimizer);
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
	validate(input, expect, batch_size, callback) {

		// console.log(this.model.outp.outputLayers[0]);
		let in_shape = Object.assign([], this.model.layers[0].batchInputShape, { 0: batch_size });
		const in_tensor = tf.tensor(input, in_shape);
		const out_tensor = tf.tensor(expect, [batch_size, 10]); // TODO: replace magic value (10) here

		let result = this.model.evaluate(in_tensor, out_tensor, { batchSize: batch_size });
		// result = [ loss_tensor, accuracy_tensor ]

		result[1].data().then((accuracy) => {
			console.log('accuracy: ' + accuracy);
			if (typeof callback === "function") callback(accuracy);
		});
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
	train(input, expect, batch_size, callback) {
		let in_shape = Object.assign([], this.model.layers[0].batchInputShape, { 0: batch_size });
		const in_tensor = tf.tensor(input, in_shape);
		const out_tensor = tf.tensor(expect, [batch_size, 10]); // TODO: replace magic value (10) here

		this.model.fit(in_tensor, out_tensor, {batchsize: batch_size, epochs: 1 }).then((history) => {
			if (typeof callback === "function")
				this.save((weights) => callback(weights, history.history.loss[0]) );
				
		});
	}

	/**
	 * Callback for training
	 * @callback trainingCallback
	 * @param {Float32Array} weights - Updated weights from model
	 * @param {number} accuracy - Accuracy of model's output
	 */

	/**
	 * Saves model weights.
	 * @returns {Array} ArrayBuffer containing weights and biases from all layers in model
	 */
	save(callback) {
		// Array to hold weights, bias, etc. from every layer of model
		var weights = [];
		var p = [];
		for (let layer of this.model.layers) { // LayerVarable
			for (let weight of layer.weights) {
				p.push(
					weight.val.data().then((w) => weights.push(w))
				);
			}
		}
		Promise.all(p).then(() => {
			callback(weights);
		});
	}

	/**
	 * Loads weights into model, or generates new ones if needed.
	 * @param {Tensor} weights - weights to load into model, if null then weights will be generated
	 */
	load(weights) {
		let i = 0;
		// console.log(weights);
		for (let layer of this.model.layers) { // LayerVarable
			for (let weight of layer.weights) {
				// console.log(weights[i]);
				let t = tf.tensor(Object.assign([], weights[i++]), weight.shape);
				weight.val.assign(
					t
				);
			}
		}
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
