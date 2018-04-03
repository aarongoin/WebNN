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
		this.getSize();
		console.log('model size: ' + this.size);

		this.mergeLayer = tf.layers.average();
		if (weights) this.load(weights);
	}

	getSize() {
		let size = 0;
		this.sizes = [];
		for (let layer of this.model.layers) { // LayerVarable
			for (let weight of layer.weights) {
				let s = weight.shape.reduce((s, n) => s !== null ? s * n : s, 1);
				this.sizes.push(s);
				size += s;
			}
		}
		this.size = size;
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
		// console.log(in_shape);
		// console.log(input);
		const in_tensor = tf.tensor(input, in_shape);
		const out_tensor = tf.tensor(expect, [batch_size, 10]); // TODO: replace magic value (10) here

		let result = this.model.evaluate(in_tensor, out_tensor, { batchSize: batch_size });
		// result = [ loss_tensor, accuracy_tensor ]

		result[1].data().then((accuracy) => {
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
		// TypedArray to hold weights, bias, etc. from every layer of model
		let weights = new Float32Array(this.size);
		let o = 0;
		let i = 0;
		let p = []; // array to hold promises for each set of weights
		for (let layer of this.model.layers) { // LayerVarable
			for (let weight of layer.weights) {
				p.push(
					weight.val.data().then((w) => {
						weights.set(w, o);
						o += this.sizes[i++];
					})
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
		this.currentWeights = weights;
		let o = 0;
		let i = 0;
		// console.log(weights);
		for (let layer of this.model.layers) { // LayerVarable
			for (let weight of layer.weights) {
				let t = tf.tensor( weights.subarray(o, o + this.sizes[i]), weight.shape );
				o += this.sizes[i++];
				weight.val.assign(t);
			}
		}
	}

	/**
	 * Merge weights into current set
	 * @param {*} weights - Weights to merge into current model weights
	 * @param {number} scale - Determines how weights are merged
	 */
	merge(weights, callback) {
		let temp = tf.tensor(weights, [weights.length, 1, 1, 1]);
		let current = tf.tensor(currentWeights, [weights.length, 1, 1, 1]);
		let merged = this.mergeLayer.apply([current, weights]);
		merged.data().then(data => {
			this.load(data);
			if (typeof callback === 'function') callback();
		})
	}

	readDataBatch(batch) {
		const n = new Float32Array(batch, 0, 1)[0]; // num of samples
		// length is num_samples * num_input nodes
		// We have to really dig to get the num_input nodes
		// TODO: add convenience methods or find a better way
		const length = n * this.model.getInputAt(1).shape.reduce((r, v) => v !== null ? r * v : r, 1);
		const x = new Float32Array(batch, 4, length);
		const y = new Uint8Array(batch, (length * 4) + 4);
		return { n, x, y };
	}
}
module.exports = Model;
