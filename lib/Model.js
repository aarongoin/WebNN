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
	constructor(model, weights, steps, learning_rate) {

		this.model = tf.sequential();
		this.description = model;
		this.lr = learning_rate;

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
		// console.log('model size: ' + this.size);

		this.mergeLayer = tf.layers.average();
		if (weights) this.load(weights);
		this.steps = steps;

		this.updateLearningRate = this.updateLearningRate.bind(this);
	}

	getSize() {
		let size = 0;
		this.sizes = [];
		for (let layer of this.model.layers) { // LayerVariable
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
	 * @param {Float32Array} inputs - Features input to model
	 * @param {Float32Array} expect - Expected model output
	 * @param {?validationCallback} callback
	 */
	validate(inputs, expect, batch_size, callback) {

		let in_shape = Object.assign([], this.model.layers[0].batchInputShape, { 0: batch_size });
		// console.log(in_shape);
		// console.log(input);
		const in_tensor = tf.tensor(inputs, in_shape);
		const out_tensor = tf.tensor(expect, [batch_size, 10]); // TODO: replace magic value (10) here

		let result = this.model.evaluate(in_tensor, out_tensor, { batchSize: batch_size });

		const read = [
			result[0].data(), // loss
			result[1].data()  // accuracy
		];
		Promise.all(read).then((results) => {
			console.log(results);
			in_tensor.dispose();
			out_tensor.dispose();
			if (typeof callback === "function") callback(results[0][0], results[1][0]);
		});
	}
	/**
	 * Callback for validation
	 * @callback validationCallback
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
			in_tensor.dispose();
			out_tensor.dispose();
			this.accuracy = history.history.acc[0];
			if (typeof callback === "function")
				this.save((weights) => callback(weights, history.history.loss[0], history.history.acc[0]) );
				
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
	load(weights) { tf.tidy(() => {
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
	})}

	/**
	 * Merge weights into current set
	 * @param {*} weights - Weights to merge into current model weights
	 * @param {function?} callback - Callback for triggering an action, passed no arguments
	 */
	merge(weights, steps, accuracy, callback) {
		let temp = tf.tensor(weights, [weights.length]);
		let current = tf.tensor(this.currentWeights, [weights.length]);
		let merged = this.mergeLayer.apply([current, temp]);
		merged.data().then(data => {
			// cleanup tensors
			temp.dispose();
			current.dispose();
			merged.dispose();

			this.load(data);
			if (typeof callback === 'function') callback();
		})
	}

	/**
	 * Merge weights into current set
	 * @param {*} weights - Weights to merge into current model weights
	 * @param {function?} callback - Callback for triggering an action, passed no arguments
	 */
	copyMerge(weights, steps, accuracy, callback) {
		let temp = tf.tensor(weights, [weights.length]);
		let current = tf.tensor(this.currentWeights, [weights.length]);
		let merged = this.mergeLayer.apply([current, temp]);

		if (accuracy > this.accuracy) {
			this.load(weights);
			if (typeof callback === 'function') callback();
		}
		
	}

	/**
	 * Merge weights into current set weighted by number of training steps each weight has trained
	 * @param {*} weights - Weights to merge into current model weights
	 * @param {number} steps - Number of times weights have been trained by a client
	 * @param {function?} callback - Callback for triggering an action, passed no arguments
	 */
	weightedMerge(weights, steps, accuracy, callback) {
		let temp = tf.tensor(weights, [weights.length]);
		let current = tf.tensor(this.currentWeights, [weights.length]);

		// weight each weight by how many times it's been trained
		let scaledTemp = temp.mul(tf.scalar(steps));
		let scaledCurrent = current.mul(tf.scalar(this.steps));

		// add both sets of weighted weights together
		let sum = tf.add(scaledTemp, scaledCurrent);

		// divide sum by number of training steps
		let stepDiv = tf.scalar(1 / (steps + this.steps)); // do division once on cpu
		let merged = tf.mul(sum, stepDiv);
		
		if (this.steps < steps) this.steps = steps;

		merged.data().then(data => {
			// cleanup tensors
			temp.dispose();
			current.dispose();
			scaledTemp.dispose();
			scaledCurrent.dispose();
			sum.dispose();
			merged.dispose();
			stepDiv.dispose();

			this.load(data);
			if (typeof callback === 'function') callback();
		})
	}

	/**
	 * Custom training solution here, could be awful
	 * @param {*} weights - Weights to merge into current model weights
	 * @param {number} steps - Number of times weights have been trained by a client
	 * @param {function?} callback - Callback for triggering an action, passed no arguments
	 */
	mimicMerge(weights, steps, accuracy, callback) {
		let temp = tf.tensor(weights, [weights.length]);
		let current = tf.tensor(this.currentWeights, [weights.length]);

		// add both sets of weights together
		let d = steps - this.steps;
		let dif = d > 0 ? tf.sub(temp, current) : tf.sub(current, temp);
		d = Math.abs(d);

		if (d == 0) d = 1;
	
		// merge based on ratio between weights step count
		let ratio = tf.scalar(1 / (d + 1));
		let delta = tf.mul(dif, ratio);
		let merged = tf.add(current, delta);

		if (this.steps < steps) this.steps = steps;

		merged.data().then(data => {
			// cleanup tensors
			temp.dispose();
			current.dispose();
			dif.dispose();
			delta.dispose();
			merged.dispose();
			ratio.dispose();

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

	updateLearningRate(arraybuffer) {
		let lr = new Float32Array(arraybuffer)[0];
		// console.log(lr);
		if (this.lr != lr)
			this.model.optimizer.optimizer.setLearningRate(lr);
	}
}
module.exports = Model;
