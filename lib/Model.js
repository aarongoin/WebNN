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
	 * @param {Object} model - Object describing model layers
	 * @param {?(Float32Array)} weights - Weights for all layers
	 * @param {?Integer} steps - Training step for model weights
	 * @param {Float} learning_rate - Current learning rate for model
	 */
	constructor(model, weights, steps, learning_rate) {

		this.model = tf.sequential();
		this.description = model;
		this.lr = learning_rate;
		this.input_shape = model.inputs;
		this.output_shape = model.outputs;

		for (let layer of model.layers) {
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
	validate(inputs, expect, samples, callback) {

		const in_tensor = tf.tensor(inputs, Object.assign([1, 1, 1, 1], [samples].concat(this.input_shape)));
		const out_tensor = tf.tensor(expect, [samples].concat(this.output_shape));

		let result = this.model.evaluate(in_tensor, out_tensor, { batchSize: samples });

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
	 * @param {Float32Array} input - Features input to model
	 * @param {Float32Array} expect - Expected model output
	 * @param {?trainingCallback} callback
	 */
	train(inputs, expect, samples, callback) {
		const in_tensor = tf.tensor(inputs, Object.assign([1, 1, 1, 1], [samples].concat(this.input_shape)));
		const out_tensor = tf.tensor(expect, [samples].concat(this.output_shape));

		this.model.fit(in_tensor, out_tensor, { batchsize: samples, epochs: 1 }).then((history) => {
			in_tensor.dispose();
			out_tensor.dispose();
			this.accuracy = history.history.acc[0];
			if (typeof callback === "function")
				this.save((weights) => callback(weights, history.history.loss[0], history.history.acc[0]) );
				
		});
	}

	/**
	 * Run model on input
	 * @param {Float32Array} input - Features input to model
	 * @return {Float32Array} - Model output
	 */
	run(inputs, samples) {
		const in_tensor = tf.tensor(inputs, Object.assign([1, 1, 1, 1], [samples].concat(this.input_shape)));
		return this.model.predict(in_tensor)
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
