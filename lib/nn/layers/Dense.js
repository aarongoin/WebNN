var ndarray = require("ndarray"),
	TF = (global && global.TF) || window.TF,
	GL = (global && global.GL) || window.GL,

	Funcs = require('./Activations'),
	generateWeights = require('../util/generateWeights'),

	ForwardBiased = `
		uniform Tensor W; // layer weights
		uniform Tensor I; // layer inputs
		float process(ivec4 pos) { // for each unit in output (x: unit, y: sample)
				float n = 0.0;
				for(int i = 0; i < #(W.shape).y; i++){ // for each weight
					if (i == #(W.shape).y - 1) {
						n += W.read(pos.x, i);
					} else {
						n += I.read(i, pos.y) * W.read(pos.x, i);
					}
				}
				return n;
		}
	`,
	ForwardUnbiased = ` // for each output node
		uniform Tensor W; // layer weights
		uniform Tensor I; // layer inputs
		float process(ivec4 pos) {
			float n = 0.0;
			for(int i = 0; i < #(W.shape).y; i++){
				n += I.read(i, pos.y) * W.read(pos.x, i);
			}
			return n;
		}
	`,
	BackwardBiased = ` // for each input node
		uniform Tensor E; // local error (from activation)
		uniform Tensor W; // weights
		float process(ivec4 pos) { // position in input gradient Tensor
			float e = 0.0; // sum output error
			for(int i = 0; i < #(E.shape).x; i++){
				if (pos.y != #(E.shape).x) {
					e += W.read(i, pos.x) * E.read(i, pos.y);
				}
			}
			return e;
		}
	`,
	BackwardUnbiased = ` // for each input node
		uniform Tensor E; // local error (from activation)
		uniform Tensor W; // weights
		float process(ivec4 pos) { // position in input gradient Tensor
			float e = 0.0; // sum output error
			for(int i = 0; i < #(E.shape).x; i++){
				e += W.read(i, pos.x) * E.read(i, pos.y);
			}
			return e;
		}
	`,
	Weights = `
		uniform Tensor E; // local error (from activation)
		uniform Tensor W; // weights
		uniform Tensor I; // input
		uniform float l; // learning rate
		float process(ivec4 pos) { // pos in weights Tensor
			float e = 0.0; // avg node batch error
			for(int i = 0; i < #(E.shape).y; i++){
				if (pos.y == #(I.shape).x) { // handle bias layer ?
					e += E.read(pos.x, i);
				} else {
					e += E.read(pos.x, i) * I.read(pos.y, i);
				}
			}
			return W.read(pos) - (l * e);
		}
	`,
	Activation = (activationFunction) => `
		uniform Tensor O; // weighted input
		float process(ivec4 pos) {
			float n = O.read(pos);
			float o;
			${ activationFunction }
			return o;
		}
	`,
	Gradient = (derivativeFunction) => `
		uniform Tensor E;	// downstream error
		uniform Tensor O;	// layer output
		uniform Tensor H;	// weighted input
		float process(ivec4 pos) {
			float d;
			float o = O.read(pos);
			${ derivativeFunction }
			d *= E.read(pos);
			return d;
		}
	`;

/**
 * Dense fully connected layer
 */
class Dense {
	/**
	 * Create a Dense layer
	 * @param {Object} layer - Object describing layer
	 * @param {string} layer.dense - Activation function for each node in layer
	 * @param {?boolean} layer.bias - If layer uses bias
	 * @param {number} layer.out - Number of output nodes in layer
	 */
	constructor(layer, numInputs) {
		// produce Output Tensor given input, weights, and bias Tensors
		this.forward = layer.bias ? ForwardBiased : ForwardUnbiased;

		this.activation = Activation(Funcs.Activation[layer.activation]);
							
		// produce upstream error Tensor given downstream error, input, weights, bias
		this.backward = layer.bias ? BackwardBiased : BackwardUnbiased;
		this.gradient = Gradient(Funcs.Derivative[layer.activation]);
		// adjust weights Tensor given error and input Tensor
		this.update = Weights;

		this.shape = [layer.out, numInputs];
		this.input = null;
		this.output = null;
		this.weightedOutput = null;
		this.weights = null;
		this.bias = layer.bias;
		this.inputs = numInputs;
		this.outputs = layer.out;
		this.size = layer.out * numInputs + (this.bias ? layer.out : 0);
	}
	/**
	 * Load in weights to layer
	 * @param {Float32Array} array - Array to read weights from (array contains weights for entire model)
	 * @param {number} offset - Index of first weight for this layer in array
	 * @returns {number} Index of element following last weight in layer
	 */
	load(array, offset) {
		var length = this.size;
		// read in weights (and bias)
		this.weights = new TF.InPlaceTensor(GL, ndarray(array.subarray(offset, offset + length), [this.outputs, this.inputs + (this.bias ? 1 : 0)] ) );
		offset += length;
		return offset;
	}
	/**
	 * Sets layer weights to random values centered at 0 and standard deviation relative to number of inputs.
	 */
	randomWeights() {
		this.weights = new TF.InPlaceTensor(GL, 
			ndarray(
				generateWeights(this.shape, (this.bias ? this.outputs : 0)), // values
				[this.outputs, this.inputs + (this.bias ? 1 : 0)] // shape
			)
		);
	}
	/**
	 * Read out layer weights
	 * @returns {Float32Array} Weights and biases (if layer is biased)
	 */
	save() {
		return this.weights.read().data;
	}
	/**
	 * Run the layer on the input.
	 * @param {Float32Array|Tensor} input - input to this layer
	 * @returns {Tensor} Output from layer
	 */
	forward(input) {
		let samples = 0;
		if (input instanceof Float32Array) {
			samples = (input.length / this.inputs) >> 0;
			this.input = new TF.Tensor(GL, ndarray( input, [ this.inputs, samples ]));
		} else this.input = input;

		this.weightedOutput = new TF.OutputTensor(GL, [ this.outputs, samples ]);
		this.weightedOutput.run(this.forward, {W: this.weights, I: this.input});


		this.output = new TF.OutputTensor(GL, [ this.outputs, samples ]);
		this.output.run(this.activation, {O: this.weightedOutput});

		return this.output;
	}
	/**
	 * Backprop error through layer. Will break if layer has not been run() first.
	 * @param {Tensor} error - output error for layer
	 * @param {number} learning_rate - learning rate
	 * @returns {Tensor} Error to propogate to input nodes
	 */
	backward(error, learning_rate) {
		this.partial = new TF.OutputTensor(GL, this.input.shape);
		this.local = new TF.OutputTensor(GL, this.output.shape);

		// calculate local error from weightedOutput (strips out error from activation function)
		this.local.run(this.gradient, {E: error, O: this.output, H: this.weightedOutput});

		// calculate upstream errors from input
		this.partial.run(this.backward, {E: this.local, W: this.weights});

		// train weights based on local error
		this.weights.run(this.update, {W: this.weights, E: this.local, I: this.input, l: learning_rate});

		return this.partial;
	}
}
module.exports = Dense;