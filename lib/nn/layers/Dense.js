const 
ndarray = require("ndarray"),
TF = (global && global.TF) || window.TF,
GL = (global && global.GL) || window.GL,

Funcs = require('./Activations'),
generateWeights = require('../../util/generateWeights');

const ForwardBiased = (activationFunction) => `
	uniform Tensor W; // layer weights
	uniform Tensor I; // layer inputs
	float process(ivec4 pos) { // for each unit in output (x: unit, y: sample)
		float n = 0.0;
		float o = 0.0;
		for(int i = 0; i < #(W.shape).y; i++){ // for each weight
			if (i == #(W.shape).y - 1) {
				n += W.read(pos.x, i);
			} else {
				n += I.read(i, pos.y) * W.read(pos.x, i);
			}
		}
		${ activationFunction }
		return o;
	}
`;
const ForwardUnbiased = (activationFunction) => ` // for each output node
	uniform Tensor W; // layer weights
	uniform Tensor I; // layer inputs
	float process(ivec4 pos) {
		float n = 0.0;
		float o = 0.0;
		for(int i = 0; i < #(W.shape).y; i++){
			n += I.read(i, pos.y) * W.read(pos.x, i);
		}
		${ activationFunction }
		return o;
	}
`;
const BackwardBiasedShader = ` // for each input node
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
`;
const BackwardUnbiasedShader = ` // for each input node
	uniform Tensor E; // local error (from activation)
	uniform Tensor W; // weights
	float process(ivec4 pos) { // position in input gradient Tensor
		float e = 0.0; // sum output error
		for(int i = 0; i < #(E.shape).x; i++){
			e += W.read(i, pos.x) * E.read(i, pos.y);
		}
		return e;
	}
`;
const WeightsShader = `
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
`;
const Gradient = (derivativeFunction) => `
	uniform Tensor E;	// downstream error
	uniform Tensor O;	// layer output
	float process(ivec4 pos) {
		float d;
		float o = O.read(pos);
		${ derivativeFunction }
		d *= E.read(pos);
		return d;
	}
`;
const MergeShader = `
	uniform Tensor A; // original weights
	uniform Tensor B; // new weights
	uniform int o; // offset
	float process(ivec4 pos) {
		return ( A.read(pos) + B.read(pos) ) * 0.5;
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
		this.forwardShader = layer.bias ? ForwardBiased(Funcs.Activation[layer.dense]) : ForwardUnbiased(Funcs.Activation[layer.dense]);
							
		// produce upstream error Tensor given downstream error, input, weights, bias
		this.backwardShader = layer.bias ? BackwardBiasedShader : BackwardUnbiasedShader;
		this.gradientShader = Gradient(Funcs.Derivative[layer.dense]);
		// adjust weights Tensor given error and input Tensor
		this.updateShader = WeightsShader;

		this.shape = [layer.out, numInputs];
		this.input = null;
		this.output = null;
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
		// read in weights (and bias)
		this.weights = new TF.InPlaceTensor(GL, ndarray(array.subarray(offset, offset + this.size), [this.outputs, this.inputs + (this.bias ? 1 : 0)] ) );
		offset += this.size;
		return offset;
	}

	/**
	 * Merge weights from array
	 * @param {Float32Array} array - Array to read weights from (array contains weights for entire model)
	 * @param {number} offset - Index of first weight for this layer in array
	 * @returns {number} Index of element following last weight in layer
	 */
	merge(array, offset) {
		let weights = new TF.Tensor(GL, ndarray(array.subarray(offset, offset + this.size), [this.outputs, this.inputs + (this.bias ? 1 : 0)]));
		// read in weights (and bias)
		this.weights.run(MergeShader, {A: this.weights, B: weights, o: offset});
		offset += this.size;
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
		} else {
			this.input = input;
			samples = input.shape[1];
		}

		this.output = new TF.InPlaceTensor(GL, [ this.outputs, samples ]);
		this.output.run(this.forwardShader, { W: this.weights, I: this.input });

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

		// calculate local error from weightedOutput (strips out error from activation function)
		this.output.run(this.gradientShader, {E: error, O: this.output});

		// calculate upstream errors from input before weights are adjusted
		this.partial.run(this.backwardShader, {E: this.output, W: this.weights});

		// train weights based on local error
		this.weights.run(this.updateShader, {W: this.weights, E: this.output, I: this.input, l: learning_rate});

		return this.partial;
	}
}
module.exports = Dense;