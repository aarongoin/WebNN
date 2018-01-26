var ndarray = require("ndarray"),
	TF,
	GL,

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
Dense.prototype.load = function(array, offset) {
	var length = this.size;
	// read in weights (and bias)
	this.weights = new TF.InPlaceTensor(GL, ndarray( array.subarray(offset, offset + length), [this.shape[0], this.shape[1] + (this.bias ? 1 : 0)] ) );
	offset += length;
	return offset;
}
Dense.prototype.randomWeights = function() {
	this.weights = new TF.InPlaceTensor(GL, 
		ndarray(
			generateWeights(this.shape, (this.bias ? this.shape[0] : 0)), // values
			[this.shape[0], this.shape[1] + (this.bias ? 1 : 0)] // shape
		)
	);
}
Dense.prototype.save = function() {
	return this.weights.read().data;
}
Dense.prototype.run = function(input) {
	if (input instanceof Float32Array) {
		this.input = new TF.Tensor(GL, ndarray( input, [ this.shape[1], (input.length / this.shape[1]) >> 0 ]));
	} else this.input = input;
	//console.log(this.input.shape);
	//console.log("Calculon- input " + this.l + ": " + this.input.read().data);
	//console.log("Calculon- weights " + this.l + ": " + this.weights.read().data);

	this.weightedOutput = new TF.OutputTensor(GL, [ this.shape[0], this.input.shape[1] ]);
	this.weightedOutput.run(this.forward, {W: this.weights, I: this.input});

	//console.log("Calculon- weightedOutput " + this.l + ": " + this.weightedOutput.read().data);

	this.output = new TF.OutputTensor(GL, [ this.shape[0], this.input.shape[1] ]);
	this.output.run(this.activation, {O: this.weightedOutput});

	//console.log("output " + this.l: " + this.output.read().data);
	return this.output;
};
Dense.prototype.train = function(error, learning_rate) {
	this.partial = new TF.OutputTensor(GL, this.input.shape);
	this.local = new TF.OutputTensor(GL, this.output.shape);

	//console.log("Calculon- error: " + error.read().data);
	//console.log("Calculon- weights " + this.l: " + this.weights.read().data);

	// calculate local error from weightedOutput (strips out error from activation function)
	this.local.run(this.gradient, {E: error, O: this.output, H: this.weightedOutput});
	//console.log("Calculon- localE: " + local.read().data);

	// calculate upstream errors from input
	this.partial.run(this.backward, {E: this.local, W: this.weights});

	// train weights based on local error
	this.weights.run(this.update, {W: this.weights, E: this.local, I: this.input, l: learning_rate});
	//console.log("Calculon- updated " + this.l: " + this.weights.read().data);

	return this.partial;
};

module.exports = function(tensorfire, glContext) {
	TF = tensorfire;
	GL = glContext;
	return Dense;
};