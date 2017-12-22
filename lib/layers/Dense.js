var ndarray = require("ndarray"),
	TF,
	GL,

	Funcs = require('./Activations'),
	generateWeights = require('../util/generateWeights'),

	Forward = "uniform Tensor W; \n" /* weights */
			+ "uniform Tensor I; \n" /* input */
			+ "float process(ivec4 pos) { \n"
				+ "float n = 0.0; \n"
				+ "for(int i = 0; i < #(W.shape).y; i++){ \n"
					+ "if (i == #(W.shape).y - 1) { n += W.read(pos.x, i); } \n"
					+ "else { n += I.read(i, pos.y) * W.read(pos.x, i); } \n"
				+ "} \n"
				+ "return n;\n"
			+ "} \n"
			,
	Backward= "uniform Tensor E; \n" /* local error (from activation) */
			+ "uniform Tensor W; \n" /* weights */
			+ "float process(ivec4 pos) { \n" // position in partial Tensor
				+ "float e = 0.0; \n" /* sum output error */
				+ "for(int i = 0; i < #(E.shape).x ; i++){ \n"
					+ "e += W.read(pos.x, i) * E.read(i, pos.y); \n"
				+ "} \n"
				+ "return e; \n"
			+ "} \n"
			,
	Weights = "uniform Tensor E; \n" /* local error (from activation) */
			+ "uniform Tensor W; \n" /* weights */
			+ "uniform Tensor I; \n" /* input */
			+ "uniform float l; \n" /* learning rate */
			+ "float process(ivec4 pos) { \n" // pos in weights Tensor
				+ "float e = 0.0; \n" /* avg node batch error */
				+ "for(int i = 0; i < #(E.shape).y; i++){ \n"
					+ "if (pos.y == #(I.shape).x) { \n" /* handle bias layer ? */
						+ "e += E.read(pos.x, i) / float(#(E.shape).y); \n"
					+ "} else { \n"
						+ "e += E.read(pos.x, i) * I.read(pos.y, i) / float(#(E.shape).y); \n"
					+ "} \n"
				+ "} \n"
				+ "return W.read(pos) - (l * e); \n"
			+ "} \n"
			,
	ActA 	= "uniform Tensor O; \n" /* weighted output */
			+ "float process(ivec4 pos) { \n"
				+ "float n = O.read(pos); \n"
				+ "float o; \n"
			,
	ActB 	= 	  "return o; \n"
			+ "} \n"
			,
	GradA	= "uniform Tensor E; \n"
			+ "uniform Tensor O; \n"
			+ "uniform Tensor H; \n"
			+ "float process(ivec4 pos) { \n"
				+ "float d; \n"
				+ "float o = O.read(pos); \n"
			,
	GradB 	= 	  "d *= E.read(pos); \n"
			+ 	  "return d; \n"
			+ "} \n"
			;



function Dense(layer, index) {
	this.l = index;
	// produce Output Tensor given input, weights, and bias Tensors
	this.forward = Forward;

	this.activation = ActA + Funcs.Activation[layer.activation] + ActB;
						
	// produce upstream error Tensor given downstream error, input, weights, bias
	this.backward 	= Backward;
	this.gradient 	= GradA + Funcs.Derivative[layer.activation] + GradB;
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
	this.weights = new TF.InPlaceTensor(GL, ndarray( generateWeights(this.shape, (this.bias ? this.shape[0] : 0)), [this.shape[0], this.shape[1] + (this.bias ? 1 : 0)] ) );
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

	//console.log("output " + this.l + ": " + this.output.read().data);
	return this.output;
};
Dense.prototype.train = function(error, learning_rate) {
	var partial = new TF.OutputTensor(GL, this.input.shape);
	var local = new TF.OutputTensor(GL, this.output.shape);

	//console.log("Calculon- error: " + error.read().data);
	//console.log("Calculon- weights " + this.l + ": " + this.weights.read().data);

	local.run(this.gradient, {E: error, O: this.output, H: this.weightedOutput});
	//console.log("Calculon- localE: " + local.read().data);

	// train weights
	this.weights.run(this.update, {W: this.weights, E: local, I: this.input, l: learning_rate});



	//console.log("Calculon- updated " + this.l + ": " + this.weights.read().data);

	// calculate upstream errors
	partial.run(this.backward, {E: error, I: this.input, W: this.weights, O: this.output});

	return partial;
};

module.exports = function(tensorfire, glContext) {
	TF = tensorfire;
	GL = glContext;
	return Dense;
};