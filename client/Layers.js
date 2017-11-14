var ndarray = require("ndarray"),
	TF,
	GL;

// Standard Normal variate using Box-Muller transform.
function random(mean, stdDev) {
	mean = mean || 0;
	stdDev = stdDev || 1;
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    //return 0.4;
    return (Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v )) * stdDev + mean;
}

function generate(shape, bias) {
	var result = new Float32Array(shape[0] * shape[1] + bias);
	var l = -1;
	while (++l < result.length) {
		result[l] = random(0, Math.sqrt(2 / shape[1]));
	}
	//console.log(result[0]);
	return result;
}

var Activation = {
	"linear": "o = n; \n",
	//"binary": "if (n > 0.0) { o = 0.0; } else { o = 1.0; } \n",
	"relu": "o = max(0.0, n); \n",
	"lrelu": "if (n > 0.0) { o = n; } else { o = 0.01 * n; } \n",
	"sigmoid": "o = 1.0 / (1.0 + exp(0.0 - n)); \n",
	"tanh": "o = (2.0 / (1.0 + exp(-2.0 * n))) - 1.0; \n",
	"softplus": "o = log(1.0 + exp(n)); \n",
	"softmax": "float k = 0.0; \nfor(int i = 0; i < #(O.shape).x; i++){\nk += exp(O.read(i, pos.y));\n}\no = exp(n) / k; \n",
	//"softsign": "o = n / (1.0 + abs(n)); \n"
};
var Derivative = {
	"linear": "d = 1.0; \n",
	//"binary": "if (o == 0.0) { d = 0.0; } else { d = 0.0; } \n",
	"relu": "if (o >= 0.0) { d = 1.0; } else { d = 0.0; } \n",
	"lrelu": "if (o >= 0.0) { d = 1.0; } else { d = 0.01; } \n",
	"sigmoid": "d = o * ( 1.0 - o ); \n",
	"tanh": "d = ( 1 - pow(o, 2.0) ); \n",
	"softplus": "d = 1.0 - ( 1.0 / exp(o) ); \n",
	"softmax": "d = o * ( 1.0 - o ); \n" // same as sigmoid?
	//"softsign": "var = "
};

function DenseLayer(layer, index) {
	this.l = index;
	// produce Output Tensor given input, weights, and bias Tensors
	this.forward 	= "uniform Tensor W; \n" /* weights */
					+ "uniform Tensor I; \n" /* input */
					+ "float process(ivec4 pos) { \n"
						+ "float n = 0.0; \n"
						+ "for(int i = 0; i < #(W.shape).y; i++){ \n"
							+ "if (i == #(W.shape).y - 1) { n += W.read(pos.x, i); } \n"
							+ "else { n += I.read(i, pos.y) * W.read(pos.x, i); } \n"
						+ "} \n"
						+ "return n;\n"
					+ "} \n"
					;

	this.activation = "uniform Tensor O; \n" /* weighted output */
					+ "float process(ivec4 pos) { \n"
						+ "float n = O.read(pos); \n"
						+ "float o; \n"
						+ Activation[layer.activation]
						+ "return o; \n"
					+ "} \n"
					;
	// produce upstream error Tensor given downstream error, input, weights, bias
	this.backward 	= "uniform Tensor E; \n" /* local error (from activation) */
					+ "uniform Tensor W; \n" /* weights */
					+ "float process(ivec4 pos) { \n" // position in partial Tensor
						+ "float e = 0.0; \n" /* sum output error */
						+ "for(int i = 0; i < #(E.shape).x ; i++){ \n"
							+ "e += W.read(pos.x, i) * E.read(i, pos.y); \n"
						+ "} \n"
						+ "return e; \n"
					+ "} \n"
					;
	this.gradient 	= "uniform Tensor E; \n"
					+ "uniform Tensor O; \n"
					+ "uniform Tensor H; \n"
					+ "float process(ivec4 pos) { \n"
						+ "float d; \n"
						+ "float o = O.read(pos); \n"
						+ Derivative[layer.activation]
						+ "d *= E.read(pos); \n"
						+ "return d; \n"
					+ "} \n"
					;
	// adjust weights Tensor given error and input Tensor
	this.update		= "uniform Tensor E; \n" /* local error (from activation) */
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
					;

	this.shape = layer.shape;
	this.input = null;
	this.output = null;
	this.weightedOutput = null;
	this.weights = null;
	this.bias = layer.bias;
	this.size = this.shape[0] * this.shape[1] + (this.bias ? this.shape[0] : 0);

}
DenseLayer.prototype.load = function(array, offset) {
	var length = this.size;
	// read in weights (and bias)
	this.weights = new TF.InPlaceTensor(GL, ndarray( array.subarray(offset, offset + length), [this.shape[0], this.shape[1] + (this.bias ? 1 : 0)] ) );
	offset += length;
	return offset;
}
DenseLayer.prototype.randomWeights = function() {
	this.weights = new TF.InPlaceTensor(GL, ndarray( generate(this.shape, (this.bias ? this.shape[0] : 0)), [this.shape[0], this.shape[1] + (this.bias ? 1 : 0)] ) );
}
DenseLayer.prototype.save = function() {
	return this.weights.read().data;
}
DenseLayer.prototype.run = function(input) {
	var t = ndarray( input, [ this.shape[1], input.length / this.shape[1] ]);
	if (input instanceof Float32Array) {
		this.input = new TF.Tensor(GL, ndarray( input, [ this.shape[1], input.length / this.shape[1] ]));
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
DenseLayer.prototype.train = function(error, learning_rate) {
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

function LossMSE() {
	// calculate loss gradients
	this.grad 	= "uniform Tensor O; \n"
				+ "uniform Tensor E; \n"
				+ "float process(ivec4 pos) { \n"
					+ "return O.read(pos) - E.read(pos); \n"
				+ "} \n"
				;

	// calculate batch average loss
	this.lossF 	= "uniform Tensor G; \n"
				+ "float process(ivec4 pos) { \n"
					+ "float loss = 0.0; \n"
					+ "for(int i = 0; i < #(G.shape).y; i++){ \n" /* iterate over each sample */
						+ "float l = 0.0; \n"
						+ "for(int j = 0; j < #(G.shape).x; j++){ \n" /* iterate over every output and calculate average */
							+ "l += pow(float(G.read(j, i)), 2.0) / float(#(G.shape).x); \n"
						+ "} \n"
						+ "loss += l / float(#(G.shape).y); \n"
					+ "} \n"
					+ "return loss; \n"
				+ "} \n"
				;

	this.loss = new TF.OutputTensor(GL, [1]);
	this.output = null;
	this.batchLoss = 0.0;
}
LossMSE.prototype.deltas = function(output, expect) {
	if (expect instanceof Float32Array)
		expect = new TF.Tensor(GL, ndarray( expect, output.shape));

	//console.log("Calculon- expected: " + expect.read().data);

	this.output = new TF.OutputTensor(GL, output.shape);
	this.output.run(this.grad, { O: output, E: expect });
	//console.log("Calculon- gradient: " + this.output.read().data);

	this.loss.run(this.lossF, { G: this.output });

	this.batchLoss = this.loss.read().data[0];

	return this.output;
}

function CrossEntropy() {
	// calculate loss gradients
	this.grad 	= "uniform Tensor O; \n"
				+ "uniform Tensor E; \n"
				+ "float process(ivec4 pos) { \n"
					+ "return 0.0 - E.read(pos) / O.read(pos); \n"
				+ "} \n"
				;

	// calculate batch average loss
	this.lossF 	= "uniform Tensor O; \n"
				+ "uniform Tensor E; \n"
				+ "float process(ivec4 pos) { \n"
					+ "float loss = 0.0; \n"
					+ "for(int i = 0; i < #(O.shape).y; i++){ \n" /* iterate over each sample */
						+ "float l = 0.0; \n"
						+ "for(int j = 0; j < #(O.shape).x; j++){ \n" /* iterate over every output and calculate average */
							+ "l -= E.read(j, i) * log(O.read(j, i)); \n"
						+ "} \n"
						+ "loss = l / float(#(O.shape).y); \n"
					+ "} \n"
					+ "return loss; \n"
				+ "} \n"
				;

	this.loss = new TF.OutputTensor(GL, [1]);
	this.output = null;
	this.batchLoss = 0.0;
}
CrossEntropy.prototype.deltas = function(output, expect) {
	if (expect instanceof Float32Array)
		expect = new TF.Tensor(GL, ndarray( expect, output.shape));

	//console.log("Calculon- expected: " + expect.read().data);

	this.output = new TF.OutputTensor(GL, output.shape);
	this.output.run(this.grad, { O: output, E: expect });
	//console.log("Calculon- gradient: " + this.output.read().data);

	this.loss.run(this.lossF, { O: output, E: expect });
	console.log(this.loss.read());

	this.batchLoss = this.loss.read().data[0];

	return this.output;
}

module.exports = module.exports = function(tensorfire, glContext) {

	TF = tensorfire;
	GL = glContext;

	return {
		"dense": DenseLayer,
		"mse": LossMSE,
		"xentropy": CrossEntropy
	};
};