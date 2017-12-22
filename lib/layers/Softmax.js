var ndarray = require("ndarray"),
	TF,
	GL,

	Act 	= "uniform Tensor O; \n" /* input */
			+ "float process(ivec4 pos) { \n"
				+ "float n = O.read(pos); \n"
				+ "float o; \n"
				+ "float k = 0.0; \n"
				+ "for(int i = 0; i < #(O.shape).x; i++){ \n"
					+ "k += exp(O.read(i, pos.y)); \n"
				+ "} \n"
				+ "return exp(n) / k; \n"
			+ "} \n"
			,
	Grad	= "uniform Tensor O; \n"
			+ "uniform Tensor E; \n"
			+ "float process(ivec4 pos) { \n"
				+ "return O.read(pos) - E.read(pos); \n"
			+ "} \n"
			,
	Loss 	= "uniform Tensor G; \n"
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


function Softmax(layer, index) {
	this.l = index;

	this.activation = Act;

	this.gradient = Grad;
	this.lossF = Loss;	

	this.input = null;
	this.output = null;
	this.loss = new TF.OutputTensor(GL, [1]);
}
Softmax.prototype.run = function(input) {

	this.input = input;
	this.output = new TF.OutputTensor(GL, this.input.shape);

	this.output.run(this.activation, {O: this.input});

	//console.log("output " + this.l + ": " + this.output.read().data);
	return this.output;
}
Softmax.prototype.train = function(expected) {
	var partial = new TF.OutputTensor(GL, this.input.shape);

	if (expected instanceof Float32Array)
		expected = new TF.Tensor(GL, ndarray(expected, this.input.shape));

	// calculate upstream errors
	partial.run(this.gradient, {O: this.output, E: expected});

	// calculate batch training loss
	this.loss.run(this.lossF, { G: partial });
	//console.log(this.loss.read());

	return partial;
}

module.exports = function(tensorfire, glContext) {
	TF = tensorfire;
	GL = glContext;
	return Softmax;
}