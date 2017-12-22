var ndarray = require("ndarray"),
	TF,
	GL,

	Grad 	= "uniform Tensor O; \n"
			+ "uniform Tensor E; \n"
			+ "float process(ivec4 pos) { \n"
				+ "return 0.0 - E.read(pos) / O.read(pos); \n"
			+ "} \n"
			,

	Loss 	= "uniform Tensor O; \n"
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

function CrossEntropy() {
	// calculate loss gradients
	this.grad = Grad;

	// calculate batch average loss
	this.lossF = Loss;

	this.loss = new TF.OutputTensor(GL, [1]);
	this.output = null;
}
CrossEntropy.prototype.deltas = function(output, expect) {
	if (expect instanceof Float32Array)
		expect = new TF.Tensor(GL, ndarray( expect, output.shape));

	//console.log("Calculon- expected: " + expect.read().data);

	this.output = new TF.OutputTensor(GL, output.shape);
	this.output.run(this.grad, { O: output, E: expect });
	//console.log("Calculon- gradient: " + this.output.read().data);

	this.loss.run(this.lossF, { O: output, E: expect });
	//console.log(this.loss.read());

	return this.output;
}

module.exports = function(tensorfire, glContext) {
	TF = tensorfire;
	GL = glContext;
	return CrossEntropy;
};