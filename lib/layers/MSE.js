var ndarray = require("ndarray"),
	TF,
	GL,

	Grad 	= `
		uniform Tensor O;
		uniform Tensor E;
		float process(ivec4 pos) {
			return O.read(pos) - E.read(pos);
		}
	`,

	Accuracy = `
		uniform Tensor O;
		uniform Tensor E;
		float process(ivec4 pos) {
			float s = 0.0;
			for (int i = 0; i < #(O.shape).x; i++) { // iterate over every output
				s += pow((E.read(i, pos.x) - O.read(i, pos.x)), 2.0);
			}
			return 1.0 - clamp(s / float(#(O.shape).x), 0.0, 1.0);
		}
	`,

	Loss 	= `
		uniform Tensor G;
		float process(ivec4 pos) {
			float loss = 0.0;
			for(int i = 0; i < #(G.shape).y; i++){ // iterate over each sample
				float l = 0.0;
				for(int j = 0; j < #(G.shape).x; j++){ // iterate over every output and calculate average
					l += pow(float(G.read(j, i)), 2.0);
				}
				loss += l / float(#(G.shape).y);
			}
			return loss;
		}
	`;

function MSE() {
	// calculate loss gradients
	this.grad = Grad;

	// calculate batch average loss
	this.lossF = Loss;

	this.Accuracy = Accuracy;

	this.loss = new TF.OutputTensor(GL, [1]);
	this.output = null;
}
MSE.prototype.deltas = function(output, expect) {
	if (expect instanceof Float32Array)
		expect = new TF.Tensor(GL, ndarray( expect, output.shape));

	//console.log("Calculon- expected: " + expect.read().data);

	this.output = new TF.OutputTensor(GL, output.shape);
	this.output.run(this.grad, { O: output, E: expect });
	//console.log("Calculon- gradient: " + this.output.read().data);

	this.loss.run(this.lossF, { G: this.output });

	return this.output;
}

module.exports = function(tensorfire, glContext) {
	TF = tensorfire;
	GL = glContext;
	return MSE;
};