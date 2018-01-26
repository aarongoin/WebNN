var ndarray = require("ndarray"),
	TF,
	GL,

	Act = `
		uniform Tensor I; // input
		float process(ivec4 pos) {
			float k = 0.0;
			for(int i = 0; i < #(I.shape).x; i++){
				k += exp(I.read(i, pos.y));
			}
			return exp(I.read(pos)) / k;
		}
	`,
	Grad = `
		uniform Tensor O; // Softmax output
		uniform Tensor E; // expected output
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

	// Accuracy = `
	// 	uniform Tensor O;
	// 	uniform Tensor E;
	// 	float process(ivec4 pos) {
	// 		int l = #(O.shape).x;
	// 		float v = -100000.0;
	// 		for (int i = 0; i < #(O.shape).x; i++) { // iterate over every output
	// 			if (O.read(i, pos.x) > v) {
	// 				v = O.read(i, pos.y); /* get largest category in output */
	// 				l = i; /* save index of largest value */
	// 			}
	// 		}
	// 		if (E.read(l, pos.y) > 0.9) {
	// 			return 1.0;
	// 		} else {
	// 			return 0.0;
	// 		}
	// 	}
	// `,


	Loss = `
		uniform Tensor G;
		float process(ivec4 pos) {
			float loss = 0.0;
			for(int i = 0; i < #(G.shape).y; i++){ /* iterate over each sample */
				float l = 0.0;
				for(int j = 0; j < #(G.shape).x; j++){ /* iterate over every output and calculate average */
					l += pow(float(G.read(j, i)), 2.0) / float(#(G.shape).x);
				}
				loss += l / float(#(G.shape).y);
			}
			return loss;
		}
	`;


function Softmax(layer, index) {
	this.l = index;

	this.activation = Act;

	this.gradient = Grad;
	this.lossF = Loss;	

	this.Accuracy = Accuracy;


	this.input = null;
	this.output = null;
	this.loss = new TF.OutputTensor(GL, [1]);
}
Softmax.prototype.run = function(input) {

	this.input = input;
	this.output = new TF.OutputTensor(GL, this.input.shape);

	this.output.run(this.activation, {I: this.input});

	//console.log("output " + this.l + ": " + this.output.read().data);
	return this.output;
}
Softmax.prototype.deltas = function(output, expected) {
	this.partial = new TF.OutputTensor(GL, this.input.shape);

	if (expected instanceof Float32Array)
		expected = new TF.Tensor(GL, ndarray(expected, this.input.shape));

	// calculate upstream errors
	this.partial.run(this.gradient, {O: output, E: expected});

	// calculate batch training loss
	this.loss.run(this.lossF, { G: this.partial });
	//console.log(this.loss.read());

	// console.log(output.read().data);

	return this.partial;
}

module.exports = function(tensorfire, glContext) {
	TF = tensorfire;
	GL = glContext;
	return Softmax;
}