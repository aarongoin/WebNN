var ndarray = require("ndarray"),
	TF,
	GL,
	Softmax = require('./Softmax'),
	MSE = require('./MSE'),
	CrossEntropy = require('./CrossEntropy'),

	AccSum	= `
		uniform Tensor A;
		float process(ivec4 pos) {
			float acc = 0.0;
			for (int i = 0; i < #(A.shape).y; i++){ // iterate over each sample
				acc += A.read(pos.x, i);
			}
			return acc / float(#(A.shape).y);
		}
	`;

function Output(layer, index) {
	this.output = null;
	if (layer.activation === "softmax" && layer.loss === "xentropy") {
		this.output = new Softmax(layer, index);
		this.run = (input) => {
			this._output = this.output.run(input);
			return this._output;
		};
	} else {
		switch (layer.loss) {
			case "xentropy":
				this.output = new CrossEntropy();
				break;
			case "mse":
				this.output = new MSE();
				break;
		}
		this.run = this.run.bind(this);
		this.train = this.train.bind(this);
	}

	this._output = null;
	this.accuracy = 0;
}
Output.prototype.run = function(input) {
	this._output = input;
	return input;
};
Output.prototype.train = function(expected) {
	//console.log("Expected: " + expected);
	if (expected instanceof Float32Array)
		expected = new TF.Tensor(GL, ndarray( expected, this._output.shape));

	
	//console.log("  Output: " + this._output.read().data);

	this.batchAccuracy = new TF.OutputTensor(GL, [1, this._output.shape[1]]);
	this._accuracy = new TF.OutputTensor(GL, [1]);
	this.batchAccuracy.run(this.output.Accuracy, { O: this._output, E: expected });
	this._accuracy.run(AccSum, { A: this.batchAccuracy });
	this.accuracy = this._accuracy.read().data[0];

	return this.output.deltas(this._output, expected);
};


module.exports = function(tensorfire, glContext) {

	TF = tensorfire;
	GL = glContext;

	Softmax = Softmax(tensorfire, glContext);
	MSE = MSE(tensorfire, glContext);
	CrossEntropy = CrossEntropy(tensorfire, glContext);

	return Output;
};