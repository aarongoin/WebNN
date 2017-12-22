var ndarray = require("ndarray"),
	TF,
	GL,
	Softmax = require('./Softmax'),
	MSE = require('./MSE'),
	CrossEntropy = require('./CrossEntropy'),

	Accuracy= "uniform Tensor O; \n"
			+ "uniform Tensor E; \n"
			+ "float process(ivec4 pos) { \n"
				+ "int l = #(O.shape).x; \n"
				+ "float v = 0.0; \n"
				+ "for (int i = 0; i < #(O.shape).x; i++) { \n" /* iterate over every output and calculate average */
					+ "if (O.read(i, pos.x) >= v) { \n"
						+ "v = O.read(i, pos.y); \n" /* get largest category in output */
						+ "l = i; \n" /* save index of largest value */
					+ "} \n"
				+ "} \n"
				+ "if (E.read(l, pos.y) > 0.9) { \n"
					+ "return 1.0; \n"
				+ "} else { \n"
					+ "return 0.0; \n"
				+ "} \n"
			+ "} \n"
			,
	AccSum	= "uniform Tensor A; \n"
			+ "float process(ivec4 pos) { \n"
				+ "float acc = 0.0; \n"
				+ "for(int i = 0; i < #(A.shape).y; i++){ \n" /* iterate over each sample */
					+ "acc += A.read(pos.x, i); \n"
				+ "} \n"
				+ "acc = acc / float(#(A.shape).y); \n"
				+ "return acc; \n"
			+ "} \n"
			;

function Output(layer, index) {
	this.output = null;
	if (layer.activation === "softmax" && layer.loss === "xentropy") {
		this.output = new Softmax(layer, index);
		this.output.deltas = this.output.train;
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
	if (expected instanceof Float32Array)
		expected = new TF.Tensor(GL, ndarray( expected, this._output.shape));

	this.batchAccuracy = new TF.OutputTensor(GL, this._output.shape);
	this._accuracy = new TF.OutputTensor(GL, [1]);
	this.batchAccuracy.run(Accuracy, { O: this._output, E: expected });
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