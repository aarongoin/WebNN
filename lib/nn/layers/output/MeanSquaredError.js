var ndarray = require("ndarray"),
	TF = (global && global.TF) || window.TF,
    GL = (global && global.GL) || window.GL,
    Output = require('./Output');

const GradientShader = `
	uniform Tensor O;
	uniform Tensor E;
	float process(ivec4 pos) {

		return O.read(pos) - E.read(pos);
	}
`;

const AccuracyShader = `
	uniform Tensor O;
	uniform Tensor E;
	float process(ivec4 pos) {

		float s = 0.0;
		for (int i = 0; i < #(O.shape).x; i++) { // iterate over every output
			s += pow((E.read(i, pos.x) - O.read(i, pos.x)), 2.0);
		}
		return 1.0 - clamp(s / float(#(O.shape).x), 0.0, 1.0);
	}
`;

const LossShader 	= `
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

/**
 * Output layer for regression models
 * @extends {Output}
 */
class MeanSquaredError extends Output {
    /**
     * Create output layer using MeanSquaredError for loss function
     * @param {Object} layer - Object describing output layer
	 * @param {number} layer.output - Number of nodes to output
     */
    constructor(layer) {
		super(layer);
		// from super:
        // this.accuracy = 0;
		// this.loss = 0;
        // this.layer = layer;
        // this.inputs = this.outputs = layer.out;

        this.gradientShader = GradientShader;
        this.lossShader = LossShader;
        this.accuracyShader = AccuracyShader;

        this.lossTensor = new TF.OutputTensor(GL, [1]);
        this.outputTensor = null;
    }
    /**
     * Calculate error of output for backprop.
	 * @override
     * @param {Float32Array|Tensor} expectedTensor - Expected model output
     */
    backward(expectedTensor) {
        if (expectedTensor instanceof Float32Array)
            expectedTensor = new TF.Tensor(GL, ndarray( expectedTensor, this.outputTensor.shape ));

		// calculate error gradients
		this.outputTensor.run(this.gradientShader, { O: this.outputTensor, E: expectedTensor });

		// calculate model loss
		this.lossTensor.run(this.lossShader, { G: this.outputTensor });
		this.loss = this.lossTensor.read().data[0];

		// calculate model accuracy
        this.lossTensor.run(this.accuracyShader, { O:this.outputTensor , E: expectedTensor });
        this.accuracy = this.lossTensor.read().data[0];

        return this.outputTensor;
    }
}
module.exports = MeanSquaredError;