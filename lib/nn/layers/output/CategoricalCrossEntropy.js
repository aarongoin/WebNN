var ndarray = require("ndarray"),
	TF = (global && global.TF) || window.TF,
    GL = (global && global.GL) || window.GL,
    Output = require('./Output');

const SoftmaxShader = `
    uniform Tensor I; // input
    float process(ivec4 pos) {

        float k = 0.0;
        for(int i = 0; i < #(I.shape).x; i++){
            k += exp(I.read(i, pos.y));
        }
        return exp(I.read(pos)) / k;
    }
`;

const GradientShader = `
    uniform Tensor O;
    uniform Tensor E;
    float process(ivec4 pos) {

        return 0.0 - E.read(pos) / O.read(pos);
    }
`;

const LossShader = `
    uniform Tensor O;
    uniform Tensor E;
    float process(ivec4 pos) {

        float loss = 0.0;
        for(int i = 0; i < #(O.shape).y; i++){ // iterate over each sample
            float l = 0.0;
            for(int j = 0; j < #(O.shape).x; j++){ // iterate over every output and calculate average
                l -= E.read(j, i) * log(O.read(j, i));
            }
            loss = l / float(#(O.shape).y);
        }
        return loss;
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

/**
 * Output layer for classifiers or other categorical models
 * @extends {Output}
 */
class CategoricalCrossEntropy extends Output {
    /**
	 * Create output layer using Softmax activation and CrossEntropy loss function
	 * @param {Object} layer - Object describing output layer
	 * @param {number} layer.output - Number of nodes to output
	 */
    constructor(layer, numInputs) {
        super(layer, numInputs);
        // from super:
        // this.accuracy = 0;
		// this.loss = 0;
        // this.layer = layer;
        // this.inputs = numInputs;

        this.SoftmaxShader = SoftmaxShader;
        this.GradientShader = GradientShader;

        this.lossTensor = new TF.OutputTensor(GL, [1]); // would loss be better served running on cpu?
        this.outputTensor = new TF.OutputTensor(GL, [numInputs, layer.output]);
    }

    /**
	 * Perform softmax activation on input
	 * @override
	 * @param {Tensor} inputTensor - Input tensor from last hidden layer
	 * @returns {Tensor} Model output after softmax activation
	 */
    forward(inputTensor) {
        this.outputTensor.run(this.SoftmaxShader, { I: inputTensor });

        return this.outputTensor;
    }
    /**
     * Calculate error of output for backprop.
	 * @override
     * @param {Float32Array|Tensor} expectedTensor - Expected model output
     * @returns {Tensor} Model output's gradient for backprop
     */
    backward(expectedTensor) {
        if (expectedTensor instanceof Float32Array)
            expectedTensor = new TF.Tensor(GL, ndarray(expectedTensor, [1, this.layer.output]));

        this.outputTensor.run(this.GradientShader, { O: this.outputTensor, E: expectedTensor });
        this.lossTensor.run(this.LossShader, { G: this.outputTensor });
        this.loss = this.lossTensor.read().data[0];

        this.lossTensor.run(this.AccuracyShader, { O: this.outputTensor, E: expectedTensor });
        this.accuracy = this.lossTensor.read().data[0];

        return this.outputTensor;
    }
}

module.exports = CategoricalCrossEntropy;