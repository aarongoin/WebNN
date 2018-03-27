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

        return O.read(pos) - E.read(pos);
    }
`;

const LossShader = `
    uniform Tensor O;
    uniform Tensor E;
    float process(ivec4 pos) {

        float l = 0.0;
        float e = 0.0;
        float o = 0.0;
        for(int i = 0; i < #(O.shape).y; i++){ // iterate over each sample
            for(int j = 0; j < #(O.shape).x; j++){ // iterate over every output and calculate average
                e = E.read(j, i);
                o = O.read(j, i);
                l -= ( e * log(o) ) + ( (1.0 - e) * log(1.0 - o) );
            }
        }
        return ( l / float(#(O.shape).y) );
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
    constructor(layer) {
        super(layer);
        // from super:
        // this.accuracy = 0;
		// this.loss = 0;
        // this.layer = layer;
        // this.inputs = this.outputs = layer.out;

        this.softmaxShader = SoftmaxShader;
        this.gradientShader = GradientShader;
        this.lossShader = LossShader;
        this.accuracyShader = AccuracyShader;

        this.lossTensor = new TF.OutputTensor(GL, [1]); // would loss be better served running on cpu?
        this.outputTensor = null;
    }

    /**
	 * Perform softmax activation on input
	 * @override
	 * @param {Tensor} inputTensor - Input tensor from last hidden layer
	 * @returns {Tensor} Model output after softmax activation
	 */
    forward(inputTensor) {
        if (this.outputTensor === null)
            this.outputTensor = new TF.InPlaceTensor(GL, inputTensor.shape)
        this.outputTensor.run(this.softmaxShader, { I: inputTensor });

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

        // calculate model loss
        this.lossTensor.run(this.lossShader, { O: this.outputTensor, E: expectedTensor });
        this.loss = this.lossTensor.read().data[0];

        // calculate model accuracy
        this.lossTensor.run(this.accuracyShader, { O: this.outputTensor, E: expectedTensor });
        this.accuracy = this.lossTensor.read().data[0];

        // calculate error gradients
        this.outputTensor.run(this.gradientShader, { O: this.outputTensor, E: expectedTensor });

        return this.outputTensor;
    }
}

module.exports = CategoricalCrossEntropy;