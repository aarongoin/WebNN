const AccuracySummationShader = `
	uniform Tensor A;
	float process(ivec4 pos) {
		
		float acc = 0.0;
		for (int i = 0; i < #(A.shape).y; i++){ // iterate over each sample
			acc += A.read(pos.x, i);
		}
		return acc / float(#(A.shape).y);
	}
`;

/**
 * Output layer used by Model
 */
class Output {
	/**
	 * 
	 * @param {Object} layer - Object describing output layer
	 * @param {number} layer.output - Number of nodes to output
	 * @param {string} layer.loss - Loss function used when evaluating output
	 */
	constructor(layer, numInputs) {
		this.accuracy = 0;
		this.loss = 0;
        this.layer = layer;
        this.inputs = numInputs;
        this.outputs = layer.out;
	}
	/**
	 * Run output layer (no-op unless using softmax classification)
	 * @abstract
	 * @param {Tensor} input - Input tensor from last hidden layer
	 * @returns {Tensor} Model tensor output
	 */
	forward(inputTensor) {
		this.outputTensor = inputTensor;
		return inputTensor;
	};
	/**
	 * Calculate error of output for backprop.
	 * @abstract
	 * @param {Float32Array|Tensor} expectedTensor - Expected output from layer
	 * @returns {Tensor} Gradient error
	 */
	backward(expectedTensor) {
		if (expectedTensor instanceof Float32Array)
			expectedTensor = new TF.Tensor(GL, ndarray( expectedTensor, [ this.inputs, this.outputs ]));

		return expectedTensor;
	}
}
module.exports = Output