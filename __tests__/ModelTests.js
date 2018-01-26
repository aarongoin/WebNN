var TF = require('tensorfire'),
	GL = require('gl')(512, 512, { preserveDrawingBuffer: true }),
	Model = require('../lib/Model')(TF, GL),
	ndarray = require('ndarray'),
	testModel,
	result;

// MODEL TEST
describe('Testing Model Class', () => {

	describe('w/ Softmax Activation & Cross-Entropy Loss', () => {
		test('on input: [0,1], and expecting: [1],', () => {
			
			var weights0 = [0.5, 0.4, 0.25, 0.3, 0.5, 0.6],
				weights1 = [0.1, 0.4, 0.3];

			testModel = new Model({ layers: [
				{type:"dense", activation:"tanh", shape:[3,2], bias:false},
				{type:"dense", activation:"tanh", shape:[1,3], bias:false},
				{type:"output", activation: 'none', loss: 'mse'}
			]}, weights0.concat(weights1));


		for (var i = 0; i < 10; i++) {
			testModel.train(1, 1, new Float32Array([0,1, 1,0, 1,1, 0,0]), new Float32Array([1, 1, 0, 0]));

			var output = `
Layer 0:
   Layer Input => [${testModel.layers[0].input.read().data}] [${testModel.layers[0].partial.read().data}] <= Backprop
   	   Weights => [${weights0}] [${testModel.layers[0].weights.read().data}] <= New Weights
Weighted Input => [${testModel.layers[0].weightedOutput.read().data}] [${testModel.layers[0].local.read().data}] <= Gradient
  Layer Output => [${testModel.layers[0].output.read().data}] [${testModel.layers[1].partial.read().data}] <= Error
		
Layer 1:
   Layer Input => [${testModel.layers[1].input.read().data}] [${testModel.layers[1].partial.read().data}] <= Backprop
   	   Weights => [${weights1}] [${testModel.layers[1].weights.read().data}] <= New Weights
Weighted Input => [${testModel.layers[1].weightedOutput.read().data}] [${testModel.layers[1].local.read().data}] <= Gradient
  Layer Output => [${testModel.layers[1].output.read().data}] [${testModel.layers[2].output.output.read().data}] <= Error
			`;

			output = output.replace(/\d{12},/g, ",");
			output = output.replace(/\d{12}]/g, "]");
			console.log(output);
		}

			expect(result[0]).toBeCloseTo(0.5);
			expect(result[1]).toBeCloseTo(0.18014991283416748);
			expect(result[2]).toBeCloseTo(0.5);
			expect(result[3]).toBeCloseTo(0.5);
			expect(result[4]).toBeCloseTo(0.18014991283416748);
			expect(result[5]).toBeCloseTo(0.5);
		});
	});
});