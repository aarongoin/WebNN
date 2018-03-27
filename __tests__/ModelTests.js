var TF = global.TF = TF = require('tensorfire'),
	GL = global.GL = GL = require('gl')(512, 512, { preserveDrawingBuffer: true }),
	Model = require('../lib/nn/Model'),
	ndarray = require('ndarray'),
	testModel,
	result;

// MODEL TEST
describe('Testing Model Class', () => {

	describe('w/ No Activation with Mean Squared Error Loss', () => {
		test('on input: [0,1], expect: [1], weights:[[0.5, 0.4, 0.25, 0.3, 0.5, 0.6], [0.1, 0.4, 0.3]], output close to [0.3055] and gradient close to [-0.6944]', () => {

			var weights0 = [0.5, 0.4, 0.25, 0.3, 0.5, 0.6],
				weights1 = [0.1, 0.4, 0.3];

			var input = [0, 1],
				expected = [1];

			testModel = new Model([
				{ input: 2 },
				{ dense: 'tanh', out: 3 },
				{ dense: 'tanh', out: 1 },
				{ output: 1, loss: 'mean_squared_error' }
			], weights0.concat(weights1));

			testModel.forward(new Float32Array(input));

			result = testModel.output.outputTensor.read().data;
			expect(result[0]).toBeCloseTo(0.3055);

			result = testModel.output.backward(new Float32Array(expected)).read().data;
			expect(result[0]).toBeCloseTo(-0.6944);
		});

		test('training model twice on the same data should give a lower loss the second time', () => {

			var weights0 = [0.5, 0.4, 0.25, 0.3, 0.5, 0.6],
				weights1 = [0.1, 0.4, 0.3];

			var input = [0, 1, 1, 0, 1, 1, 0, 0],
				expected = [1, 1, 0, 0];

			testModel = new Model([
				{ input: 2 },
				{ dense: 'tanh', out: 3 },
				{ dense: 'tanh', out: 1 },
				{ output: 1, loss: 'mean_squared_error' }
			], weights0.concat(weights1));

			testModel.train(new Float32Array(input), new Float32Array(expected), 1, (weights, accuracy) => {
				var loss1 = testModel.output.loss;
				testModel.train(new Float32Array(input), new Float32Array(expected), 1, (weights, accuracy) => {
					expect(testModel.output.loss).toBeLessThan(loss1);
				});
			});
		});
	});

	describe('w/ Softmax Activation & Cross-Entropy Loss', () => {
		test('on input: [0, 1, 1], and expecting: [1, 0], weights:[0.5, 0.4, 0.25, 0.3, 0.5, 0.6], output close to [0.4430, 0.5569] and gradient close to [-0.5569, 0.5569]', () => {
			
			var weights = [0.5, 0.4, 0.25, 0.3, 0.5, 0.6];

			var input = [0,1,1],
				expected = [1, 0];

			testModel = new Model([
				{ input: 3 },
				{ dense: 'tanh', out: 2 },
				{ output: 2, loss: 'categorical_crossentropy'}
			], weights);

			testModel.forward(new Float32Array(input));

			result = testModel.output.outputTensor.read().data;
			expect(result[0]).toBeCloseTo(0.4430);
			expect(result[1]).toBeCloseTo(0.5569);
			expect(result[0] + result[1]).toBeCloseTo(1);

			result = testModel.output.backward(new Float32Array(expected)).read().data;
			expect(result[0]).toBeCloseTo(-0.5569);
			expect(result[1]).toBeCloseTo(0.5569);
		});

		test('training model twice on the same data should give a lower loss the second time', () => {

			var weights = [0.5, 0.4, 0.25, 0.3, 0.5, 0.6];

			var input = [0, 1, 1],
				expected = [1, 0];

			testModel = new Model([
				{ input: 3 },
				{ dense: 'tanh', out: 2 },
				{ output: 2, loss: 'categorical_crossentropy' }
			], weights);

			testModel.train(new Float32Array(input), new Float32Array(expected), 1, (weights, accuracy) => {
				var loss1 = testModel.output.loss;
				testModel.train(new Float32Array(input), new Float32Array(expected), 1, (weights, accuracy) => {
					expect(testModel.output.loss).toBeLessThan(loss1);
				});
			});
		});
	});
});