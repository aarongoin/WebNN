/*
	All tests using npm module 'gl' must live in this test file due to bug
	where testing with the library "gl" (aka headless-gl) breaks if you have
	more than one test file. <https://github.com/facebook/jest/issues/2029>
*/

var TF = global.TF = TF = require('tensorfire'),
	GL = global.GL = GL = require('gl')(512, 512, { preserveDrawingBuffer: true }),
	Dense = require('../lib/nn/layers/Dense'),
	Output = require('../lib/nn/layers/Output'),
	ndarray = require('ndarray'),
	testLayer,
	result;

afterAll(() => {
	GL.getExtension('STACKGL_destroy_context').destroy();
});



describe('Testing Dense Layer', () => {

	describe('w/ Linear Activation', () => {
		test('on input: [0,0,0,0,0], weights: [1,1,1,1,1], activation: linear, outputs: [0]', () => {
			testLayer = new Dense({dense: 'linear', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([0,0,0,0,0])).read().data;

			expect(result[0]).toBe(0);
		});

		test('on input: [1,0,0,0,0], weights: [1,1,1,1,1], activation: linear, outputs: [1]', () => {
			testLayer = new Dense({dense: 'linear', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([1,0,0,0,0])).read().data;

			expect(result[0]).toBe(1);
		});

		test('on input: [1,1,1,1,1], weights: [1,1,1,1,1], activation: linear, outputs: [5]', () => {
			testLayer = new Dense({dense: 'linear', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([1,1,1,1,1])).read().data;

			expect(result[0]).toBe(5);
		});

		test('on input: [1,1,1,1,1], weights: [1,1,-1,-1,-1], activation: linear, outputs: [-1]', () => {
			testLayer = new Dense({dense: 'linear', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,-1,-1,-1]), 0);
			result = testLayer.forward(new Float32Array([1,1,1,1,1])).read().data;

			expect(result[0]).toBe(-1);
		});

		test('on error: [0,-0.5], weights: [1,0,1, 0,1,0], activation: linear, learning_rate: 0.1, trained weights: [1,0,1,0,1,0]', () => {
			var error = new TF.OutputTensor(GL, ndarray(new Float32Array([0,-0.5]), [2]));

			testLayer = new Dense({ dense: 'linear', out: 2 }, 3);
			testLayer.load(new Float32Array([1,0,1, 0,1,0]), 0);

			result = testLayer.forward(new Float32Array([1,0,0])).read().data

			expect(result[0]).toBeCloseTo(1);
			expect(result[1]).toBeCloseTo(0);

			result = testLayer.backward(error, 0.1).read().data;

			expect(result[0]).toBeCloseTo(0);
			expect(result[1]).toBeCloseTo(-0.5);
			expect(result[2]).toBeCloseTo(0);

			result = testLayer.weights.read().data;

			expect(result[0]).toBeCloseTo(1);
			expect(result[1]).toBeCloseTo(0);
			expect(result[2]).toBeCloseTo(1);
			expect(result[3]).toBeCloseTo(0.05);
			expect(result[4]).toBeCloseTo(1);
			expect(result[5]).toBeCloseTo(0);
		});
	});

	describe('w/ Relu Activation', () => {
		test('on input: [0,0,0,0,0], weights: [1,1,1,1,1], activation: relu, outputs: [0]', () => {
			testLayer = new Dense({ dense: 'relu', out: 1 }, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([0,0,0,0,0])).read().data;

			expect(result[0]).toBe(0);
		});

		test('on input: [1,0,0,0,0], weights: [1,1,1,1,1], activation: relu, outputs: [1]', () => {
			testLayer = new Dense({ dense: 'relu', out: 1 }, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([1,0,0,0,0])).read().data;

			expect(result[0]).toBe(1);
		});

		test('on input: [1,1,1,1,1], weights: [1,1,1,1,1], activation: relu, outputs: [5]', () => {
			testLayer = new Dense({dense: 'relu', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([1,1,1,1,1])).read().data;

			expect(result[0]).toBe(5);
		});

		test('on input: [1,1,1,1,1], weights: [1,1,-1,-1,-1], activation: relu, outputs: [0]', () => {
			testLayer = new Dense({dense: 'relu', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,-1,-1,-1]), 0);
			result = testLayer.forward(new Float32Array([1,1,1,1,1])).read().data;

			expect(result[0]).toBe(0);
		});
	});

	describe('w/ Leaky Relu Activation', () => {
		test('on input: [0,0,0,0,0], weights: [1,1,1,1,1], activation: lrelu, outputs: [0]', () => {
			testLayer = new Dense({dense: 'lrelu', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([0,0,0,0,0])).read().data;

			expect(result[0]).toBe(0);
		});

		test('on input: [1,0,0,0,0], weights: [1,1,1,1,1], activation: lrelu, outputs: [1]', () => {
			testLayer = new Dense({dense: 'lrelu', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([1,0,0,0,0])).read().data;

			expect(result[0]).toBe(1);
		});

		test('on input: [1,1,1,1,1], weights: [1,1,1,1,1], activation: lrelu, outputs: [5]', () => {
			testLayer = new Dense({dense: 'lrelu', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([1,1,1,1,1])).read().data;

			expect(result[0]).toBe(5);
		});

		test('on input: [1,1,1,1,1], weights: [1,1,-1,-1,-1], activation: lrelu, outputs: [-0.01]', () => {
			testLayer = new Dense({dense: 'lrelu', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,-1,-1,-1]), 0);
			result = testLayer.forward(new Float32Array([1,1,1,1,1])).read().data;

			expect(result[0]).toBeCloseTo(-0.01);
		});
	});

	describe('w/ Sigmoid Activation', () => {
		test('on input: [0,0,0,0,0], weights: [1,1,1,1,1], activation: sigmoid, outputs: [0.5]', () => {
			testLayer = new Dense({dense: 'sigmoid', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([0,0,0,0,0])).read().data;

			expect(result[0]).toBeCloseTo(0.5);
		});

		test('on input: [1,0,0,0,0], weights: [1,1,1,1,1], activation: sigmoid, outputs: [0.7310]', () => {
			testLayer = new Dense({dense: 'sigmoid', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([1,0,0,0,0])).read().data;

			expect(result[0]).toBeCloseTo(0.7310);
		});

		test('on input: [1,1,1,1,1], weights: [1,1,1,1,1], activation: sigmoid, outputs: [0.9933]', () => {
			testLayer = new Dense({dense: 'sigmoid', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([1,1,1,1,1])).read().data;

			expect(result[0]).toBeCloseTo(0.9933);
		});

		test('on input: [1,1,1,1,1], weights: [1,1,-1,-1,-1], activation: sigmoid, outputs: [0.2689]', () => {
			testLayer = new Dense({dense: 'sigmoid', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,-1,-1,-1]), 0);
			result = testLayer.forward(new Float32Array([1,1,1,1,1])).read().data;

			expect(result[0]).toBeCloseTo(0.2689);
		});
	});

	describe('w/ Tanh Activation', () => {
		test('on input: [0,0,0,0,0], weights: [1,1,1,1,1], activation: tanh, outputs: [0]', () => {
			testLayer = new Dense({dense: 'tanh', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([0,0,0,0,0])).read().data;

			expect(result[0]).toBe(0);
		});

		test('on input: [1,0,0,0,0], weights: [1,1,1,1,1], activation: tanh, outputs: [0.7615]', () => {
			testLayer = new Dense({dense: 'tanh', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([1,0,0,0,0])).read().data;

			expect(result[0]).toBeCloseTo(0.7615);
		});

		test('on input: [1,1,1,1,1], weights: [1,1,1,1,1], activation: tanh, outputs: [0.9999]', () => {
			testLayer = new Dense({dense: 'tanh', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([1,1,1,1,1])).read().data;

			expect(result[0]).toBeCloseTo(0.9999);
		});

		test('on input: [1,1,1,1,1], weights: [1,1,-1,-1,-1], activation: tanh, outputs: [-0.7615]', () => {
			testLayer = new Dense({dense: 'tanh', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,-1,-1,-1]), 0);
			result = testLayer.forward(new Float32Array([1,1,1,1,1])).read().data;

			expect(result[0]).toBeCloseTo(-0.7615);
		});
	});

	describe('w/ Softplus Activation', () => {
		test('on input: [0,0,0,0,0], weights: [1,1,1,1,1], activation: softplus, outputs: [0.6931]', () => {
			testLayer = new Dense({dense: 'softplus', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([0,0,0,0,0])).read().data;

			expect(result[0]).toBeCloseTo(0.6931);
		});

		test('on input: [1,0,0,0,0], weights: [1,1,1,1,1], activation: softplus, outputs: [1.3132]', () => {
			testLayer = new Dense({dense: 'softplus', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([1,0,0,0,0])).read().data;

			expect(result[0]).toBeCloseTo(1.3132);
		});

		test('on input: [1,1,1,1,1], weights: [1,1,1,1,1], activation: softplus, outputs: [5.0067]', () => {
			testLayer = new Dense({dense: 'softplus', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,1,1,1]), 0);
			result = testLayer.forward(new Float32Array([1,1,1,1,1])).read().data;

			expect(result[0]).toBeCloseTo(5.0067);
		});

		test('on input: [1,1,1,1,1], weights: [1,1,-1,-1,-1], activation: softplus, outputs: [0.3132]', () => {
			testLayer = new Dense({dense: 'softplus', out: 1}, 5);
			testLayer.load(new Float32Array([1,1,-1,-1,-1]), 0);
			result = testLayer.forward(new Float32Array([1,1,1,1,1])).read().data;

			expect(result[0]).toBeCloseTo(0.3132);
		});
	});
});

// OUTPUT LAYER TEST
describe('Testing Output Layer', () => {

	describe('w/ Softmax Activation & Cross-Entropy Loss', () => {
		test('on input: [0,0,0,0,1], and expecting: [0,0,0,0,1], error output: [0.1488, 0.1488, 0.1488, 0.1488, -0.5953]', () => {
			var testOutput = new TF.Tensor(GL, ndarray(new Float32Array([0,0,0,0,1]), [5, 1]));
			testLayer = Output({ output: 5, loss: 'categorical_crossentropy'});
			console.log(testLayer.forward(testOutput).read().data);
			result = testLayer.backward(new Float32Array([0,0,0,0,1])).read().data;
			console.log(result);

			expect(result[0]).toBeCloseTo(0.1488475799560547);
			expect(result[1]).toBeCloseTo(0.1488475799560547);
			expect(result[2]).toBeCloseTo(0.1488475799560547);
			expect(result[3]).toBeCloseTo(0.1488475799560547);
			expect(result[4]).toBeCloseTo(-0.5953903198242188);
		});
	});

	describe('w/ No Activation & Mean-Square-Error Loss', () => {
		test('on input: [0,0,0,0,1], and expecting: [0,1,0,0,1], error output: [0,-1,0,0,0]', () => {
			var testOutput = new TF.InPlaceTensor(GL, ndarray(new Float32Array([0,0,0,0,1]), [5, 1]));
			testLayer = Output({ output: 5, loss: 'mean_squared_error' });
			testLayer.forward(testOutput);
			result = testLayer.backward(new Float32Array([0,1,0,0,1])).read().data;

			expect(result[0]).toBeCloseTo(0);
			expect(result[1]).toBeCloseTo(-1);
			expect(result[2]).toBeCloseTo(0);
			expect(result[3]).toBeCloseTo(0);
			expect(result[4]).toBeCloseTo(0);
		});
	});
});