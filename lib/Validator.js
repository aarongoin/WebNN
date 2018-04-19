const FS = require("fs");

const Model = require("../nn/Model");
const Bytes = require('../util/byte');

class Validator {
	/**
	 *
	 * @param {string} model - model layers
	 * @param {DataSource} dataSource - Object to pull validation data from
	 */
	constructor(model, dataSource, weights) {
		this.model = new Model(model, weights);
		this.dataSource = dataSource;
	}

	/**
	 * Validate weights
	 * @param {?ArrayBuffer} weights - Weights to validate
	 * @param {?validtionCallback} callback
	 */
	validateWeights(weights, callback) {
		this.model.load(weights);
		console.log('weights loaded into Validator.');

		const data = this.model.readDataBatch(
			this.dataSource.nextTestBatch(64).buffer
		);
		console.log('Aquired dataset.');

		this.model.validate(data.x, data.y, data.n, (loss, accuracy) => {
			console.log('accuracy: ' + accuracy);
			// console.log(this.model.output.outputTensor.read().data);
			callback(accuracy);
		});
	}

	getWeights(callback) {
		return this.model.save((weights) => callback(weights));
	}
};
module.exports = Validator;
