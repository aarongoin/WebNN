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
		// console.log(weights);
		this.model.load(weights);
		const data = this.dataSource.nextTestBatch(64);

		this.model.validate(data.x, data.y, 64, (accuracy) => {
			// console.log(this.model.output.outputTensor.read().data);
			callback(accuracy);
		});
	}

	getWeights(callback) {
		return this.model.save((weights) => callback(weights));
	}
};
module.exports = Validator;
