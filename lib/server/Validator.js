const FS = require("fs");
global.TF = require("tensorfire");
global.GL = require("gl")(512, 512, { preserveDrawingBuffer: true });

const Model = require("../nn/Model");
const Bytes = require('../util/byte');

class Validator {
	/**
	 * 
	 * @param {string} model - model layers
	 * @param {DataSource} dataSource - Object to pull validation data from
	 * @param {boolean} byte_weights - is model using byte_weights or not
	 */
	constructor(model, dataSource, byte_weights, weights) {
		this.model = new Model(model, (weights && byte_weights ? Bytes.floatFromByteArray(weights) : weights) );
		this.dataSource = dataSource;
		this.byte_weights = byte_weights;
	}

	/**
	 * Validate weights
	 * @param {?ArrayBuffer} weights - Weights to validate
	 * @param {?validtionCallback} callback
	 */
	validateWeights(weights, callback) {
		// console.log(weights);
		this.model.load( this.byte_weights ? 
			Bytes.floatFromByteArray( weights )
			: weights
		);
		const data = this.dataSource.getTestBatch(64);

		this.model.validate(data.x, data.y, (accuracy) => {
			// console.log(this.model.output.outputTensor.read().data);
			callback(accuracy);
		});
	}

	getWeights() {
		return this.byte_weights ? Bytes.byteFromFloatArray(this.model.save(false)) : this.model.save(false);
	}
};
module.exports = Validator;