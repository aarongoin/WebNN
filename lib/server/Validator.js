const FS = require("fs");
global.TF = require("tensorfire");
global.GL = require("gl")(512, 512, { preserveDrawingBuffer: true });

var Model = require("../nn/Model");

/**
 * Legacy model validator
 * @extends Model
 */
class Validator extends Model {
	/**
	 * 
	 * @param {string} modelpath - Path to directory containing model
	 * @param {string} testpath - Path to director containing validation minibatches
	 * @param {?(Float32Array|ArrayBuffer)} weights - Weights to load into Validator
	 * @param {number} batches - Number of validation minibatches
	 */
	constructor(modelpath, testpath, weights, batches) {
		// open model file
		super(JSON.parse( FS.readFileSync(modelpath + "model.json", 'utf8') ), weights);
		this.testpath = testpath;
		this.trainingMeta = trainingMeta;

		// load all validation data in to memory
		var origPoolSize = Buffer.poolSize;
		Buffer.poolSize = 161 * 4; // TODO: refactor as 161 is a magic number that only works for iris dataset
		this.batches = [];
		for (var i = 0; i < batches; i++)
			this.batches.push(
				new Float32Array(
					FS.readFileSync("" + testpath + i).buffer
				)
			);
		this.last_validation = 0;

		Buffer.poolSize = origPoolSize;

		console.log(this.batches);
	}

	/**
	 * Validate weights
	 * @param {?(Float32Array|ArrayBuffer)} weights - Weights to validate
	 * @param {?validtionCallback} callback
	 */
	validateWeights(weights, callback) {

		if (this.last_validation === this.batches.length)
			this.last_validation = 0;
		
		var data = this.batches[this.last_validation];
		var len = data[0] * this.model.layers[0].shape[1]; // first float is number of samples in this batch

		this.load(weights);
		this.validate(data.subarray(1, ++len), data.subarray(len), callback); 
	}
};
module.exports = Validator;