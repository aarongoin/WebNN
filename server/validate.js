const
FS = require("fs"),
TF = require("tensorfire"),
GL = require("gl")(512, 512, { preserveDrawingBuffer: true });

var Model = require("../lib/Model")(TF, GL);

module.exports = class Validator extends Model {
	constructor(modelpath, testpath, weights, trainingMeta) {
		// open model file
		super(JSON.parse( FS.readFileSync(modelpath + "model.json", 'utf8') ), weights);
		this.testpath = testpath;
		this.trainingMeta = trainingMeta;

		// load all validation data in to memory
		var origPoolSize = Buffer.poolSize;
		Buffer.poolSize = 161 * 4;
		this.batches = [];
		for (var i = 0; i < trainingMeta.validation_minibatches; i++)
			this.batches.push(
				new Float32Array(
					FS.readFileSync("" + testpath + i).buffer
				)
			);
		this.last_validation = 0;

		Buffer.poolSize = origPoolSize;

		console.log(this.batches);
	}

	validateWeights(weights, callback) {

		if (this.last_validation === this.batches.length)
			this.last_validation = 0;
		
		var data = this.batches[this.last_validation];
		var len = data[0] * this.model.layers[0].shape[1]; // first float is number of samples in this batch

		this.load(weights);
		this.validate(data.subarray(1, ++len), data.subarray(len), callback); 
	}
};