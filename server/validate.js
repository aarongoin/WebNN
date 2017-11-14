const
FS = require("fs"),
TF = require("tensorfire"),
GL = require("gl")(512, 512, { preserveDrawingBuffer: true });

var Model = require("../client/Model")(TF, GL);

module.exports = class Validator extends Model {
	constructor(modelpath, testpath, weights, trainingMeta) {
		// open model file
		super(JSON.parse( FS.readFileSync(modelpath + "model.json", 'utf8') ), weights);
		this.testpath = testpath;
		this.trainingMeta = trainingMeta;
	}

	validateWeights(weights, callback) {

		this.trainingMeta.last_validation++;
		if (this.trainingMeta.last_validation === this.trainingMeta.validation_minibatches) {
			this.trainingMeta.last_validation = 0;
		}

		FS.readFile(this.testpath + this.trainingMeta.last_validation, (error, data) => {
			var len;
			if (error) throw error;
			else {
				// unpack training batch
				data = new Float32Array(data.buffer);
				len = data[0] * this.model.layers[0].shape[1]; // first float is number of samples in this batch

				this.load(weights);
				this.validate(data.subarray(1, ++len), data.subarray(len), callback); 
			}
		});
	}
};