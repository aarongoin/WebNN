const
FS = require("fs"),
ndarray = require("ndarray"),
TF = require("tensorfire"),
GL = require("gl")(512, 512, { preserveDrawingBuffer: true });

module.exports = class Merger {
	constructor(path, validateEvery) {
		this.shader  = "uniform Tensor W; \n" /* weights */
					+ "uniform Tensor N; \n" /* new weights */
					+ "uniform float l; \n" /* learning rate */
					+ "float process(ivec4 pos) { \n" // pos in weights Tensor
						+ "float o = W.read(pos); \n"
						+ "float d = N.read(pos) - o; \n" /* difference between old and new */
						+ "d *= l; \n" /* scale update by rate */
						+ "return o + d; \n"
					+ "} \n"
					;

		this.path = path;

		if (FS.existsSync(path))
			this.load(null, FS.readFileSync(path));
		else this.weights = null;

		this.shouldValidate = false;
		this.validateEvery = validateEvery;
		this.merges = validateEvery - 1;
	}

	merge(weights, scale) {
		var newWeights = new TF.Tensor(GL, ndarray( weights, [ weights.length ]));
		if (this.weights !== null) this.weights.run(this.shader, {W: this.weights, N: newWeights, l: scale});
		else this.weights = new TF.InPlaceTensor(GL, ndarray(weights, [weights.length]));
		this.shouldValidate = false;
		if (++this.merges === this.validateEvery) {
			this.shouldValidate = true;
			this.merges = 0;
		}
	}

	load(error, data) {
		if (error) throw error;
		else {
			data = new Float32Array(data.buffer);
			this.weights = new TF.InPlaceTensor(GL, ndarray(data, [data.length]));
			
		}
	}

	save() {
		FS.writeFile(this.path, Buffer.from(this.weights.read().data.buffer), "binary", function(error) { if (error) throw error; });
	}
};