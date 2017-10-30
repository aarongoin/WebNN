const
FS = require("fs"),
ndarray = require("ndarray"),
TF = require("tensorfire"),
GL = require("gl")(512, 512, { preserveDrawingBuffer: true });

module.exports = class Merger {
	constructor(path) {
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
			FS.readFile(path, this.load.bind(this));
		else {
			this.weights = null;
		}
	}

	merge(weights, scale, callback) {
		var newWeights = new TF.Tensor(GL, ndarray( weights, [ weights.length ]));
		console.log("weights length: " + weights.length + " scale: " + scale);
		if (this.weights !== null) this.weights.run(this.shader, {W: this.weights, N: newWeights, l: scale});
		else this.weights = new TF.InPlaceTensor(GL, ndarray(weights, [weights.length]));
	}

	load(error, data) {
		if (error) throw error;
		else {
			data = new Float32Array(data.buffer);
			//console.log("loading data: " + data.length);
			this.weights = new TF.InPlaceTensor(GL, ndarray(data, [data.length]));
		}
	}

	save() {
		FS.writeFile(this.path, Buffer.from(this.weights.read().data.buffer), "binary", function(error) { if (error) throw error; });
	}
};