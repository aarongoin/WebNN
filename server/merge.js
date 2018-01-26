const
FS = require("fs"),
ndarray = require("ndarray"),
TF = require("tensorfire"),
GL = require("gl")(512, 512, { preserveDrawingBuffer: true });

module.exports = class Merger {
	constructor(path, validateEvery, modelSize) {
		this.shader = `
			uniform Tensor W; // weights
			uniform Tensor N; // new weights
			uniform float l; // learning rate
			float process(ivec4 pos) { // pos in weights Tensor
				float o = W.read(pos);
				float d = N.read(pos) - o; // difference between old and new
				d *= l; // scale update by rate
				return o + d;
			}
		`;
		var origPoolSize;

		this.modelSize = modelSize;
		this.path = path;

		if (FS.existsSync(path)) {
			origPoolSize = Buffer.poolSize;
			Buffer.poolSize = modelSize * 4;
			this.load(null, FS.readFileSync(path));
			Buffer.poolSize = origPoolSize;
		} else this.weights = null;

		this.shouldValidate = false;
		this.validateEvery = validateEvery;
		this.merges = validateEvery - 1;
	}

	merge(weights, scale) {
		var newWeights = new TF.Tensor(GL, ndarray( weights, [ weights.length ]));
		//console.log("Merging weights, scale: " + scale);
		if (this.weights !== null) this.weights.run(this.shader, {W: this.weights, N: newWeights, l: scale});
		else this.weights = new TF.InPlaceTensor(GL, ndarray(weights, [weights.length]));
		this.check();
	}

	check() {
		this.shouldValidate = false;
		if (++this.merges === this.validateEvery) {
			this.shouldValidate = true;
			this.merges = 0;
		}
	}

	load(error, data) {
		if (error) throw error;
		else {
			//console.log(data);
			//console.log(data.buffer);
			data = new Float32Array(data.buffer);
			console.log(data);
			this.weights = new TF.InPlaceTensor(GL, ndarray(data, [data.length]));
		}
		this.check();
	}

	save() {
		FS.writeFile(this.path, Buffer.from(this.weights.read().data.buffer), "binary", function(error) { if (error) throw error; });
	}
};