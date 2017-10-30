var Layers = require("./Layers");

var Model = function(model, layers) {
	this.layers = [];
	this.loss = 0.0;
	this.size = 0.0;

	// construct layers
	var offset = 0,
		layer,
		l = -1;

	if (layers != null) {
		layers = new Float32Array(layers);
		console.log("Weights: " + layers.length);
	} else {
		console.log("Calculon- Generating random weights")
	}
	while (++l < model.layers.length) {
		layer = model.layers[l];
		layer = new Layers[layer.type](layer, l);
		this.size += layer.size;
		if (layers != null)
			offset = layer.load(layers, offset);
		else layer.randomWeights();
		this.layers.push( layer );	
	}

	//console.log(JSON.stringify(this.layers[0].save()));

	// construct loss layer
	this.lossLayer = new Layers[model.loss]([ layer.shape[1] ]);
}
Model.prototype.run = function(input) {
	var output = input,
		l = -1;
	while (++l < this.layers.length)
		output = this.layers[l].run(output);
}
Model.prototype.train = function(learn, iterations, input, expect, callback) {
	var output,
		e = 0,
		l;
	while (e++ < iterations) {
		output = input;
		//console.warn("Calculon- Iteration: " + e + ", Forward pass\n");
		// forward propogation
		l = -1;
		while (++l < this.layers.length) {
			output = this.layers[l].run(output);
			//console.log("Calculon- output " + l + ": " + output.read().data);
		}

		//console.log("Calculon- output: " + output.read().data);
		// calculate loss
		output = this.lossLayer.deltas(output, expect);
		this.loss = this.lossLayer.batchLoss

		//console.warn("Calculon- Iteration: " + e + ", Backward pass");
		// backward propogation
		l = this.layers.length;
		while (l-- > 0) {
			output = this.layers[l].train(output, learn);
		}
		// chance to send out data from model (metadata and log data)
		if (typeof this.afterIteration === "function") this.afterIteration(this, e);

		//console.warn("Calculon- Iteration: " + e + ", Loss: " + this.loss);
	}
	if (typeof callback === "function") callback(this);
}
Model.prototype.save = function() {
	// TypedArray to hold weights, bias, etc. from every layer of model
	var weights = new Float32Array(this.size);
	
	var l = -1,
		o = 0;
	// pull out trained weights for each layer
	while (++l < this.layers.length) {
		weights.set( this.layers[l].save(), o);
		o += this.layers[l].size;
	}
	console.log("weights: " + weights);
	return weights.buffer;
}

module.exports = Model;