var Layers = require("./Layers");

var Model = function(model, layers) {
	this.layers = new Array(model.layers.length);
	this.loss = 0.0;
	this.size = 0.0;
	this.model = model;
	this.load(layers);

	//console.log(JSON.stringify(this.layers[0].save()));
};
Model.prototype.run = function(input) {
	var output = input,
		l = -1;
	while (++l < this.layers.length)
		output = this.layers[l].run(output);
};
Model.prototype.forward = function(output) {
	//console.warn("Calculon- Forward pass\n");
	// forward propogation
	var l = -1;
	while (++l < this.layers.length) {
		output = this.layers[l].run(output);
		//console.log("Calculon- output " + l + ": " + output.read().data);
	}
	return output;
};
Model.prototype.backward = function(output, learn) {
	//console.warn("Calculon- Backward pass");
	// backward propogation
	var l = this.layers.length - 1;
	while (l-- > 0) {
		output = this.layers[l].train(output, learn);
	}
};

Model.prototype.validate = function(input, expect, callback) {
	var output = input,
		lossLayer = this.layers[this.layers.length - 1];
	output = this.forward(output);

	// calculate loss
	output = lossLayer.train(expect);
	if (typeof callback === "function") callback(lossLayer.accuracy)

}

Model.prototype.train = function(learn, iterations, input, expect, callback) {
	var output,
		e = 0,
		lossLayer = this.layers[this.layers.length - 1];
	while (e++ < iterations) {
		output = input;
		output = this.forward(output);

		//console.log("Calculon- output: " + output.read().data);
		// calculate loss
		output = lossLayer.train(expect);
		this.loss = lossLayer.accuracy;
		console.log("Accuracy: " + lossLayer.accuracy);

		this.backward(output, learn);

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
	while (++l < (this.layers.length - 1)) {
		weights.set( this.layers[l].save(), o);
		o += this.layers[l].size;
	}
	//console.log("weights: " + weights);
	return weights.buffer;
};
Model.prototype.load = function(layers) {
	// construct layers
	var offset = 0,
		layer,
		l = -1;


	this.size = 0;
	if (layers != null) {
		layers = new Float32Array(layers);
	}
	while (++l < (this.layers.length - 1)) {
		layer = this.model.layers[l];
		layer = new Layers[layer.type](layer, l);
		this.size += layer.size;
		if (layers != null)
			offset = layer.load(layers, offset);
		else layer.randomWeights();
		this.layers[l] = layer;
	}
	// initialize output layer
	layer = this.model.layers[l];
	layer = new Layers[layer.type](layer, l);
	this.layers[l] = layer;

};

module.exports = function(tensorfire, glContext) {
	Layers = Layers(tensorfire, glContext);
	return Model;
};