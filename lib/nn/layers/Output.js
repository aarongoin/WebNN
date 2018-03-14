const
ndarray = require("ndarray"),
TF = (global && global.TF) || window.TF,
GL = (global && global.GL) || window.GL,
Output = require('./output/Output');

const Outputs = {
	'mean_squared_error': require('./output/MeanSquaredError'),
	'categorical_crossentropy': require('./output/CategoricalCrossEntropy'),
	'default': Output
};

module.exports = function(layer) {
	return new Outputs[layer.loss || 'default'](layer);
};