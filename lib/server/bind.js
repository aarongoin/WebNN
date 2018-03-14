// convenience method
// binds every function in instance's prototype to the instance itself
module.exports = function(instance) {
	var proto = instance.constructor.prototype,
		keys = Object.getOwnPropertyNames(proto),
		key;
	while (key = keys.pop()) if (typeof proto[key] === 'function' && key !== 'constructor') instance[key] = instance[key].bind(instance);
}