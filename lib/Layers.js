var Output = require('./layers/Output'),
	Dense = require('./layers/Dense');

module.exports = function(tensorfire, glContext) {
	return {
		"dense": Dense(tensorfire, glContext),
		"output": Output(tensorfire, glContext)
	};
};