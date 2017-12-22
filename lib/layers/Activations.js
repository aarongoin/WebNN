module.exports = {
	Activation: {
		"linear": "o = n; \n",
		//"binary": "if (n > 0.0) { o = 0.0; } else { o = 1.0; } \n",
		"relu": "o = max(0.0, n); \n",
		"lrelu": "if (n > 0.0) { o = n; } else { o = 0.01 * n; } \n",
		"sigmoid": "o = 1.0 / (1.0 + exp(0.0 - n)); \n",
		"tanh": "o = (2.0 / (1.0 + exp(-2.0 * n))) - 1.0; \n",
		"softplus": "o = log(1.0 + exp(n)); \n",
		"softmax": "float k = 0.0; \nfor(int i = 0; i < #(O.shape).x; i++){\nk += exp(O.read(i, pos.y));\n}\no = exp(n) / k; \n",
		//"softsign": "o = n / (1.0 + abs(n)); \n"
	},
	Derivative: {
		"linear": "d = 1.0; \n",
		//"binary": "if (o == 0.0) { d = 0.0; } else { d = 0.0; } \n",
		"relu": "if (o >= 0.0) { d = 1.0; } else { d = 0.0; } \n",
		"lrelu": "if (o >= 0.0) { d = 1.0; } else { d = 0.01; } \n",
		"sigmoid": "d = o * ( 1.0 - o ); \n",
		"tanh": "d = ( 1.0 - pow(o, 2.0) ); \n",
		"softplus": "d = 1.0 - ( 1.0 / exp(o) ); \n",
		"softmax": "d = o * ( 1.0 - o ); \n" // same as sigmoid?
		//"softsign": "var = "
	}
};