module.exports = {
	Activation: {
		"linear": `
			o = n;
		`,
		// "binary": `
		// 	if (n > 0.0) { o = 0.0; } else { o = 1.0; }
		// `,
		"relu": `
			o = max(0.0, n);
		`,
		"lrelu": `
			if (n >= 0.0) { o = n; } else { o = 0.01 * n; }
		`,
		"sigmoid": `
			o = 1.0 / (1.0 + exp(0.0 - n));
		`,
		"tanh": `
			o = (2.0 / (1.0 + exp(-2.0 * n))) - 1.0;
		`,
		"softplus": `
			o = log(1.0 + exp(n));
		`,
		"softmax": `
			float k = 0.0;
			for(int i = 0; i < #(O.shape).x; i++){
				k += exp(O.read(i, pos.y));
			}
			o = exp(n) / k;
		`
	},
	Derivative: {
		"linear": `
			d = 1.0;
		`,
		// "binary": `
		// 	if (o == 0.0) {
		// 		d = 0.0;
		// 	} else {
		// 		d = 0.0;
		// 	}
		// `,
		"relu": `
			if (o >= 0.0) {
				d = 1.0;
			} else {
				d = 0.0;
			}
		`,
		"lrelu": `
			if (o >= 0.0) {
				d = 1.0;
			} else {
				d = 0.01;
			}
		`,
		"sigmoid": `
			d = o * ( 1.0 - o );
		`,
		"tanh": `
			d = ( 4.0 / pow(( exp(-o) + exp(o)), 2.0) );
		`,
		"softplus": `
			d = 1.0 - ( 1.0 / exp(o) );
		`,
		"softmax": `
			d = o * ( 1.0 - o );
		` 
	}
};