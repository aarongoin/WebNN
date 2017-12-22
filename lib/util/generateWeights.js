// Standard Normal variate using Box-Muller transform.
function random(mean, stdDev) {
	mean = mean || 0;
	stdDev = stdDev || 1;
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    //return 0.4;
    return (Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v )) * stdDev + mean;
}

module.exports = function generateWeights(shape, bias) {
	var result = new Float32Array(shape[0] * shape[1] + bias);
	var l = -1;
	while (++l < result.length) {
		result[l] = random(0, Math.sqrt(2 / shape[1]));
	}
	//console.log(result[0]);
	return result;
}