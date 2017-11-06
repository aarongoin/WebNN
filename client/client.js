var Model = require("./Model");

function GET(path, responseType, callback) {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status === 200) {
			callback(r.response);
		}
	};
	r.open("GET", path);
	r.responseType = responseType;
	r.send();
}

function PUT(path, contentType, body, callback) {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status !== 200) {
			if (r.readyState === XMLHttpRequest.DONE && r.status === 200) {
				callback(r.response);
			}
		}
	}
	r.open("PUT", path);
	r.setRequestHeader("Content-Type", contentType);
	r.send(body);
}

function POST(path, contentType, body) {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status !== 200) {
			// TODO - resend or save to local?
		}
	}
	r.open("POST", path);
	if (contentType !== undefined)
		r.setRequestHeader("Content-Type", contentType);
	if (body !== undefined)
		r.send(body);
	else
		r.send();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*

	1. Get model from server
	2. Get weights from server
	3. Get data from server
	4. Train and return updates


*/

(function main() {
	var run = true,
		net,
		model;

	function Train(weights, batch) {
		var delta = 0;
		var e = net.log_rate;

		model = new Model(net, weights);

		model.afterIteration = function(model, iteration) {
			if (--e > 0) return;
			// send training logs to server
			PUT("./log/" + net.id, "text", ""+(net.current_iteration + iteration)+","+model.loss);
			e = net.log_rate;
			//console.log("Iteration: " + iteration + " Loss: " + model.loss);
		};

		delta = window.performance.now();
		model.train(net.learning_rate, net.iterations, batch.x, batch.y, function(model) {
			delta = window.performance.now() - delta;
			console.log("Time to train " + net.iteration + " iteration: " + (delta / 1000) + " seconds");
			// post results to server
			PUT("./weights/" + net.id, "arraybuffer", model.save());
			net.current_iteration++;
			update();
		});
	}

	function withModel(weights) {
		// request training data
		GET("./data/" + net.id, "arraybuffer", function(data) {

			// create Float32 view of arraybuffer
			var view = new Float32Array(data);

			// unpack training batch
			var len = view[0] * net.layers[0].shape[1], // first float is number of samples in this batch
				batch = {
					x: view.subarray(1, ++len),
					y: view.subarray(len)
				};

			Train(weights, batch);
		});
	}

	function update() {
		GET("./weights/" + net.id, "arraybuffer", withModel);
	}

	//var server = io();

	// request model to train
	GET("./model", "application/json", function(model) {
		net = JSON.parse(model);
		window.onbeforeunload = function() {
			POST("./close/" + net.id, "string")
		};
		

		if (net.get_weights) {
			// request model weights
			update();
		} else {
			// generate random weights
			withModel(null);
		}

		
	});
})();