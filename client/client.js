var Model = require("../lib/Model"),
	TF = require("../node_modules/tensorfire/src/index"),
	GL = TF.createGL();

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
		if (r.readyState === XMLHttpRequest.DONE && r.status === 200) {
			if (callback) callback(r.response);
		}
	}
	r.open("PUT", path);
	if (callback) r.responseType = contentType;
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
		model,
		iterations,
		times = {
			requested: null,	// model weights and data request sent
			received: null,		// model data & weights received from server
			loaded: null, 		// model weights loaded
			trained: null, 		// model finished training
			updated: null 		// model updates sent
		};

	Model = Model(TF, GL);

	function update(arraybuffer) {

		var iteration = new Float32Array(arraybuffer, 0, 1)[0],
			testView = new Float32Array(arraybuffer),
			view,
			weights,
			data,
			len,
			i,
			batch;

		console.log(testView);
		times.received = window.performance.now();

		view = new Float32Array(arraybuffer, 4);


		if (iteration >= 0) { // includes new weights and data
			iterations = iteration;
			i = model.size;
			weights = view.subarray(0, i);
			len = view[i] * net.layers[0].shape[1]; // first float is number of samples in this batch
			len += ++i;
			batch = {
				x: view.subarray(i, len),
				y: view.subarray(len)
			};

			model.load(weights);

		} else { // weights are fresh, so data only
			iterations++;
			len = view[0] * net.layers[0].shape[1]; // first float is number of samples in this batch
			batch = {
				x: view.subarray(1, ++len),
				y: view.subarray(len)
			};
		}

		// TRAIN
		times.loaded = window.performance.now();
		model.train(net.learning_rate, net.iterations, batch.x, batch.y, function(weights, accuracy) {
			var r = 0, log = "", w = new Float32Array(weights);
			times.trained = window.performance.now();
			//console.log("Time to train: " + (delta / 1000) + " seconds");
			// post results to server
			PUT("./weights/" + net.id, "arraybuffer", weights, update);
			r = window.performance.now();
			log += net.weights_version + ",";
			log += accuracy + ",";
			log += times.requested + ",";
			log += times.received + ",";
			log += times.loaded + ",";
			log += times.trained + "\n";
			// send time and training log to server
			PUT("./log/" + net.id, "text", log);
			times.requested = r;
			net.weights_version++;
		});
	}

	//var server = io();

	// request model to train
	GET("./model", "application/json", function(jsonModel) {
		net = JSON.parse(jsonModel);

		model = new Model(net, null);
		window.onbeforeunload = function() {
			POST("./close/" + net.id, "string")
		};
		times.requested = window.performance.now();
		GET("./weights/" + net.id, "arraybuffer", update);
	});
})();