var TF = require("../node_modules/tensorfire/src/index"),
    GL = TF.createGL(),
    Model = require("../lib/Model")(TF, GL);

function GET(path, responseType, callback) {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status === 200) {
			callback(r.response);
		}
	};
	r.open("GET", path);
	//r.setRequestHeader("Accept-Encoding", "gzip,deflate");
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
	//r.setRequestHeader("Accept-Encoding", "gzip, deflate");
	r.setRequestHeader("Content-Type", contentType);
	r.send(body);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*

	1. Setup model
	2. Get weights from server
	3. Get data from server
    4. Train model
    5. Send training log and updated weights to server

*/

var run = true,
    net = JSON.parse("MODEL_HERE"),
    model = new Model(net, null),
    iteration,
    times = {
        requested: null,	// model weights and data request sent
        received: null,		// model data & weights received from server
        loaded: null, 		// model weights loaded
        trained: null, 		// model finished training
        updated: null 		// model updates sent
    };

function onUpdate(arraybuffer) {
    times.received = window.performance.now()
    iteration = new Float32Array(arraybuffer, 0, 1)[0]
}

function onData(arraybuffer) {
    var view,
        data,
        len,
        i,
        batch;
}

function update(arraybuffer) {

    var iteration = new Float32Array(arraybuffer, 0, 1)[0],
        testView = new Float32Array(arraybuffer),
        view,
        weights,
        len,
        i,
        batch;

    //console.log(testView);
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
        log += iterations + ",";
        log += accuracy + ",";
        log += times.requested + ",";
        log += times.received + ",";
        log += times.loaded + ",";
        log += times.trained;
        // send time and training log to server
        PUT("./log/" + net.id, "text", log);
        times.requested = r;
        net.weights_version++;
    });
}

times.requested = window.performance.now();
GET("./weights/" + net.id, "arraybuffer", update);