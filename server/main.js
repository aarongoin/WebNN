const
FS = require("fs"),
SERVER = require("./Server"),
READLINE = require("readline"),
UUID = (function(){
	var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
				 + "abcdefghijklmnopqrstuvwxyz"
				 + "0123456789";
	return function(length) {
		var id = "";
		length = length || 5;
		while (length--) {
			id += possible.charAt(Math.random() * 62 >> 0);
		}
		return id;
	};
})(),
MERGER = require("./merge");
//fork = require("child-process").fork;

var
sinceRefresh = null,
currentModel = {},
modelPath = "",
num_clients = 0,
paths = {},
Merger = null,
longPoll = null,
shouldQuit = false,

refreshServer = function() {
	longPoll.writeHead(200, {"Content-Type": "application/json"});
	longPoll.write(JSON.stringify(sinceRefresh));
	longPoll.end();
	sinceRefresh = {};
	longPoll = null;
},

loadLogs = function() {
	// get logs for currentModel and send data to Master
	var log_directory = "./models/" + currentModel.model + "/version/" + currentModel.version + "/logs/";
	//console.log("sinceRefresh: " + sinceRefresh);
	FS.readdir(log_directory, function(error, files) {
		var length = files.length;

		if (length == 1) return;

		sinceRefresh = {};

		files.forEach(function(filename, index) {
			if (filename === ".DS_Store") {
				length--;
				return;
			}

			sinceRefresh[filename] = [];

			var rl, firstLine = true;

			rl = READLINE.createInterface({
				input : FS.createReadStream(log_directory + filename),
				terminal: false
			});

			rl.on('close', function() {
				length--;
				if (length === 0) {
					refreshServer();
				}
			});

			rl.on('line', function(line) {
				if (firstLine) {
					firstLine = false;
					return;
				}
				line = line.split(",");
				sinceRefresh[filename].push({x: Number(line[0]), y: Number(line[1])});
			});
		});
	});
},

Routes = {

// MASTER ROUTES //////////////////////////////////////////////////////////////
	"/master": { // get master interface
		"GET": function(request, response, verbose) {
			FS.readFile('./master/master.html', 'utf8', function(error, data) {
				if (error) throw error;
				else {
					sinceRefresh = null;
					response.writeHead(200, {"Content-Type": "text/html"});
					response.write(data);
					response.end();
				}
			});
		}
	},
	"/master.js": { // get master.js
		"GET": function(request, response, verbose) {
			FS.readFile('./master/master.js', 'utf8', function(error, data) {
				if (error) throw error;
				else {
					response.writeHead(200, {"Content-Type": "text/js"});
					response.write(data);
					response.end();
				}
			});
		}
	},
	"/train": { // request model to be the one trained
		"PUT": function(request, response, verbose) {
			var train = JSON.parse(request.body);
			modelPath = "./models/" + train.model + "/version/" + train.version + "/";
			// TODO: save currentModel to disk if switching
			data = FS.readFileSync(modelPath + "model.json", 'utf8');
			
			//if (Merger !== null) Merger.save();
			Merger = new MERGER(modelPath + "current");

			currentModel = JSON.parse(data);
			response.writeHead(200);
			response.end();
			sinceRefresh = null;
		},
		"GET": function(request, response, verbose) {
			if (sinceRefresh === null) loadLogs();
			longPoll = response;
		}
	},
	"/stop": {
		"PUT": function(request, response, verbose) {
			Merger.save();
			shouldQuit = true;
		}
	},

// CLIENT ROUTES //////////////////////////////////////////////////////////////
	"/": {
		"GET": function(request, response, verbose) {
			FS.readFile('./client/client.html', 'utf8', function(error, data) {
				if (error) throw error;
				else {
					response.writeHead(200, {"Content-Type": "text/html"});
					response.write(data);
					response.end();
				}
			});
		}
	},
	"/calculon.jpg": { // get image
		"GET": function(request, response, verbose) {
			FS.readFile('./client/calculon.jpg', function(error, data) {
				if (error) throw error;
				else {
					response.writeHead(200, {"Content-Type": "image/jpeg"});
					response.write(data);
					response.end();
				}
			});
		}
	},
	"/calculon.js": { // get client-side script
		"GET": function(request, response, verbose) {
			FS.readFile('./client/calculon.js', 'utf8', function(error, data) {
				if (error) throw error;
				else {
					response.writeHead(200, {"Content-Type": "text/js"});
					response.write(data);
					response.end();
				}
			});
		}
	},
	"/model": { // get the current model to train
		"GET": function(request, response, verbose) {
			
			var id = UUID();

			paths[id] = {
				root: currentModel.root,
				iteration: currentModel.current_iteration,
				data: "./models/" + currentModel.model + "/data/",
				path: "./models/" + currentModel.model + "/version/" + currentModel.version + "/",
				log: "./models/" + currentModel.model + "/version/" + currentModel.version + "/logs/" + id
			};

			currentModel.id = id;
			num_clients++;
			CreateRoutes(id, currentModel.model, currentModel.version);

			response.writeHead(200, {"Content-Type": "application/json"});
			response.write( JSON.stringify(currentModel) );
			response.end();
		}
	}
},

CreateRoutes = function(id) {

	Routes["/data/" + id] = { // get data for the model
		"GET": function(request, response, verbose) {
			// data: <Buffer> [ batchSize, batchX, batchY ]
			currentModel.last_batch++;
			if (currentModel.last_batch === currentModel.batches) {
				currentModel.last_batch = 0;
			}
			FS.readFile(paths[id].data + currentModel.last_batch, function(error, data) {
				if (error) throw error;
				else {
					response.writeHead(200, {"Content-Type": "arraybuffer"});
					response.write(data);
					response.end();
				}
			});
		}
	};

	Routes["/close/" + id] = { // done training
		"POST": function(request, response, verbose) {
			
			// remove routes
			delete Routes["/model/" + id];
			delete Routes["/log/" + id];
			delete Routes["/data/" + id];
			delete Routes["/close/" + id];

			num_clients--;
			paths[id] = undefined;

			response.writeHead(200);
			response.end();
		}
	};

	Routes["/weights/" + id] = {
		"GET": function(request, response, verbose) {
			// send weights for layers
			// data: <Buffer> [ ...layers ]
			FS.readFile(paths[id].path + "current", function(error, data) {
				if (error) throw error;
				else {
					response.writeHead(200, {"Content-Type": "arraybuffer"});
					response.write(data);
					response.end();
				}
			});
		},
		"PUT": function(request, response, verbose) {
			FS.appendFile(paths[id].path + "weights/" + id, request.body, function(error) {
				var staleness = (currentModel.root - paths[id].root) + 1; // staleness >= 1
				if (error) throw error;
				else {
					// integrate data into model
					Merger.merge(new Float32Array(new Uint8Array(request.body).buffer), ( 1 / (staleness * num_clients)));
					response.writeHead(200, {"Content-Type": "arraybuffer"});
					response.write(Buffer.from(Merger.weights.read().data.buffer));
					response.end();
					
					paths[id].root = ++currentModel.root;
					currentModel.current_iteration += currentModel.iterations;
					if (currentModel.get_weights == false) currentModel.get_weights = true;
					Merger.save();
				}
			});
		}
	};

	Routes["/log/" + id] = {
		"PUT": function(request, response, verbose) {
			FS.appendFile(paths[id].path + "logs/" + id, request.body, function(error) {
				if (error) throw error;
				else {
					FS.appendFile(paths[id].path + "logs/" + id, ( "," + (new Date()).toISOString() + "\n" ), function(error) { if (error) throw error; });
					var line;
					response.writeHead(200);
					response.end();

					if (sinceRefresh !== null) {
						sinceRefresh[id] = sinceRefresh[id] || [];
						line = request.body.toString().toString().split(",");
						sinceRefresh[id].push({x: Number(line[0]), y: Number(line[1])});
						if (longPoll !== null) refreshServer();
					}
				}
			});
		}
	}
},

onStop = function() {
	Server.stop();
	console.log("Goodbye.");
	FS.writeFileSync(modelPath + "model.json", JSON.stringify(currentModel), "utf8")
};



const Server = new SERVER(25749, Routes, true);
Server.start();


// I read that this doesn't work on Windows (but did not verify)
process.on('SIGINT', onStop);