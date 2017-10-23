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
})();

var
refreshing = false,
sinceRefresh = null,
currentModel = {},
paths = {},

Routes = {

// MASTER ROUTES //////////////////////////////////////////////////////////////
	"/master": { // get master interface
		"GET": function(request, response, verbose) {
			FS.readFile('./master/master.html', 'utf8', function(error, data) {
				if (error) throw error;
				else {
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
			// TODO: save currentModel to disk if switching
			data = FS.readFileSync("./models/" + train.model + "/version/" + train.version + "/model.json", 'utf8');
				
			currentModel = JSON.parse(data);
			response.writeHead(200);
			response.end();
			sinceRefresh = null;
		},
		"GET": function(request, response, verbose) {
			// get logs for currentModel and send data to Master
			var log_directory = "./models/" + currentModel.model + "/version/" + currentModel.version + "/logs/";
			//console.log("sinceRefresh: " + sinceRefresh);
			if (sinceRefresh === null) FS.readdir(log_directory, function(error, files) {
				var length = files.length;
				refreshing = true;

				response.writeHead(200, {"Content-Type": "application/json"});

				if (length == 1) {
					response.write("{}");
					response.end();
					return;
				}

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
							response.write(JSON.stringify(sinceRefresh));
							response.end();
							sinceRefresh = {};
							refreshing = false;
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
			else if (!refreshing) {
				response.write(JSON.stringify(sinceRefresh));
				response.end();
				sinceRefresh = {};
			}
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
				iteration: currentModel.current_iteration,
				data: "./models/" + currentModel.model + "/data/",
				path: "./models/" + currentModel.model + "/version/" + currentModel.version + "/",
				log: "./models/" + currentModel.model + "/version/" + currentModel.version + "/logs/" + id
			};

			currentModel.id = id;
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
				if (error) throw error;
				else {
					// TODO: integrate data into model
					// TODO: send new batch data

					response.writeHead(200);
					response.end();
				}
			});
		}
	};

	Routes["/log/" + id] = {
		"PUT": function(request, response, verbose) {
			FS.appendFile(paths[id].path + "logs/" + id, request.body, function(error) {
				if (error) throw error;
				else {
					FS.appendFile(paths[id].path + "logs/" + id, "," + (new Date()).toISOString() + "\n");
					var line;
					response.writeHead(200);
					response.end();

					if (sinceRefresh !== null) {
						sinceRefresh[id] = sinceRefresh[id] || [];
						line = request.body.toString().toString().split(",");
						sinceRefresh[id].push({x: Number(line[0]), y: Number(line[1])})
					}
				}
			});
		}
	}
}

const Server = new SERVER(8888, Routes);
Server.start();

// I read that this doesn't work on Windows (but did not verify)
process.on('SIGINT', Server.stop);