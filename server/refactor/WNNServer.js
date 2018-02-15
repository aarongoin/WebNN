const
FS = require("fs"),
SERVER = require("./Server"),
READLINE = require("readline"),
CHILD_PROCESS = require('child_process'),
ZLIB = require('zlib'),
ModelServer = require('./ModelServer'),

getRequestIP = function(request) {
    return  request.headers['x-forwarded-for'].split(',').pop() || 
            request.connection.remoteAddress || 
            request.socket.remoteAddress || 
            request.connection.socket.remoteAddress;
};

var
Clients = {},
Models,
Server,
Routes = {
    "/online": {
        "GET": function(request, response, verbose) {
            response.writeHead(200);
			response.end();
        }
    },
    "/stop": {
		"GET": function(request, response, verbose) {
            response.writeHead(200);
            response.end();
			onStop();
		}
    },
    "/reboot": {
        "GET": function(request, response, verbose) {
            response.writeHead(200);
            response.end();
            stopModels();
            main();
        }
    }
};

function onStop() {
    Server.stop();
    console.log("Goodbye.");
	process.exit();
}

function stopModels() {
    for (let model of Models) {
        // TODO: stop Model process
    }
}

function main() {
    Models = {};

    // read model.conf file and create each model server
    if (FS.existsSync('./models.conf')) {
        let rl = READLINE.createInterface({
            input : FS.createReadStream('./models.conf'),
            terminal: false
        });

        rl.on('close', function() {
            if (Server) return;
            Server = new SERVER(8888, Routes, true);
            Server.start();
        });

        rl.on('line', function(line) {
            let base = '/' + line.trim();
            let header = {
                "Content-Type": "arraybuffer",
                "Content-Encoding": "gzip"
            }
            const Model = Models[base] = new ModelServer(base);
            Client[base] = {}
            
            // create model routes
            Routes[base + '/data'] = {
            // get training batch of data
                GET: function(request, response, verbose) {
                    response.writeHead(200, header)
                    response.end( ZLIB.gzipSync(Model.getData()) )
                }
            }
            
            Routes[base + '/update'] = {
            // get latest weights
                GET: function(request, response, verbose) {
                    // get client ip address to use as identifier
                    let client_ip = getRequestIP(request);

                    // save time of request
                    Client[base][client_ip] = new Date().toISOString()
                    
                    // send back model weights
                    response.writeHead(200, header)
                    response.end( ZLIB.gzipSync(Model.getWeights()) )
                },
            // recieve client weights and pass back updated weights
                PUT: function(request, response, verbose) {
                    // get client ip address
                    let client_ip = getRequestIP(request);

                    // write update to log.csv

                    // save time of request
                    Client[base][client_ip] = new Date().toISOString()

                    // send weights to model
                    Model.putWeights()

                    // send back model weights
                    response.writeHead(200, header)
                    response.end( ZLIB.gzipSync(Model.getWeights()) )
                }
            }
        });
    } else throw Error("ERROR! Missing file: ./models.conf");
}

main();

// I read that this doesn't work on Windows (but did not verify)
process.on('SIGINT', onStop);