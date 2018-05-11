const
FS = require("fs"),
SERVER = require("./Server"),
ModelServer = require('./ModelServer');

var
Server,
Model,
metricsBuffer = new Uint8Array(8);

function endGracefully() {
    console.log("Stopping training.");
    Server.stop();
    Model.stop();
    console.log("Goodbye.");
    process.exit();
}

function getRequestIP(request) {
    return (request.headers['x-forwarded-for'] && request.headers['x-forwarded-for'].split(',').pop()) ||
            request.connection.remoteAddress ||
            request.socket.remoteAddress ||
            request.connection.socket.remoteAddress;
};

function networkTime(request) {
    return {
        now: new Date().toISOString(),
        then: new Date(request.headers['client-request-date']).toISOString()
    }
}

function saveLog(path, client, request, method) {
    let t = networkTime(request);
    request.networkTime = t;
    FS.appendFile(path.substring(1)+'log.csv', `${t.now},${client},${method},${t.then}\n`, 'utf8', (error) => (error ? console.log(error) : null));
}

module.exports = function(name, path, port) {
    try {
        console.log('Preparing model...');
        Model = new ModelServer(name, path, endGracefully);
    } catch (error) {
        console.error(error.message);
    } finally {
        if (Model) {
            let routes = {
                '/validate': {
                    // get validation batch of data
                    GET: function (request, response, verbose) {
                        if (verbose) console.log('Client requested validation data.');
                        // get client ip address to use as identifier
                        let client_ip = getRequestIP(request);

                        // log request
                        saveLog(path, client_ip, request, 'get data');

                        response.writeHead(200, { 'Content-Type': 'arraybuffer' });

                        response.end(Model.getValidationData());
                    },
                    POST: function (request, response, verbose) {

                        response.writeHead(200);
                        response.end();

                        // get client ip address
                        let client_ip = getRequestIP(request);

                        for (var i in metricsBuffer)
                            metricsBuffer[i] = request.body[i];

                        Model.onValidation(client_ip, metricsBuffer.buffer, networkTime(request));
                    }
                },
                '/data': {
                    // get training batch of data
                    GET: function (request, response, verbose) {
                        if (verbose) console.log('Client requested data.');
                        // get client ip address to use as identifier
                        let client_ip = getRequestIP(request);

                        // log request
                        saveLog(path, client_ip, request, 'get data');

                        response.writeHead(200, { 'Content-Type': 'arraybuffer' });

                        response.end( Model.getData() );
                    }
                },
                '/weights': {
                    // get latest weights
                    GET: function (request, response, verbose) {
                        // get client ip address to use as identifier
                        let client_ip = getRequestIP(request);

                        // log request
                        saveLog(path, client_ip, request, 'get weights');

                        if (verbose) console.log('Sending weights to client @ ' + client_ip);

                        response.writeHead(200, { 'Content-Type': 'arraybuffer' });

                        response.end( Model.getWeights() );
                    },
                    // recieve client weights and pass back weights for client to merge with
                    POST: function (request, response, verbose) {
                        // get client ip address
                        let client_ip = getRequestIP(request);

                        // log request
                        saveLog(path, client_ip, request, 'put weights');

                        if (verbose) console.log('Recieved weights from client @ ' + client_ip);

                        response.writeHead(200, { 'Content-Type': 'arraybuffer' });

                        response.end( 
                          Model.putWeights(client_ip, request.body)
                        );
                    }
                },
                '/metrics': {
                    POST: function(request, response) {

                        // get client ip address
                        let client_ip = getRequestIP(request);

                        for (var i in metricsBuffer)
                            metricsBuffer[i] = request.body[i];

                        // log request
                        // have to remove commas since output is a csv file and both datapoints are going into the same column
                        let strMetrics = new Float32Array(metricsBuffer.buffer).toString().replace(',', ' & ');
                        saveLog(path, client_ip, request, `training loss & accuracy: ${strMetrics}`);

                        response.writeHead(200, { 'Content-Type': 'arraybuffer' });
                        response.end(
                            Model.onMetrics(client_ip, metricsBuffer.buffer, request.networkTime)
                        );
                    }
                },
                '/client': {
                    // return dummy client
                    GET: function (request, response, verbose) {
                        if (verbose) console.log('Dumb client requested.');

                        response.writeHead(200, { 'Content-Type': 'text/html', 'Content-Encoding': 'gzip' })
                        response.end( Model.getClient() );
                    },
                },
                '/model': {
                    // return model client
                    GET: function (request, response, verbose) {
                        if (verbose) console.log('Model client requested.');

                        // get client ip address
                        let client_ip = getRequestIP(request);

                        response.writeHead(200, { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip' })
                        response.end( Model.getModel(client_ip) );
                        if (!Model.isRunning) Model.start();
                    },
                    POST: function (request, response, verbose) {
                        if (verbose) console.log('Client leaving.');

                        // get client ip address
                        let client_ip = getRequestIP(request);

                        Model.onClientExit(client_ip);
                    }
                },
                '/watch': {
                    // send updates
                    GET: function (request, response, verbose) {
                        if (verbose) console.log('Watcher requesting data.');
                        response.writeHead(200, { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip' });
                        response.end(Model.getWatcherData());
                        if (verbose) console.log('Watcher data sent.');
                    }
                },
                '/watcher': {
                    // request progress gui
                    GET: function (request, response, verbose) {
                        if (!Model.isRunning) Model.start();
                        
                        response.writeHead(200, { 'Content-Type': 'text/html', 'Content-Encoding': 'gzip' })
                        response.end(Model.getWatcher());
                    },
                    PUT: function (request, response, verbose) {
                        response.writeHead(200);
                        response.end();
                        Model.endWatcher();
                    }
                },
                '/charts': {
                    PUT: function (request, response, verbose) {
                        response.writeHead(200);
                        response.end();
                        Model.saveChart(request.headers['image-name'], request.body);
                    }
                }

            };

            Server = new SERVER(port, routes, false);
            Server.start();

            // I read that this doesn't work on Windows (but did not verify)
            process.on('SIGTERM', endGracefully);
        }
    }
}
