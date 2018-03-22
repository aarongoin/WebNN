const
FS = require("fs"),
SERVER = require("./Server"),
Pako = require('pako'),
ModelServer = require('./ModelServer');

function getRequestIP(request) {
    return (request.headers['x-forwarded-for'] && request.headers['x-forwarded-for'].split(',').pop()) || 
            request.connection.remoteAddress || 
            request.socket.remoteAddress || 
            request.connection.socket.remoteAddress;
};

function saveLog(path, client, request, method) {
    let now = new Date().toISOString();
    let then = new Date(request.headers['client-request-date']).toISOString();
    FS.appendFile(path.substring(4)+'log.csv', `${now},${client},${method},${then}\n`, 'utf8', (error) => (error ? console.log(error) : null));
}

module.exports = function(path, port) {
    var
    Server,
    Model;

    try {
        console.log('Preparing model...');
        Model = new ModelServer(path);
    } catch (error) {
        console.error(error.message);
    } finally {
        if (Model) {
            let routes = {
                '/data': {
                    // get training batch of data
                    GET: function (request, response, verbose) {
                        if (verbose) console.log('Client requested data.');
                        // get client ip address to use as identifier
                        let client_ip = getRequestIP(request);

                        // log request
                        saveLog(path, client_ip, request, 'get data');

                        response.writeHead(200, { 'Content-Type': 'arraybuffer' });

                        response.write( Model.getLearningRate() );
                        response.end( Model.getData() );
                    }
                },
                '/update': {
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
                    PUT: function (request, response, verbose) {
                        // get client ip address
                        let client_ip = getRequestIP(request);

                        // log request
                        saveLog(path, client_ip, request, 'put weights');

                        if (verbose) console.log('Recieved weights from client @ ' + client_ip);

                        response.writeHead(200, { 'Content-Type': 'arraybuffer' });
                        
                        response.end( Model.putWeights(client_ip, request.body) );
                    }
                },
                '/loss': {
                    PUT: function(request, response) {
                        response.writeHead(200);
                        response.end();

                        // TODO
                    }
                },
                '/client': {
                    // return dummy client
                    GET: function (request, response, verbose) {
                        if (verbose) console.log('Training client requested.');

                        response.writeHead(200, { 'Content-Type': 'text/html', 'Content-Encoding': 'gzip' })
                        response.end( Model.getClient() );
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
                        response.writeHead(200, { 'Content-Type': 'text/html', 'Content-Encoding': 'gzip' })
                        response.end(Model.getWatcher());
                    }
                }
                
            };

            Server = new SERVER(port, routes, true);
            Server.start();
            Model.start();

            // I read that this doesn't work on Windows (but did not verify)
            process.on('SIGTERM', () => {
                Server.stop();
                Model.stop();
                console.log("Goodbye.");
                process.exit();
            });
        }
    }
}