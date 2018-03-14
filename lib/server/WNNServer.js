const
FS = require("fs"),
SERVER = require("./Server"),
READLINE = require("readline"),
CHILD_PROCESS = require('child_process'),
ZLIB = require('zlib'),
ModelServer = require('./ModelServer');

function getRequestIP(request) {
    return  request.headers['x-forwarded-for'].split(',').pop() || 
            request.connection.remoteAddress || 
            request.socket.remoteAddress || 
            request.connection.socket.remoteAddress;
};
var
Server,
Model,
Clients = {};

try {
    console.log('Preparing model...');
    Model = new ModelServer(process.argv[2]);
} catch (error) {
    console.error(error.message);
    Model.stop();
    Model = undefined;
} finally {
    if (Model) {
        let routes = {
            '/data': {
                // get training batch of data
                GET: function (request, response, verbose) {
                    if (verbose) console.log('Client requested data.');

                    response.writeHead(200, { 'Content-Type': 'arraybuffer', 'Content-Encoding': 'gzip' });
                    response.end(Model.getData());
                }
            },
            '/update': {
                // get latest weights
                GET: function (request, response, verbose) {
                    // get client ip address to use as identifier
                    let client_ip = getRequestIP(request);

                    if (verbose) console.log('Sending weights to client @ ' + client_ip);

                    // save time of request
                    Client[client_ip] = new Date().toISOString();

                    // send back model weights
                    response.writeHead(200, { 'Content-Type': 'arraybuffer' });
                    response.end(Model.getWeights());
                },
                // recieve client weights and pass back weights for client to merge with
                PUT: function (request, response, verbose) {
                    // get client ip address
                    let client_ip = getRequestIP(request);

                    if (verbose) console.log('Recieved weights from client @ ' + client_ip);

                    // write update to log.csv

                    // save time of request
                    Client[base][client_ip] = new Date().toISOString();

                    // send weights to model
                    let weights = Model.putWeights(request.body);

                    // send back model weights
                    response.writeHead(200, { 'Content-Type': 'arraybuffer' });
                    response.end(weights);
                }
            },
            '/client': {
                // return dummy client
                GET: function (request, response, verbose) {
                    if (verbose) console.log('Training client requested.');

                    response.writeHead(200, { 'Content-Type': 'text/html', 'Content-Encoding': 'gzip' })
                    response.end(Model.getClient());
                }
            }
        };

        if (process.argv[4]) {
            routes[process.argv[4]] = {
                // request progress gui
                GET: function (request, response, verbose) {
                    response.writeHead(200, { 'Content-Type': 'text/html', 'Content-Encoding': 'gzip' })
                    response.end(Model.getWatcher());
                },
                // send updates
                PUT: function (request, response, verbose) {
                    response.writeHead(200, { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip' })
                    response.end(Model.getWatcherData());
                }
            };
        }

        Server = new SERVER(parseInt(process.argv[3]), routes, true);
        Server.start();
        Model.start();

        // I read that this doesn't work on Windows (but did not verify)
        process.on('SIGINT', () => {
            Server.stop();
            Model.stop();
            console.log("Goodbye.");
            process.exit();
        });
    }
}