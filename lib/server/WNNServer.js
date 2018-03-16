const
FS = require("fs"),
SERVER = require("./Server"),
ZLIB = require('zlib'),
ModelServer = require('./ModelServer');

function getRequestIP(request) {
    return  request.headers['x-forwarded-for'].split(',').pop() || 
            request.connection.remoteAddress || 
            request.socket.remoteAddress || 
            request.connection.socket.remoteAddress;
};

function saveLog(client, request, method) {
    let now = new Date().toISOString();
    let then = new Date(request.getHeader('Date')).toISOString();
    FS.appendFile(path, `${now},${client},${method},${then}\n`, 'utf8', () => null);
}

var
path = process.argv[2],
Server,
Model;

try {
    console.log('Preparing model...');
    Model = new ModelServer(path);
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

                    // log request
                    saveLog(client_ip, request, 'get data');

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
                    saveLog(client_ip, request, 'get weights');

                    if (verbose) console.log('Sending weights to client @ ' + client_ip);

                    response.writeHead(200, { 'Content-Type': 'arraybuffer' });

                    response.end( ZLIB.gzipSync(Model.getWeights()) );
                },
                // recieve client weights and pass back weights for client to merge with
                PUT: function (request, response, verbose) {
                    // get client ip address
                    let client_ip = getRequestIP(request);

                    // log request
                    saveLog(client_ip, request, 'put weights');

                    if (verbose) console.log('Recieved weights from client @ ' + client_ip);

                    response.writeHead(200, { 'Content-Type': 'arraybuffer' });
                    
                    response.end( Model.putWeights(request.body) );
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
            }
        };

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