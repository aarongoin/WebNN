const
	HTTP = require("http"),
	URL = require("url"),
	BIND = require("./bind.js"),
	IO = require("socket.io");

var INSTANCE;

module.exports = class Server {
	constructor(port=8888, routes={}, verbose=true) {
		this.port = port;
		this.routes = routes;
		this.verbose = verbose;
		this.io = null;

		BIND(this);

		INSTANCE = this;
	}

	start() {
		this.server = HTTP.createServer(this.handle);
		this.io = IO(this.server);
		this.io.on("connection", function(socket) {
			console.log("Socket connected!");
			socket.on("disconnect", function() {
				console.log("Socket disconnected.");
			});
		});
		this.server.listen(this.port);
		if (this.verbose)
			console.log("Server started at port " + this.port + ".");
	}

	stop() {
		this.server.close();
		if (this.verbose)
			console.log("\nServer stopped.");
	}

	handle(request, response) {
		var reqUrl = URL.parse(request.url),
			body = [];

		if (this.verbose && reqUrl.pathname !== "/train") console.log("\n" + request.method + " Request for " + reqUrl.pathname + " recieved.");

		if (request.method === "PUT") {
			request.on("data", function(chunk) {
				body.push(chunk);
			}).on("end", function() {
				request.body = Buffer.concat(body);
				
				if (INSTANCE.routes[reqUrl.pathname] && INSTANCE.routes[reqUrl.pathname][request.method]) {
					INSTANCE.routes[reqUrl.pathname][request.method](request, response);
				} else {
					response.writeHead(404);
					response.end();
				}
			});

		} else {
			if (this.routes[reqUrl.pathname] && this.routes[reqUrl.pathname][request.method]) {

				this.routes[reqUrl.pathname][request.method](request, response, this.verbose);
			} else {
				response.writeHead(404);
				response.end();
			}
		}
	}
}