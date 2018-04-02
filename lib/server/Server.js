const
HTTP = require("http"),
URL = require("url"),
BIND = require("./bind.js");

module.exports = class Server {
	constructor(port=8888, routes={}, verbose=true) {
		this.port = port;
		this.routes = routes;
		this.verbose = verbose;

		BIND(this);
	}

	start() {
		this.server = HTTP.createServer(this.handle);
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
		// var body = [];
		var body = '';

		if (request.method === "PUT") {
			request
				// .on("data", (chunk) => body.push(chunk))
				// .on("end", () => {
				// 	request.body = Buffer.concat(body);
				// 	this.route(request, response);
				// });
				.on('data', chunk => body += chunk)
				.on("end", () => {
					request.body = body;
					this.route(request, response);
				});

		} else
			this.route(request, response);
	}

	route(request, response) {
		const reqUrl = URL.parse(request.url)

		if (this.routes[reqUrl.pathname]
		&&  this.routes[reqUrl.pathname][request.method]) {

			this.routes[reqUrl.pathname][request.method](request, response, this.verbose);
		} else {
			response.writeHead(404);
			response.end();
		}
	}
}