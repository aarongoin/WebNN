
function GET(path, responseType, callback) {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status === 200) {
			callback(r.response);
		}
	};
	r.open("GET", path);
	r.responseType = responseType;
	r.send();
}

function PUT(path, contentType, body) {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status !== 200) {
			// TODO - resend or save to local?
		}
	}
	r.open("PUT", path);
	r.setRequestHeader("Content-Type", contentType);
	r.send(body);
}

function POST(path, contentType, body) {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status !== 200) {
			// TODO - resend or save to local?
		}
	}
	r.open("POST", path);
	if (contentType !== undefined)
		r.setRequestHeader("Content-Type", contentType);
	if (body !== undefined)
		r.send(body);
	else
		r.send();
}

(function main() {

	var chart;
	var randomColor = function() {
			return "rgb(" + (Math.random() * 256 >> 0) + "," + (Math.random() * 256 >> 0) + "," + (Math.random() * 256 >> 0) + ")";
		};

	var shouldUpdate = true;

	window.onbeforeunload = function() {
		shouldUpdate = false;
	}

	PUT("./train", "application/json", JSON.stringify({model: "mnist", version: "1"}));

	GET("./train", "application/json", function(logs) {

		var ctx = document.getElementById("chart").getContext("2d"),
			datasets = [];

		logs = JSON.parse(logs);
		console.log(logs);
		console.log("hello");

		for (var client in logs) {
			datasets.push({
				label: "Client " + client,
				backgroundColor: "rgba(0, 0, 0, 0)",
				borderColor: randomColor(),
				data: logs[client]
			});

		}


		chart = new Chart(ctx, {
			type: "scatter",
			data: { datasets: datasets },
			options: {
				title: {
					display: true,
					text: "Training Loss for MNIST Model",
					fontSize: 20,
					padding: 20
				},
				legend: {
					position: 'right'
				},
				elements: {
					line: {
						tension: 0, // disables bezier curves
					}
				}
			}
		});

		function update() {
			console.log("updating...");
			GET("./train", "application/json", function(logs) {
				logs = JSON.parse(logs);
				chart.data.datasets.forEach(function(dataset) {
					var id = dataset.label.split(" ")[1];
					if (logs[id] !== undefined) {
						Array.prototype.push.apply(dataset.data, logs[id]);
						delete logs[id];
					}
				});

				for (var client in logs) {
					console.log(client + " " + JSON.stringify(logs[client]));
					chart.data.datasets.push({
						label: "Client " + client,
						backgroundColor: "rgba(0, 0, 0, 0)",
						borderColor: randomColor(),
						data: logs[client]
					});
				}

				chart.update();
				if (shouldUpdate) update();
			});
		}

		update();
	});
})();