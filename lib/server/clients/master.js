
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
			return "rgb(" + ((Math.random() * 240 >> 0) + 16) + "," + ((Math.random() * 240 >> 0) + 16) + "," + ((Math.random() * 240 >> 0) + 16) + ")";
		};

	var shouldUpdate = true;

	window.onbeforeunload = function() {
		shouldUpdate = false;
		PUT("./stop", "application/json", "");
	}

	PUT("./train", "application/json", JSON.stringify({model: "iris", version: "1"}));

	GET("./train", "application/json", function(logs) {

		var ctx = document.getElementById("chart").getContext("2d"),
			datasets = [];

		logs = JSON.parse(logs);

		for (var client in logs) {
			datasets.push({
				label: client,
				backgroundColor: "rgba(0,0,0,0)",
				borderColor: client == "view_at_scale" ? "rgba(0,0,0,0)" : randomColor(),
				borderJoinStyle: "miter",
				borderWidth: 2,
				pointRadius: 0,
				data: logs[client]
			});

		}


		chart = new Chart(ctx, {
			type: "scatter",
			data: { datasets: datasets },
			options: {
				title: {
					display: true,
					text: "Training Accuracy for IRIS Model",
					fontSize: 20,
					padding: 20
				},
				legend: {
					position: 'right'
				},
				elements: {
					cubicInterpolationMode: "monotone",
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
					if (logs[dataset.label] !== undefined) {
						Array.prototype.push.apply(dataset.data, logs[dataset.label]);
						delete logs[dataset.label];
					}
				});

				for (var client in logs) {
					console.log(client + " " + JSON.stringify(logs[client]));
					chart.data.datasets.push({
						label: client,
						backgroundColor: "rgba(0,0,0,0)",
						borderColor: client == "view_at_scale" ? "rgba(0,0,0,0)" : randomColor(),
						borderJoinStyle: "miter",
						borderWidth: 2,
						pointRadius: 0,
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