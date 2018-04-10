const validation_rate = 'validation_rate_here';

let updateLoop = null;

function getUpdate(alive = true) {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE){
			if (r.status === 200)
				updateCharts(r.response);
			else {
				window.clearInterval(updateLoop);
				console.log('Lost connection to server.');
			}
		}
	}
	r.open( alive ? 'GET' : 'PUT', './watch');
	r.responseType = 'json';
	r.send();
}

function randomColor() {
    return "rgb(" + ((Math.random() * 240 >> 0) + 16) + "," + ((Math.random() * 240 >> 0) + 16) + "," + ((Math.random() * 240 >> 0) + 16) + ")";
};

function gradientFrom(ctx, colorA = '#f90', colorB = '#600') {
	let gradientFill = ctx.createLinearGradient(0, 0, 0, 350);
	gradientFill.addColorStop(0, colorA);
	gradientFill.addColorStop(1, colorB);

	return gradientFill;
}

function initChart(withId, colorA = '#f90', colorB = '#600') {
	let ctx = document.getElementById(withId).getContext("2d");
	return { ctx, gradientFill: gradientFrom(ctx, colorA, colorB) };
}

let clientsElement = document.getElementById("webnn_client_stat");
let learningRateElement = document.getElementById("webnn_lr_stat");
let lossAvgElement = document.getElementById("webnn_loss_stat");
let accuracyAvgElement = document.getElementById("webnn_accuracy_stat");


let v_chart = initChart('validation_chart');
let v_reverseGradientFill = gradientFrom(v_chart.ctx, '#024', '#0ff');
let validationChart = new Chart(v_chart.ctx, {
	type: "scatter",
	data: {
		datasets: [
			{
				label: 'Accuracy',
				backgroundColor: "rgba(0,0,0,0)",
				borderColor: v_chart.gradientFill,
				borderJoinStyle: "miter",
				borderWidth: 2,
				pointRadius: 0,
				data: [],
				yAxisID: 'accuracy_axis',
				showLine: true
			},
			{
				label: 'Loss',
				backgroundColor: "rgba(0,0,0,0)",
				borderColor: v_reverseGradientFill,
				borderJoinStyle: "miter",
				borderWidth: 2,
				pointRadius: 0,
				data: [],
				yAxisID: 'loss_axis',
				showLine: true
			}
		]
	},
	options: {
		tooltip: {
			enabled: false
		},
		title: {
			display: true,
			text: "Model Validation",
			fontSize: 20,
			padding: 20
		},
		legend: {
			display: false
		},
		elements: {
			cubicInterpolationMode: "monotone",
			line: {
				tension: 0, // disables bezier curves
			}
		},
		scales: {
			xAxes: [{
				type: 'linear',
				position: 'bottom',
				scaleLabel: {
        			display: true,
        			labelString: 'Time (Seconds)'
      			}
			}],
			yAxes: [{
				id: 'loss_axis',
				type: 'linear',
				position: 'left',
				ticks: {
					min: 0,
					max: 5
				},
				scaleLabel: {
					display: true,
					labelString: 'Loss'
				}
			}, {
				id: 'accuracy_axis',
				type: 'linear',
				position: 'right',
				ticks: {
					min: 0,
					max: 100
				},
				scaleLabel: {
					display: true,
					labelString: 'Accuracy (Percent)'
				}
			}]
		}
	}
});

let c_chart = initChart('client_chart');
let c_reverseGradientFill = gradientFrom(c_chart.ctx, '#024', '#0ff');
let clientChart = new Chart(c_chart.ctx, {
	type: "scatter",
	data: {
		datasets: [
			{
				label: 'Accuracy',
				backgroundColor: "rgba(0,0,0,0)",
				borderColor: c_chart.gradientFill,
				borderJoinStyle: "miter",
				pointRadius: 1,
				data: [],
				yAxisID: 'accuracy_axis',
				showLine: false
			},
			{
				label: 'Loss',
				backgroundColor: "rgba(0,0,0,0)",
				borderColor: c_reverseGradientFill,
				borderJoinStyle: "miter",
				pointRadius: 1,
				data: [],
				yAxisID: 'loss_axis',
				showLine: false
			}
		]
	},
	options: {
		tooltip: {
			enabled: false
		},
		title: {
			display: true,
			text: "Client Metrics",
			fontSize: 20,
			padding: 20
		},
		legend: {
			display: false
		},
		elements: {
			showLine: false
		},
		scales: {
			xAxes: [{
				type: 'linear',
				position: 'bottom',
				scaleLabel: {
					display: true,
					labelString: 'Time (Seconds)'
				},
			}],
			yAxes: [{
				id: 'loss_axis',
				type: 'linear',
				position: 'left',
				ticks: {
					min: 0,
					max: 5
				},
				scaleLabel: {
					display: true,
					labelString: 'Loss'
				}
			}, {
				id: 'accuracy_axis',
				type: 'linear',
				position: 'right',
				ticks: {
					min: 0,
					max: 100
				},
				scaleLabel: {
					display: true,
					labelString: 'Accuracy (Percent)'
				}
			}]
		}
	}
});

var initTime; 

function toDate(v) {
	v.x = (new Date(v.x).getTime() / 1000) - initTime;
	return v;
}

function updateCharts(data) {

	if (data.loss.length || data.accuracy.length) {
		if (initTime === undefined) {
			initTime = (new Date(data.accuracy[0].x).getTime() / 1000);
		}
		if (data.loss.length)
			Array.prototype.push.apply(clientChart.data.datasets[1].data, data.loss.map(toDate));

		if (data.accuracy.length)
			Array.prototype.push.apply(clientChart.data.datasets[0].data, data.accuracy.map(toDate));

		clientChart.update();
	}

	if (data.validation.loss.length || data.validation.accuracy.length) {
		if (data.validation.loss.length)
			Array.prototype.push.apply(validationChart.data.datasets[1].data, data.validation.loss);

		if (data.validation.accuracy.length)
			Array.prototype.push.apply(validationChart.data.datasets[0].data, data.validation.accuracy);

		validationChart.update();
	}

	clientsElement.textContent = data.client_count;
	learningRateElement.textContent = data.lr;
	lossAvgElement.textContent = data.avg.loss;
	accuracyAvgElement.textContent = data.avg.accuracy;

	if (data.shouldEnd) {
		window.clearInterval(updateLoop);
		window.setTimeout(saveGraphs, 300, ['validation_chart', 'client_chart']);
	}
}

function saveImage(name, data) {
	var r = new XMLHttpRequest();
	r.open("PUT", '/charts', false);
	r.setRequestHeader("Content-Type", "imgage/png");
	r.setRequestHeader("Image-Name", name);
	r.send(data);
}

function saveGraphs(ids) {
	for (let id of ids) {
		let chart = document.getElementById(id);
		chart.toBlob(pictureBlob => {
			saveImage(id + '.png', pictureBlob);
		});
	}
}

window.onbeforeunload = function () {
	window.clearInterval(updateLoop);
}
updateLoop = window.setInterval(getUpdate, validation_rate * 1000);