function getUpdate() {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status === 200) {
			updateCharts(r.response);
		}
	}
	r.open('GET', './watch');
	r.responseType = 'json';
	r.send();
}

function randomColor() {
    return "rgb(" + ((Math.random() * 240 >> 0) + 16) + "," + ((Math.random() * 240 >> 0) + 16) + "," + ((Math.random() * 240 >> 0) + 16) + ")";
};

let lossData = [];

let ctx = document.getElementById("accuracy_chart").getContext("2d");

let gradientFill = ctx.createLinearGradient(0,0,0,350);
gradientFill.addColorStop(0, '#f90');
gradientFill.addColorStop(1, '#600');

let accuracyChart = new Chart(ctx, {
	type: "scatter",
	data: {
		datasets: [
			{
				label: 'Validation',
				backgroundColor: "rgba(0,0,0,0)",
				borderColor: gradientFill,
				borderJoinStyle: "miter",
				borderWidth: 2,
				pointRadius: 0,
				data: [],
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
				type: 'linear',
				position: 'left',
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

// var accuracyChart = new CanvasJS.Chart('accuracy_chart', {
// 	title: { text: 'Model Accuracy' },
// 	axisY: {
// 		title: 'Accuracy (Percent)',
// 		valueFormatString: '00.00%',
// 		minimum: -0.05,
// 		maximum: 1,
// 		interval: 0.1,
// 		gridColor: '#ccc',
// 		gridThickness: 1
// 	},
// 	axisX: {
// 		title: 'Time (seconds)',
// 		minimum: 0,
// 		interval: 10,
// 		intervalType: 'second',
// 		gridColor: '#ccc',
// 		gridThickness: 1
// 	},
// 	data: [{
// 		type: 'line',
// 		markerType: 'none',
// 		lineColor: '#f90',
// 		dataPoints: accuracyData
// 	}]
// });
// accuracyChart.render();

// var lossChart = new CanvasJS.Chart('loss_chart', {
// 	title: { text: 'Client Loss' },
// 	axisY: {
// 		title: 'Loss',
// 		valueFormatString: '0.0000',
// 		minimum: 0,
// 		maximum: 1,
// 		interval: 0.1,
// 		gridColor: '#ccc',
// 		gridThickness: 1
// 	},
// 	axisX: {
// 		title: 'Time (seconds)',
// 		minimum: 0,
// 		interval: 10,
// 		intervalType: 'second',
// 		gridColor: '#ccc',
// 		gridThickness: 1
// 	},
// 	data: [{
// 		type: 'scatter',
// 		markerType: 'none',
// 		lineColor: 'black',
// 		dataPoints: lossData
// 	}]
// });
// lossChart.render();

function updateCharts(data) {
	if (data.validation.length)
		Array.prototype.push.apply(accuracyChart.data.datasets[0].data, data.validation)

	// for (log of data.loss)
	// 	lossData.push({ x: new Date(log.x), y: log.y });

	accuracyChart.update();
	// lossChart.render();
}
window.setInterval(getUpdate, 1000);