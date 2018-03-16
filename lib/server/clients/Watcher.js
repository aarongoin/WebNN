var CanvasJS = require('canvasjs');

function PUT(path, contentType, callback) {
	var r = new XMLHttpRequest();
	r.onreadystatechange = function () {
		if (r.readyState === XMLHttpRequest.DONE && r.status !== 200) {
			callback(r.response);
		}
	}
	r.open("PUT", path);
	r.responseType = responseType;
	r.send();
}

function randomColor() {
    return "rgb(" + ((Math.random() * 240 >> 0) + 16) + "," + ((Math.random() * 240 >> 0) + 16) + "," + ((Math.random() * 240 >> 0) + 16) + ")";
};

var accuracyData = [];
// var lossData = [];

var accuracyChart = new CanvasJS.Chart('accuracy_chart', {
	title: { text: 'Model Accuracy' },
	axisY: {
		title: 'Accuracy (Percent)',
		valueFormatString: '00.00%',
		minimum: 0,
		maximum: 1,
		interval: 0.05
	},
	axisX: {
		title: 'Time (seconds)',
		minimum: 0,
		interval: 10,
		intervalType: 'second'
	},
	data: [{
		type: 'line',
		markerType: 'none',
		lineColor: 'black',
		dataPoints: accuracyData
	}]
});

// var lossChart = new CanvasJS.Chart('loss_chart', {
// 	title: { text: 'Client Loss' },
// 	axisY: {
// 		title: 'Loss',
// 		valueFormatString: '0.0000',
// 		minimum: 0,
// 		maximum: 1,
// 		interval: 0.05
// 	},
// 	axisX: {
// 		title: 'Time (seconds)',
// 		minimum: 0,
// 		interval: 10,
// 		intervalType: 'second'
// 	},
// 	data: [{
// 		type: 'scatter',
// 		markerType: 'none',
// 		lineColor: 'black',
// 		dataPoints: lossData
// 	}]
// });

function updateCharts(jsonString) {
	let data = JSON.parse(jsonString);
	
	for (let log of data.validation)
		accuracyData.push({ x: new Date(log.x), y: log.y });

	// for (log of data.loss)
	// 	lossData.push({ x: new Date(log.x), y: log.y });

	accuracyChart.render();
	// lossChart.render();

	PUT('./watch', 'application/json', updateCharts);
}
PUT('./watch', 'application/json', updateCharts);