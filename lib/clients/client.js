const ModelClient = require("../ModelClient");

// initialize Model
const Model = new ModelClient(() => getTrainingData());

function getTrainingData() {
    Model.getData(trainModel);
}

function trainModel(data) {
    Model.train(data.x, data.y, data.n, getTrainingData);
}