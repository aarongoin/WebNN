const FS = require('fs')
const DumbClient = require('./DumbClient')

module.exports = class ModelServer {
    constructor(modelPath) {
        // open model for training
        this.JSON_model = FS.readFileSync(modelPath + '/model.json', 'utf8')
        this.model = JSON.parse(this.JSON_model)
        // this.meta = FS.readFileSync(modelPath + '/')
        // hold current weights in memory--if no weights then generate them
        if (FS.existsSync(modelPath + '/weights')) {
            let origPoolSize = Buffer.poolSize;
            Buffer.poolSize = modelSize * 4
            this.currentWeights = FS.readFileSync(modelPath + '/weights')
            Buffer.poolSize = origPoolSize
        } else {
            this.currentWeights = new Float32Array(this.model.size)
        }
        // open weights merger
        this.merger = new Merger()
        // open model validator
        this.validator = new Validator()
        // load all training data in to memory
        for (let i = 0; i < trainingMeta.training_minibatches; i++) {
            availableBatches.push(i)
            trainingBatches.push(FS.readFileSync(`./models/${train.model}/data/${i}`))
        }
        this.data = []
        this.training = []
    }

    getWeights() {
        // return current weights
        return this.currentWeights
    }

    putWeights(weights, callback) {
        // merge weights
        this.merger.merge(weights, scale, () => {
            this.currentWeights = this.merger.weights.reaad().data
            callback()
        })
    }

    getData() {
        // data: <Buffer> [ batchSize, batchX, batchY ]
        // return batch of data

    }

    saveModel() {
        // save model to disk
    }

    getClient() {
        // create dumb client if need be, and return it
        if (!this.dumb_client) {
            this.dumb_client = DumbClient(this.modelPath, this.JSON_model)
        }
        return this.dumb_client
    }
}