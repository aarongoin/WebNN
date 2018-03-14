const FS = require('fs');
const DumbClient = require('./clients/DumbClient');
const Watcher = require('./clients/Watcher');
const ZLIB = require('zlib');

module.exports = class ModelServer {
    constructor(modelPath) {
        // open model for training
        this.path = modelPath;
        this.model = require(modelPath + 'model.js');
        this.config = require(modelPath + 'config.js');

        this.hasWatcher = false;
        this.watcherData = {};

        // hold current weights in memory--if no weights then generate them
        if (FS.existsSync(modelPath + 'weights.bin')) {
            let origPoolSize = Buffer.poolSize;
            Buffer.poolSize = modelSize * ( this.config.byte_weights ? 1 : 4 );
            this.currentWeights = FS.readFileSync(modelPath + 'weights.bin');
            Buffer.poolSize = origPoolSize;
        } else {
            this.currentWeights = new global[this.config.byte_weights ? 'Int8Array' : 'Float32Array'](this.model.size);
        }
        // open model validator
        // this.validator = new Validator(this.config.validation.data_delegate);
        
        this.training = this.config.training.data_delegate || null;
        // if (useData) {
        //     this.data = [];
        //     this.training = [];

        //     // load all training data in to memory
        //     for (let i = 0; i < trainingMeta.training_minibatches; i++) {
        //         training.push(i);
        //         data.push( FS.readFileSync(`./models/${train.model}/data/${i}`) );
        //     }
        // }

        this.merging = {}; // holds weights from clients sent to another client to be merged
        this.toMerge = ZLIB.gzipSync(this.currentWeights); // most recent client update
        this.needsMerged = false;

    }

    getWeights() {
        // return current weights
        return this.currentWeights;
    }

    putWeights(client, weights) {
        // exchange weights
        let toMerge = this.toMerge;
        this.toMerge = weights;
        // keep hold of weights we're sending to this client to prevent data loss
        this.merging[client] = toMerge;
        this.needsMerged = true;
        return toMerge;
    }

    mergeAll() {
        // merge all weights to single set and set to currentWeights
        let type = this.config.byte_weights ? 'Int8Array' : 'Float32Array';
        let toMerge = new global[type]( ZLIB.gunzipSync(this.toMerge) );
        this.toMerge = null;
        for (let zWeights of this.merging) {
            if (zWeights) {
                let weights = new global[type]( ZLIB.gunzipSync(zWeights) );
                // average the weights (on the CPU)
                for (let i in weights) {
                    toMerge[i] += weights[i]; 
                    toMerge[i] *= 0.5;
                }
            }
        }
        this.currentWeights = toMerge;
        this.needsMerged = false;
    }

    getData() {
        // data: <Buffer> [ batchSize, batchX, batchY ]
        let data = ZLIB.gzipSync(this.training.getBatch(this.config.training.batch_size).buffer);
        return {
            meta: new Float32Array([
                this.config.training.learning_rate,
                data.length
            ]),
            data
        };
    }

    saveModel() {
        // save model to disk
        if (this.needsMerged) this.mergeAll();
        FS.writeFileSync(this.currentWeights.buffer, this.path + 'weights.bin');
    }

    getClient() {
        // create dumb client if need be, and return it
        if (!this.dumb_client) {
            let parsedPath = this.path.split('/');
            // pass dumb client the name of the model and the model description
            this.dumb_client = ZLIB.gzipSync( DumbClient( parsedPath[ parsedPath.length - 2 ], JSON.stringify(this.model) ) );
        }
        return this.dumb_client;
    }

    getWatcher() {
        return ZLIB.gzipSync( Watcher(this.path) );
    }

    getWatcherData() {
        return ZLIB.gzipSync();
    }

    stop() {
        // TODO: shutdown gracefully
        if (this.validationInterval)
            global.clearInterval(this.validationInterval);
        
        if (this.needsMerged)
            this.mergeAll();
        this.saveModel();
    }

    start() {
        // this.validationInterval = global.setInterval(this.validate.bind(this));
    }

    validate() {
        if (this.config.validation.merge_all) {
            if (this.needsMerged)
                this.mergeAll();
            this.validator.validateWeights(this.currentWeights);
            // TODO: queue merged weights for update
        } else
            this.validator.validateWeights( ZLIB.gunzipSync(this.toMerge) );
    }
}