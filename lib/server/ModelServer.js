const FS = require('fs');
const Pako = require('pako');
const DumbClient = require('./clients/DumbClient');
const Watcher = require('./clients/WatcherClient');
const TemplateValidator = require('./TemplateValidator');
const Model = require('../nn/Model');

function DEFLATE(dataArray) {
    return Buffer.from(
        Pako.deflateRaw(
            new Uint8Array(dataArray.buffer)
        ).buffer
    );
}
function INFLATE(arraybuffer, type) {
    try {
        return new global[type](
            Pako.inflateRaw(
                new Uint8Array(arraybuffer)
            ).buffer
        );
    } catch (error) {
        console.trace(error);
    }
}

const validationAverage = {
    data: [],
    loss: 0,
    accuracy: 0
};

const updateRunningAverage = function (loss, accuracy) {
    // update running averages
    validationAverage.data.push({ loss, accuracy });
    if (validationAverage.data.length > 5)
        validationAverage.data.shift();

    validationAverage.loss = 0;
    validationAverage.accuracy = 0;

    for (let v of validationAverage.data) {
        validationAverage.loss += v.loss;
        validationAverage.accuracy += v.accuracy;
    }

    validationAverage.loss /= 5;
    validationAverage.accuracy /= 5;
}

module.exports = class ModelServer {
    constructor(modelName, modelPath, onDoneTraining) {

        // open model for training
        this.name = modelName;
        this.w_path = modelPath.substring(4);
        this.path = modelPath;
        this.model = require(modelPath + 'model.js');
        this.config = require(modelPath + 'config.js');

        this.counter = 0;
        this.isRunning = false;
        this.validateNext = false;
        this.learning_rate = this.config.training.learning_rate;

        this.dummyQueue = [];

        console.log('Validating model file...');
        // run through validation first to catch issues with setup
        TemplateValidator.model(this.model, modelPath.split('/')[3]);
        console.log('Validating config file...');
        TemplateValidator.config(this.config, modelPath.split('/')[3]);

        this.hasWatcher = false;
        this.watcherSaves = 0;
        this.watcherData = {
            validation: { loss: [], accuracy: [] }, loss: [], accuracy: [] };

        // hold current weights in memory--if no weights then generate them
        if (FS.existsSync(this.w_path + 'weights')) {
            console.log('Loading model weights...');
            let origPoolSize = Buffer.poolSize;
            Buffer.poolSize = modelSize(this.model) * ( this.config.byte_weights ? 1 : 4 );
            this.currentWeights = FS.readFileSync(this.w_path + 'weights');
            Buffer.poolSize = origPoolSize;

            this.toMerge = this.currentWeights;

        } else {
            console.log('Generating model weights...');
            (new Model(this.model)).save(weights => {
                this.currentWeights = this.toMerge = Buffer.concat([
                    Buffer.from(new Float32Array([1, 0]).buffer),
                    DEFLATE(weights)
                ]);
            });
        }
        console.log('Linking with model training delegate...');
        this.training = this.config.training.data_delegate || null;
        this.validation = this.config.validation.data_delegate;

        this.merging = {}; // holds weights from clients sent to another client to be merged
        this.clients = []; // holds keys for this.merging
        this.updateClients = [];
        this.toMerge = null; // most recent client update
        this.lastClient = null;
        this.needsMerged = false;

        this.afterValidation = this.afterValidation.bind(this);

        // handle configuration for ending training
        this.onDoneTraining = onDoneTraining;
        let end = this.config.training.end_condition;
        if (end > 0) {
            if (end < 1) { // end training when model accuracy exceeds the end_condition value
                this.endOnAccuracy = end;
            } else { // end training after end_condition seconds
                this.endOnTime = end;
                // time will begin from the very first client request
            }
        } // else training continues forever or until terminated from the command-line

        this.prerenderClients();
    }

    prerenderClients() {
        this.watcher_client = Buffer.from(
            Pako.gzip(
                Watcher(this.name, this.config.validation.validate_every)
            ).buffer
        );
        console.log('Watcher client ready.');

        // pass dumb client the name of the model and the model description
        this.dumb_client = Buffer.from(
            Pako.gzip(
                DumbClient(
                    this.name,
                    JSON.stringify(this.model),
                    this.config.training.learning_rate
                )
            ).buffer
        );
        console.log('Dumb client ready.');
    }

    getWeights() {
        return this.currentWeights;
    }

    putWeights(client_id, weights) {
        let client = null;
        
        // check if ignore weights and have client validate their weights
        if (this.validateNext) {
            this.validateNext = false;
            this.merging[client_id].weights = 'validating';

            // return current weights with step stripped out so client knows to validate
            if (this.config.validation.merge_all)
                return Buffer.concat([
                    Buffer.from(new Float32Array(1).buffer),
                    this.currentWeights.slice(4)
                ]);
            // return -1 to signal client to validate their own weights
            else
                return Buffer.from(new Float32Array([-1]).buffer);

        // check if client needs to update after merging all weights
        } else if (this.updateClients.length && (client = this.updateClients.indexOf(client_id)) > -1) {
            this.updateClients.splice(client, 1); // remove client from list

            this.merging[client_id].weights = this.currentWeights;

            return this.currentWeights;

        // check for special case where only one client is training or is much faster than it's peers
        } else if (this.clients.length === 1 || client_id === this.lastClient) {
            client = this.merging[client_id];
            this.currentWeights = this.toMerge = Buffer.concat([
                Buffer.from(new Float32Array([++client.step]).buffer),
                weights
            ]);
            client.weights = null;
            return Buffer.from(new Float32Array([-2]).buffer);
            
        } else {
            client = this.merging[client_id];

            this.lastClient = client_id;

            if (client.weights !== null) {
                // client has been sent weights to merge and has done so
                let step = new Float32Array(client.weights.buffer, 0, 1)[0];
                // if client merged with older set of weights, then bring their step up to match
                if (step > client.step)
                    client.step = step;
            }
            client.step++; // increment step because client has trained on data

            // keep client's weights, and return the ones we have for them to merge
            client.weights = this.toMerge;
            this.needsMerged = true;
            this.toMerge = Buffer.concat([
                Buffer.from(new Float32Array([client.step]).buffer),
                weights
            ]);

            return client.weights;
        }
    }

    mergeAll(clientsToMerge, save) {
        // merge all weights to single set and set to currentWeights
        let type = this.config.byte_weights ? 'Int8Array' : 'Float32Array';
        let step = new Float32Array(this.toMerge.buffer, 0, 1)[0];
        let maxStep = step;
        let toMerge = INFLATE(this.toMerge.slice(8), type);
        this.toMerge = null;
        
        if (this.clients.length === 1)
            return toMerge;
        
        let list = Object.keys(clientsToMerge);

        for (let id of list) {
            let client = clientsToMerge[id];
            if (client.weights && client.weights !== 'validating') {

                let s = new Float32Array(client.weights.buffer, 0, 1)[0];
                if (s > maxStep) maxStep = s;
                step += s;
                let weights = INFLATE(client.weights.slice(8), type);
                // average the weights
                for (let i in weights)
                    toMerge[i] = (toMerge[i] + weights[i]) * 0.5;
            }
        }
        this.toMerge = Buffer.concat([
            Buffer.from(new Float32Array([maxStep, 0.99]).buffer),
            DEFLATE(toMerge)
        ]);
        if (save)
            this.currentWeights = this.toMerge;
        this.needsMerged = false;
    }

    getData() {
        return DEFLATE(
            this.training.nextTrainBatch(this.config.training.batch_size)
        );
    }

    getValidationData() {
        return DEFLATE(
            this.validation.nextTestBatch(this.config.validation.batch_size)
        );
    }

    saveModel() {
        // save model to disk
        if (this.needsMerged) this.mergeAll(this.merging, true);
        FS.writeFileSync(this.w_path + 'weights', Buffer.from(this.currentWeights));
    }

    getClient(client_id) {
        // trigger timer if this is first client, and end_condition > 1
        if (this.clients.length === 0 && this.endOnTime !== undefined)
                global.setTimeout(this.onDoneTraining, this.endOnTime * 1000);
        
        // record if this is first time we've seen this client
        if (this.merging[client_id] === undefined) {
            this.clients.push(client_id);
            this.merging[client_id] = { weights: null, step: 0 };
        }

        // return dumb client
        return this.dumb_client;
    }

    onClientExit(client_id) {
        let i = this.clients.indexOf(client_id);
        if (i > -1) {
            this.clients.splice(i, 1);
            let removed = {};
            let client = removed[client_id] = this.merging[client_id];
            delete this.merging[client_id];
            // merge client's weights into anothers to prevent losing weights
            if (client.weights && typeof client.weights !== 'string') {
                this.mergeAll(removed, false);
            }
        }
    }

    getWatcher() {
        if (this.hasWatcher === false) {
            // TODO: read validation data from file
            this.hasWatcher = true;
        }
        return this.watcher_client;
    }

    endWatcher() {
        this.hasWatcher = false;
    }

    getWatcherData() {
        let data = Object.assign(
            this.watcherData,
            { 
                client_count: this.clients.length,
                lr: this.learning_rate,
                avg: { loss: validationAverage.loss, accuracy: validationAverage.accuracy },
                shouldEnd: this.shouldEndTraining
            }
        );
        this.watcherData = { validation: { loss: [], accuracy: [] }, loss: [], accuracy: [] };

        return Buffer.from(
            Pako.gzip(
                JSON.stringify(data)
            ).buffer
        );
    }

    stop() {
        // TODO: shutdown gracefully
        if (this.validationInterval)
            global.clearInterval(this.validationInterval);

        this.saveModel();
    }

    start() {
        this.validationInterval = global.setInterval(this.validate.bind(this), (this.config.validation.validate_every * 1000) << 0);
        this.isRunning = true;
    }

    onMetrics(client_id, arraybuffer, networkTime) { 
        let metrics = new Float32Array(arraybuffer); // metrics = [ loss, accuracy ]

        console.log(client_id);
        if (!this.merging[client_id]) console.trace(this.merging);

        // check if client is returning validation metrics or training metrics
        if (this.merging[client_id].weights === 'validating') {
            this.merging[client_id].weights = null;
            this.afterValidation(metrics[0], metrics[1], networkTime.now);
        } else {
            let [ loss, accuracy ] = metrics;
            this.watcherData.loss.push({
                x: networkTime.now,
                y: loss
            });
            this.watcherData.accuracy.push({
                x: networkTime.now,
                y: accuracy * 100
            });

            FS.appendFile(this.w_path + 'training.csv', `${networkTime.now},${client_id},${loss},${accuracy}\n`, 'utf8', (error) => (
                error ? console.log(error) : null
            ));
        }

        return Buffer.from(
            new Float32Array([ this.learning_rate ]).buffer
        );
    }

    validate() {
        if (this.config.validation.merge_all && this.clients.length > 1) {
            if (this.needsMerged)
                this.mergeAll(this.merging, true);

            // console.log('Validating w/ merge...');
            this.validateNext = true;
            
            // make list with clients that need updated weights
            this.updateClients = Object.assign([], this.clients);
            
        } else if (this.toMerge !== undefined) {
            // console.log('Validating w/o merge...');
            this.validateNext = true;
            this.currentWeights = this.toMerge;
        }
    }

    afterValidation(loss, accuracy, now) {

        updateRunningAverage(loss, accuracy);

        // adjust learning_rate
        if (this.config.training.learning_decay) {
            let lr = this.config.training.learning_rate * this.config.training.learning_rate * (1 - Math.pow(validationAverage.accuracy, validationAverage.loss * this.counter));
            if (lr > 0 && lr < this.learning_rate)
                this.learning_rate = lr;
        }

        // add new data point to our validation data
        this.watcherData.validation.accuracy.push({ x: this.counter, y: accuracy * 100 });
        this.watcherData.validation.loss.push({ x: this.counter++, y: loss });

        FS.appendFile(this.w_path + 'validation.csv', `${now},${loss},${accuracy}\n`, 'utf8', (error) => (
            error ? console.log(error) : null
        ));

        // finish training if accuracy exceeds the configured target
        if (this.endOnAccuracy !== undefined && this.endOnAccuracy < validationAverage.accuracy)
            this.shouldEndTraining = true;
        
        if (this.shouldEndTraining && !this.hasWatcher)
            this.onDoneTraining();
    }

    saveChart(name, data) {
        FS.writeFile(this.w_path + name, data, (error) => {
            if (error) console.trace(error);
            else if (++this.watcherSaves === 2) {
                this.onDoneTraining();
            }
        });
    }
}
