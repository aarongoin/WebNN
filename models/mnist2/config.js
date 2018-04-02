const dl = new (require ("./dataloader"))();
module.exports = {
    // using byte weights provides the following pros and cons:
    //      pros:
    //        - weights take up 75% less space which can be a BIG savings for network cost and transfer time
    //        - weights are kept clamped between -1 and 1, which will prevent weight saturation
    //        - weights are read the same regardless of system endianness -- a minor perk
    //      cons:
    //        - weights suffer a max precision error of 0.0078125, which could be rather awful for your model
    //        - weights must be converted twice every training iteration (from byte to float32 and from float32 to byte by each client)
    byte_weights: false,

    training: {
        // byte data provides the same pros and cons as byte weights except they need only be converted once
        // if you use byte_data your data delegates must implement the method: getByteBatch(size)
        byte_data: false,

        // batch size is the number of training samples per mini-batch sent to clients
        batch_size: 64,

        // if end_condition is 0, then model will train forever
        // if end_condition > 0 and end_condition < 1, then model will train until accuracy equals end_condition
        // if end_condition >= 1, then the model will train for end_condition seconds.
        end_condition: 0.99,

        // learning rate is how fast your model will adjust itself when it's wrong
        learning_rate: 0.01,

        // learning rate decay can help your model reach maximum accuracy, but too much can stop your model short
        // if 0, then the learning rate does not decay
        // else shrink the learning rate when learning begins to oscillate
        learning_decay: 0,

        // if you wish to train your model with an existing dataset, you must define the delegate that will provide data minibatches to be sent to dumb clients to train
        // your delegate must implement the method: getBatch(size) which should return the training data as a TypedArray
        // MiniBatch = [ N, input_set_0, ... input_set_N, output_set_0, ... output_set_N ], where N is number of training samples in minibatch, and each set is the in-order list of inputs or output
        data_delegate: dl

    },

    validation: {
        // can force all client weights to be merged into single set before validating
        // merged weights will be redistributed to every client to synchronize them
        merge_all: false,

        // how many seconds between model validations
        validate_every: 1,

        // you must define the delegate that will provide validation minibatches to the model validator
        // your delegate must implement the methods: getX() and getY() which should return the validation input and output data respectively as a Float32Array
        // MiniBatch = [ N, input_set_0, ... input_set_N, output_set_0, ... output_set_N ], where N is number of training samples in minibatch, and each set is the in-order list of inputs or output
        data_delegate: dl

    }
}
