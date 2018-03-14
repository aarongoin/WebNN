function assertionFunction(filename) {
    return function(condition, result) {
        if (!result) throw Error('Error validating ' + filename + ' : ' + condition)
    }
}

function config(object, path) {
    const assert = assertionFunction(path);
    // TODO: validate config object
    if (object.validation) {
        assert('Validation requires a data delegate fulfilling DataSource protocol.',
            object.validation.data_delegate && 
            typeof object.validation.data_delegate.getBatch === 'function');

        assert('Validation must define frequency of validation using validate_every > 0.',
            typeof object.validation.validate_every === 'number' &&
            object.validation.validate_every > 0);

        if (object.validation.merge_all !== undefined)
            assert('Merge all must be true | false | undefined.',
                object.validation.force_sync === true
             || object.validation.force_sync === false);
    }
}

// used to validate a single hidden layer
function layer(object, assert, l) {

    assert('Layer ' + l + ': Layer must be a valid type.',
        object.dense !== undefined);

    assert('Layer ' + l + ': Layer must define a number of nodes out > 0.',
        typeof object.out === 'number' && object.out > 0);
    
    // checking activation functions
    if (object.dense) {
        assert('Layer ' + l + ': Layer must use valid activation function.',
            object.dense === 'sigmoid'
         || object.dense === 'tanh'
         || object.dense === 'relu'
         || object.dense === 'lrelu'
         || object.dense === 'linear');
    }

    if (object.dropout) {
        assert('Layer ' + l + ': Dropout must be > 0 and < 1.',
            object.dropout > 0 && 
            object.dropout < 1);
    }

    if (object.bias) {
        assert('Layer ' + l + ': Bias must be true | false | undefined.',
            object.validation.force_sync === true
         || object.validation.force_sync === false);
    }
}

function model(array, path) {
    const assert = assertionFunction(path);

    assert('Model must have at least one hidden layer.',
        array.length > 2);

    assert('First layer must be an input layer.',
        array[0].input !== undefined);

    assert('Input must define number of inputs > 0.',
        typeof array[0].input === 'number' && 
        array[0].input > 0);

    let last = array.length - 1;

    assert('Last layer must be the output layer',
        array[last].output !== undefined);

    assert('Last layer must define a number of outputs > 0.',
        typeof array[last].output === 'number' && 
        array[last].output > 0);

    assert('Last layer must define a loss function for validating the model',
        array[last].loss !== undefined);

    assert('Loss function must be a valid function',
        array[last].loss === 'mean_squared_error'
     || array[last].loss === 'softmax_crossentropy'
     || array[last].loss === 'none');
    
    assert('Last hidden layer and output layer must have same number of nodes.',
        array[last].output === array[last - 1].out);

    // validate hidden layers
    while (--last > 0)
        layer(array[last], assert, last);

}

module.exports = {
    config,
    model
};