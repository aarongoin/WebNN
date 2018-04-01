// the layers of your model
// the first layer must be an input layer
// the last layer must be an output layer with loss defined (loss is optional if not training)
//      module.exports = [ { input: number_of_inputs }, { output: number_of_outputs, loss: loss_function_string } ] <- the bare minimum (and utterly useless) model
//
//      model loss is defined in the output layer
//          loss: 'mean_squared_error', 'softmax_cross_entropy'
//
//      hidden layers look like: { [type]: [activation], [bias: boolean,] out: num_output_nodes }
//          type: dense
//          activation: 'sigmoid', 'tanh', 'relu', 'lrelu', 'linear'
//          ex. Dense layer: { dense: 'tanh', bias: false, out: 100 } or even: { out: 20, dense: 'relu' }
//const IMAGE_SIZE = 784;
//const NUM_CLASSES = 10;
module.exports = [
    { conv2d: { inputShape: [28, 28, 1], kernelSize: 5, filters: 8, strides: 1, activation: 'relu', kernelInitializer: 'varianceScaling'} },
    { maxPooling2d: {poolSize: [2, 2], strides: [2, 2]} },
    { conv2d: { kernelSize: 5, filters: 16, strides: 1, activation: 'relu', kernelInitializer: 'varianceScaling'} },
    { maxPooling2d: {poolSize: [2, 2], strides: [2, 2]} },
    { flatten: 'flatten' },
    { dense: { units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax' } },
    { output: { optimizer: 'sgd', loss: 'categoricalCrossentropy', metrics: ['accuracy'] } }


]


//
// dropout inside layer? - { out: 30, dense: 'tanh', dropout: rate_zero_to_one }



// layer features:
//      bias
//      dropout
//
