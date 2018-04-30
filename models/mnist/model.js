module.exports = [
    { conv2d: { inputShape: [28, 28, 1], kernelSize: 5, filters: 8, strides: 1, activation: 'relu', kernelInitializer: 'varianceScaling'} },
    { maxPooling2d: {poolSize: [2, 2], strides: [2, 2]} },
    { conv2d: { kernelSize: 5, filters: 16, strides: 1, activation: 'relu', kernelInitializer: 'varianceScaling'} },
    { maxPooling2d: {poolSize: [2, 2], strides: [2, 2]} },
    { flatten: {} },
    { dense: { units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax' } },
    { output: { optimizer: 'sgd', loss: 'categoricalCrossentropy', metrics: ['accuracy'] } }
]
