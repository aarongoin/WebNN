module.exports = [
    { input: 4 },
    { out: 10, dense: 'tanh', bias: true },
    { out: 3, dense: 'tanh', bias: true },
    { output: 3, loss: 'mean_squared_error' }
]