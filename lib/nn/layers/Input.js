const
ndarray = require("ndarray"),
TF = (global && global.TF) || window.TF,
GL = (global && global.GL) || window.GL;

class Input {
    constructor(layer) {
        this.layer = layer;
    }
}

module.exports = Input;