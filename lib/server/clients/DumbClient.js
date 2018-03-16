const PS = require('child_process');
// create dumb client that will train the specified model
module.exports = function(modelName, model, byte_weights, byte_data) {
    let body = PS.execSync("browserify -t [ babelify --no-sourceMaps --presets [ env ] ] -e ./lib/server/clients/client.js | uglifyjs -c warnings=false -m", { encoding: 'utf8' });
    let use_byte_weights = 'byte_weights = ' + (byte_weights ? 'true' : 'false') + ';';
    let use_byte_data = 'byte_data = ' + (byte_data ? 'true' : 'false') + ';';
    let model_insert = 'model = ' + model + ';';

return `<html>
    <head>
        <title>WebNN - ${modelName}</title>
    </head>
    <body style="margin: 0; padding: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; background: linear-gradient(to bottom, #f90, #600)">
        <span style="font-size: 128px; color: #fff;">WebNN</span>
        <script type="text/javascript">
        ${body}
        ${model_insert}
        ${use_byte_weights}
        ${use_byte_data}
        </script>
    </body>
</html>`
}