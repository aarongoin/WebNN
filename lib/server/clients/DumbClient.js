const PS = require('child_process');
// create dumb client that will train the specified model
module.exports = function(modelName, model, byte_weights, byte_data) { // | uglifyjs -c warnings=false -m
    let body = PS.execSync("browserify -t [ babelify --no-sourceMaps --presets [ env ] ] -e ./lib/server/clients/client.js", { encoding: 'utf8' });

    body = body.replace(/'byte_weights_here'/gm, (byte_weights ? 'true' : 'false'));
    body = body.replace(/'byte_data_here'/gm, (byte_data ? 'true' : 'false'));
    body = body.replace(/'model_here'/gm, model);


return `<html>
    <head>
        <title>WebNN - ${modelName}</title>
    </head>
    <body style="margin: 0; padding: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; background: linear-gradient(to bottom, #f90, #600)">
        <span style="font-size: 128px; color: #fff;">WebNN</span>
        <script type="text/javascript">
        ${body}
        </script>
    </body>
</html>`
}