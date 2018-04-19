const PS = require('child_process');
// create dumb client that will train the specified model
module.exports = function(modelName, model, learning_rate) { // | uglifyjs -c warnings=false -m
    let body = PS.execSync("node ./node_modules/browserify/bin/cmd -t [ babelify --no-sourceMaps --presets [ env ] ] -e ./lib/clients/client.js", { encoding: 'utf8' });

    body = body.replace(/'learning_rate_here'/gm, learning_rate);
    body = body.replace(/'model_here'/gm, model);


return `<html>
    <head>
        <title>WebNN - ${modelName}</title>
    </head>
    <body style="font-family: sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; background: linear-gradient(to bottom, #f90, #600)">
        <span style="font-size: 128px; color: #fff;">WebNN</span>
        <script type="text/javascript">
        ${body}
        </script>
    </body>
</html>`
}
