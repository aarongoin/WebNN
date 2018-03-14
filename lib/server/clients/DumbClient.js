const PS = require('child_process');
// create dumb client that will train the specified model
module.exports = function(modelName, model) {
return `<html>
    <head>
        <title>WebNN - ${modelName}</title>
    </head>
    <body style="margin: 0; padding: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; background: linear-gradient(to bottom, #f90, #600)">
        <span style="font-size: 128px; color: #fff;">WebNN</span>
        <script type="text/javascript">
        ${PS.execSync("browserify -t [ babelify --no-sourceMaps --presets [ env ] ] -e ./lib/server/clients/client.js | uglifyjs -c warnings=false -m", {encoding: 'utf8'}).replace(/"MODEL_HERE"/m, model)}
        </script>
    </body>
</html>`
}