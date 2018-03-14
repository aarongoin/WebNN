const PS = require('child_process');
// create watcher client
module.exports = function(modelName, model) {
return `<html>
    <head>
        <title>WebNN - Watching ${modelName}</title>
    </head>
    <body style="margin: 0; padding: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; background: linear-gradient(to right, #f90, #600)">
        <span style="font-size: 64px; color: #fff;">WebNN</span>
        <script type="text/javascript">
        ${PROCESS.execSync("browserify -t [ babelify --no-sourceMaps --presets [ env ] ] -e ./lib/server/clients/client.js | uglifyjs -c warnings=false -m", {encoding: 'utf8'}).replace(/"MODEL_HERE"/m, model)}
        </script>
    </body>
</html>`
}