const PS = require('child_process');
// create watcher client
module.exports = function(modelName, model) {
return `<html>
    <head>
        <title>WebNN - Watching ${modelName}</title>
    </head>
    <body style="margin: 0; padding: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; background: #fff;">
        <div style="display: flex; flex-direction: column; justify-content: flex-start; align-items: center; width: 800; background-color: #fff;">
            <canvas id="accuracy_chart" style="position: relative; maxHeight: 500; maxWidth: 800;"></canvas>
            <div id="loss_chart" style="position: relative; height: 500; width: 800;"></div>
        </div>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.2/Chart.min.js"></script>
        <script type="text/javascript">
        ${PS.execSync("browserify -t [ babelify --no-sourceMaps --presets [ env ] ] -e ./lib/server/clients/watcher.js", {encoding: 'utf8'})}
        </script>
    </body>
</html>`
}