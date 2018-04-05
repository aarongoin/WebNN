const PS = require('child_process');
// create watcher client
module.exports = function (modelName, validation_rate) { // add back after babelify: ' --no-sourceMaps'
    let body = PS.execSync("browserify -t [ babelify --presets [ env ] ] -e ./lib/server/clients/watcher.js", { encoding: 'utf8' });

    body = body.replace(/'validation_rate_here'/gm, validation_rate);

return `<html>
    <head>
        <title>WebNN - Watching ${modelName}</title>
    </head>
    <body style="margin: 0; padding: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; background: #fff;">
        <div style="display: flex; flex-direction: column; justify-content: flex-start; align-items: center; width: 800; background-color: #fff;">
            <canvas id="validation_chart" style="position: relative; maxHeight: 500; maxWidth: 800;"></canvas>
            <canvas id="client_chart" style="position: relative; maxHeight: 500; maxWidth: 800;"></canvas>
        </div>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.2/Chart.min.js"></script>
        <script type="text/javascript">
        ${body}
        </script>
    </body>
</html>`
}