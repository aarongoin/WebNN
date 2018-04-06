const PS = require('child_process');
// create watcher client
module.exports = function (modelName, validation_rate) { // add back after babelify: ' --no-sourceMaps'
    let body = PS.execSync("browserify -t [ babelify --presets [ env ] ] -e ./lib/server/clients/watcher.js", { encoding: 'utf8' });

    body = body.replace(/'validation_rate_here'/gm, validation_rate);

return `<html>
    <head>
        <title>WebNN - Watching ${modelName}</title>
    </head>
    <body style="font-family: sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; background: #fff;">
        <div style="display: flex; justify-content: space-around; align-items: center; background-color: #fff; width: 900; color: #666;">
            <div>
                <h4>Clients Training:</h4><h4 id="webnn_client_stat">0</h4>
            </div>
            <div>
                <h4>Learning Rate:</h4><h4 id="webnn_lr_stat">1</h4>
            </div>
            <div>
                <h4>Avg Loss:</h4><h4 id="webnn_loss_stat">5</h4>
            </div>
            <div>
                <h4>Avg Accuracy:</h4><h4 id="webnn_accuracy_stat">0</h4>
            </div>
        </div>
        <div style="display: flex; flex-direction: column; justify-content: flex-start; align-items: center; background-color: #fff;">
            <div style="width: 800; padding-left: 50;">
                <canvas id="validation_chart" style="position: relative; height: 500; width: 800;"></canvas>
            </div>
            <div style="width: 850;">
                <canvas id="client_chart" style="position: relative; height: 500; width: 900;"></canvas>
            </div>
        </div>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.7.2/Chart.min.js"></script>
        <script type="text/javascript">
        ${body}
        </script>
    </body>
</html>`
}