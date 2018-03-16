const PS = require('child_process');
// create watcher client
module.exports = function(modelName, model) {
return `<html>
    <head>
        <title>WebNN - Watching ${modelName}</title>
    </head>
    <body style="margin: 0; padding: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; background: linear-gradient(to right, #f90, #600)">
        <div style="display: flex; flex-direction: column; justify-content: flex-start; align-items: center; width: 100vw; height: 100vh; background-color: #fff;">
            <div class="accuracy_chart" style="position: relative; height: 500; width: 800;"></div>
            <div class="loss_chart" style="position: relative; height: 500; width: 800;"></div>
		</div>
        <script type="text/javascript">
        ${PROCESS.execSync("browserify -t [ babelify --no-sourceMaps --presets [ env ] ] -e ./lib/server/clients/watcher.js | uglifyjs -c warnings=false -m", {encoding: 'utf8'})}
        </script>
    </body>
</html>`
}