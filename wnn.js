const HTTP = require('http');
const PS = require('child_process');
const FS = require('fs');
const WNNServer = require('./lib/server/WNNServer');

const helpText = `
usage:
    node wnn <command>

commands:
    new <name>                  - Create blank model
    reset <name>                - Resets model back to initialized stated
    copy <src_name> <dest_name> - Copies model to a new clean version ready for training
    save <src_name> <dest_name> - Copies model including all logs and weights
    train <name> <port>         - Runs training for model
    retrain <name> <port>       - Shortcut for reset <name> followed by train <name> <port>
    list                        - List all models
`;


function train(name, port) {
    const path = '../../models/' + name + '/';

    if (name) {
        console.log('Training model: ' + name + ' at port: ' + port);
        console.log('Dumb client at localhost:' + port + '/client');
        console.log('Watch model validation at localhost:' + port + '/watcher\n')
        // start serving model for training
        //PS.spawn('node', ['./lib/server/WNNServer.js', path, port], { detached: true });
        WNNServer(name, path, port);
    } else {
        console.error('Invalid command!');
        console.log(helpText)
    }
}

function reset(name) {
    const path = './models/' + name + '/';

    if (name) {
        console.log('Resetting model ' + name + ' at path: ' + path);
        // reset model for training
        PS.execFile('rm', ['-f', path + 'weights', path + 'log.csv', path + 'validation.csv']);
        PS.execFile('cp', ['./lib/templates/log.csv', './lib/templates/validation.csv', './lib/templates/training.csv', path]);
    } else {
        console.error('Invalid command!');
        console.log(helpText)
    }
}

function newModel(name) {
    const path = './models/' + name + '/';

    // TODO: check for existing name
    if (name) {
        console.log('Creating new model ' + name + ' at path: ' + path);
        // create new model with templates
        PS.execFile('mkdir', [path]);
        PS.execFile('cp', ['-R', './lib/templates/', path])
    } else {
        console.error('Invalid command!');
        console.log(helpText)
    }
}

function copy(original, copy) {
    const orig_path = './models/' + original + '/';
    const copy_path = './models/' + copy + '/';

    // TODO: check for existing name
    if (original && copy) {
        console.log('Copying ' + original + ' as model ' + copy + ' at path: ' + copy_path);
        // copy original 
        PS.execFile('mkdir', [copy_path]);
        PS.execFile('cp', [orig_path + 'model.js', orig_path + 'config.js', copy_path]);
        PS.execFile('cp', ['./lib/templates/log.csv', './lib/templates/validation.csv', './lib/templates/training.csv', copy_path]);
    } else {
        console.error('Invalid command!');
        console.log(helpText)
    }
}

function save(original, copy) {
    const orig_path = './models/' + original + '/.';
    const copy_path = './models/' + copy + '/';

    if (original && copy) {
        console.log('Saving ' + original + ' as model ' + copy + ' at path: ' + copy_path);
        // copy original 
        PS.execFile('mkdir', [copy_path]);
        PS.execFile('cp', ['-R', orig_path, copy_path]);
    } else {
        console.error('Invalid command!');
        console.log(helpText)
    }
}

function list() {
    const path = './models/';

    let models = FS.readdirSync(path);

    console.log('Models:\n');
    for (let item of models) {
        if (item.charAt(0) !== '.') {
            console.log(item);
        }
    }
}

console.log('');
const command = process.argv[2];
switch (command) {

    case 'train':
        train(process.argv[3], process.argv[4] || '8888');
        break;

    case 'retrain':
        reset(process.argv[3]);
        train(process.argv[3], process.argv[4] || 8888);
        break;

    case 'new':
        newModel(process.argv[3]);
        break;

    case 'reset':
        reset(process.argv[3]);
        break;

    case 'copy':
        copy(process.argv[3], process.argv[4]);
        break;
    
    case 'save':
        save(process.argv[3], process.argv[4]);
        break;

    case 'list':
        list();
        break;

    default: 
        console.error('Invalid command!');
        console.log(helpText)
        break;

}
console.log('');