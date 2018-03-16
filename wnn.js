const HTTP = require('http');
const PS = require('child_process');
const FS = require('fs');

const helpText = `
usage:
    node wnn <command>

commands:
    new <name>   -   Create blank model
    reset <name>   -   Resets model back to initialized stated
    copy <src_name> <dest_name> - Copies model to a new clean version.
    train <name> <port> - Runs training for model
`;

const command = process.argv[2];
switch (command) {

    case 'train': {
        const name = process.argv[3];
        const port = process.argv[4] || '8888';
        const path = './models/' + name + '/';

        if (name) {
            console.log('Training model: ' + name + ' at port: ' + port);
            console.log('Dumb client at localhost:' + port + '/client');
            console.log('Watch model validation at localhost:' + port + '/watch')
            // start serving model for training
            PS.execFile('node ./lib/server/WNNServer.js', [path, port]);
        } else {
            console.error('Invalid command!');
            console.log(helpText)
        }

        break;
    }

    case 'new': {
        const name = process.argv[3];
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
        break;
    }

    case 'reset': {
        const name = process.argv[3];
        const path = './models/' + name + '/';

        if (name) {
            console.log('Resetting model ' + name + ' at path: ' + path);
            // reset model for training
            PS.execFile('rm', ['-f', path + 'weights.bin', path + 'log.csv', path + 'validation.csv']);
            PS.execFile('cp', ['./lib/templates/log.csv', './lib/templates/validation.csv', path]);
        } else {
            console.error('Invalid command!');
            console.log(helpText)
        }
        break;
    }

    case 'copy': {
        const original = process.argv[3];
        const copy = process.argv[4];

        const orig_path = './models/' + original + '/';
        const copy_path = './models/' + copy + '/';

        // TODO: check for existing name
        if (original && copy) {
            console.log('Copying ' + original + ' as model ' + copy + ' at path: ' + copy_path);
            // copy original 
            PS.execFile('mkdir', [copy_path]);
            PS.execFile('cp', [orig_path + 'model.js', orig_path + 'config.js', copy_path]);
            PS.execFile('cp', ['./lib/templates/log.csv', './lib/templates/validation.csv', copy_path]);
        } else {
            console.error('Invalid command!');
            console.log(helpText)
        }
        break;
    }

    default: {
        console.error('Invalid command!');
        console.log(helpText)
    }
}